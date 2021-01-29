# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import itertools
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from fairscale.nn.data_parallel import ShardParamsDataParallel
from fairscale.utils.testing import DeviceAndTypeCheckModule, objects_are_equal


class DistributedTest(unittest.TestCase):
    def setUp(self):
        major, minor = torch.__version__.split(".")[:2]
        major, minor = int(major), int(minor)
        if major < 1 or minor < 6:
            raise unittest.SkipTest("Need pytorch version >= 1.6 due to autocast")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available, skipping test")
        if sys.platform == "win32":
            raise unittest.SkipTest("NCCL doesn't support Windows, skipping test")
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("distributed tests require 2+ GPUs, skipping")


class TestMixedPrecision(DistributedTest):
    def test_all_fp32(self):
        spawn_and_init(
            functools.partial(
                self.__class__._test_dtypes,
                {"mixed_precision": False},
                False,  # autocast enabled
                torch.float32,  # expected_input_dtype
                torch.float32,  # expected_param_dtype
                torch.float32,  # expected_loss_dtype
                torch.float32,  # expected_reduce_dtype
            ),
            world_size=2,
        )

    def test_mixed_precision(self):
        spawn_and_init(
            functools.partial(
                self.__class__._test_dtypes,
                {"mixed_precision": True},
                False,  # autocast enabled
                torch.float16,  # expected_input_dtype
                torch.float16,  # expected_param_dtype
                torch.float16,  # expected_loss_dtype
                torch.float16,  # expected_reduce_dtype
            ),
            world_size=2,
        )

    def test_mixed_precision_autocast(self):
        spawn_and_init(
            functools.partial(
                self.__class__._test_dtypes,
                {"mixed_precision": True},
                True,  # autocast enabled
                torch.float16,  # expected_input_dtype
                torch.float16,  # expected_param_dtype
                torch.float32,  # expected_loss_dtype
                torch.float16,  # expected_reduce_dtype
            ),
            world_size=2,
        )

    def test_mixed_precision_autocast_fp32_compute(self):
        spawn_and_init(
            functools.partial(
                self.__class__._test_dtypes,
                {"mixed_precision": True, "compute_dtype": torch.float32},
                True,  # autocast enabled
                torch.float16,  # expected_input_dtype
                torch.float32,  # expected_param_dtype
                torch.float32,  # expected_loss_dtype
                torch.float32,  # expected_reduce_dtype
            ),
            world_size=2,
        )

    def test_fp32_reduce_scatter(self):
        spawn_and_init(
            functools.partial(
                self.__class__._test_dtypes,
                {"mixed_precision": True, "fp32_reduce_scatter": True},
                False,  # autocast enabled
                torch.float16,  # expected_input_dtype
                torch.float16,  # expected_param_dtype
                torch.float16,  # expected_loss_dtype
                torch.float32,  # expected_reduce_dtype
            ),
            world_size=2,
        )

    def test_fp32_reduce_scatter_autocast(self):
        spawn_and_init(
            functools.partial(
                self.__class__._test_dtypes,
                {"mixed_precision": True, "fp32_reduce_scatter": True},
                True,  # autocast enabled
                torch.float16,  # expected_input_dtype
                torch.float16,  # expected_param_dtype
                torch.float32,  # expected_loss_dtype
                torch.float32,  # expected_reduce_dtype
            ),
            world_size=2,
        )

    @staticmethod
    def _test_dtypes(cfg, autocast, in_dtype, p_dtype, loss_dtype, reduce_dtype, rank, group):
        # Patch _reduce_scatter op to check the dtype of the reduction
        orig_reduce_scatter = ShardParamsDataParallel._reduce_scatter

        model = DeviceAndTypeCheckModule(
            expected_input_dtype=in_dtype, expected_param_dtype=p_dtype, expected_loss_dtype=loss_dtype,
        )

        def _reduce_scatter(self, tensor):
            model._check("reduce_scatter.dtype", tensor.dtype, expected=reduce_dtype)
            return orig_reduce_scatter(self, tensor)

        with mock.patch.object(ShardParamsDataParallel, "_reduce_scatter", new=_reduce_scatter):
            model = ShardParamsDataParallel(model, group, **cfg).cuda()
            device = next(model.parameters()).device
            x = torch.rand(2, 5).to(device)
            with torch.cuda.amp.autocast(enabled=autocast):
                loss = model(x)
            loss.backward()


class TestComparisonToPyTorchDDP(DistributedTest):
    """
    Compare losses and parameter values after several updates when using
    PyTorch DDP vs. ShardParamsDataParallel.
    """

    def test_transformer(self):
        # Test every combination of these options:
        keys = ["reshard_after_forward", "mixed_precision", "flatten_parameters"]
        for config in itertools.product([True, False], repeat=len(keys)):
            config = dict(zip(keys, config))
            spawn_and_init(
                functools.partial(self._test_identical_outputs, TransformerWithSharedParams, config), world_size=2,
            )

    @classmethod
    def _test_identical_outputs(cls, model_cls, config, rank, group, num_steps=3):
        if config["mixed_precision"]:
            autocast = True
            # Force the compute dtype to be torch.float32 so that we get
            # identical results as PyTorch DDP when using autocast. Note that
            # this will cause the all-gather to happen in FP32, which is slower
            # than necessary in most cases.
            config["compute_dtype"] = torch.float32
        else:
            autocast = False

        # Establish reference behavior with PyTorch DDP (+ optionally autocast).
        model = nn.parallel.DistributedDataParallel(
            model_cls().cuda(), device_ids=[rank], output_device=rank, process_group=group
        )
        ref_loss = cls._train_for_several_steps(model, num_steps, autocast)
        ref_state_dict = model.module.state_dict()

        # Confirm we get the same behavior using ShardParamsDataParallel.
        model = ShardParamsDataParallel(model_cls(), group, **config).cuda()
        shard_loss = cls._train_for_several_steps(model, num_steps, autocast)
        shard_state_dict = model.state_dict()

        try:
            torch.testing.assert_allclose(ref_loss, shard_loss)
            assert objects_are_equal(ref_state_dict, shard_state_dict, raise_exception=True)
        except (AssertionError, RuntimeError) as e:
            raise Exception(f"ShardParamsDataParallel didn't match PyTorch DDP using config: {config}" "\n\n{e}")

    @classmethod
    def _train_for_several_steps(cls, model, num_steps, autocast):
        optim = torch.optim.Adam(model.parameters(), lr=0.0001)
        for _ in range(num_steps):
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=autocast):
                device = next(model.parameters()).device
                input = model.module.get_input(device)
                output = model(*input)
                loss = model.module.get_loss(input, output)
            assert loss.dtype == torch.float32
            loss.backward()
            optim.step()
        return loss.detach()


class TransformerWithSharedParams(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)  # keep everything deterministic
        self.embed_tokens = nn.Embedding(50, 16)
        self.transformer = nn.Transformer(
            d_model=16, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=32, dropout=0.1,
        )
        self.output_proj = nn.Linear(16, 50)
        # share the embedding and output projection weights
        self.output_proj.weight = self.embed_tokens.weight

    def get_input(self, device):
        torch.manual_seed(1)  # keep everything deterministic
        src = torch.arange(12, device=device).view(6, 2)  # T x B
        tgt = torch.arange(8, device=device).view(4, 2)  # T x B
        return (src, tgt)

    def forward(self, src_ids, tgt_ids):
        src = self.embed_tokens(src_ids)
        tgt = self.embed_tokens(tgt_ids)
        x = self.transformer(src, tgt)
        return self.output_proj(x)

    def get_loss(self, input, output):
        _, tgt = input
        return nn.functional.cross_entropy(output.view(-1, output.size(-1)), tgt.view(-1), reduction="sum")


def spawn_and_init(fn, world_size, args=None):
    if args is None:
        args = ()
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        torch.multiprocessing.spawn(
            fn=functools.partial(init_and_run, fn, args),
            args=(world_size, tmp_file.name),
            nprocs=world_size,
            join=True,
        )


def distributed_init(rank, world_size, tmp_file):
    torch.distributed.init_process_group(
        backend="nccl", init_method="file://{}".format(tmp_file), world_size=world_size, rank=rank,
    )
    torch.cuda.set_device(rank)


def init_and_run(fn, args, rank, world_size, tmp_file):
    distributed_init(rank, world_size, tmp_file)
    group = torch.distributed.new_group()
    fn(rank, group, *args)


if __name__ == "__main__":
    unittest.main()
