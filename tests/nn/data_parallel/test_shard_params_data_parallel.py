# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import itertools
import sys
import tempfile
from typing import Dict
import unittest
from unittest import mock

from parameterized import parameterized
import torch
from torch import nn

from fairscale.nn.data_parallel import ShardParamsDataParallel
from fairscale.utils.testing import DeviceAndTypeCheckModule, get_cycles_per_ms, objects_are_equal

# How to use remote-pdb: https://gist.github.com/sshleifer/9d43351957179c13606e015b072927d4


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

    @staticmethod
    def _train_for_several_steps(model, num_steps, autocast):
        model_device = next(model.parameters()).device
        optim = torch.optim.Adam(model.parameters(), lr=0.0001)
        # If you set this higher implem differs from ddp in the 5th decimal place
        for _ in range(num_steps):
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=autocast):
                # Inputs always cuda regardless of move_grads_cpu, or model.device
                input = model.module.get_input(torch.device("cuda"))
                output = model(*input)
                loss = model.module.get_loss(input, output).to(model_device)
            assert loss.dtype == torch.float32
            loss.backward()
            optim.step()
        return loss.detach()

    @staticmethod
    def get_wrapped_model(group, cuda_first=False, config={}, **model_kwargs) -> ShardParamsDataParallel:
        if cuda_first:
            model = ShardParamsDataParallel(TransformerWithSharedParams(**model_kwargs).cuda(), group, **config)
        else:
            model = ShardParamsDataParallel(TransformerWithSharedParams(**model_kwargs), group, **config).cuda()
        return model


class TestMixedPrecision(DistributedTest):
    def test_all_fp32(self):
        self._spawn_test_case(
            {"mixed_precision": False},
            False,  # autocast enabled
            torch.float32,  # expected_input_dtype
            torch.float32,  # expected_param_dtype
            torch.float32,  # expected_loss_dtype
            torch.float32,  # expected_reduce_dtype
        )

    def test_mixed_precision(self):
        self._spawn_test_case(
            {"mixed_precision": True},
            False,  # autocast enabled
            torch.float16,  # expected_input_dtype
            torch.float16,  # expected_param_dtype
            torch.float16,  # expected_loss_dtype
            torch.float16,  # expected_reduce_dtype
        )

    def test_mixed_precision_autocast(self):
        """If autocast enabled, loss should be fp32."""
        self._spawn_test_case(
            {"mixed_precision": True},
            True,  # autocast enabled
            torch.float16,  # expected_input_dtype
            torch.float16,  # expected_param_dtype
            torch.float32,  # expected_loss_dtype
            torch.float16,  # expected_reduce_dtype
        )

    def test_mixed_precision_autocast_fp32_compute(self):
        self._spawn_test_case(
            {"mixed_precision": True, "compute_dtype": torch.float32},
            True,  # autocast enabled
            torch.float16,  # expected_input_dtype
            torch.float32,  # expected_param_dtype
            torch.float32,  # expected_loss_dtype
            torch.float32,  # expected_reduce_dtype
        )

    def test_fp32_reduce_scatter(self):
        self._spawn_test_case(
            {"mixed_precision": True, "fp32_reduce_scatter": True},
            False,  # autocast enabled
            torch.float16,  # expected_input_dtype
            torch.float16,  # expected_param_dtype
            torch.float16,  # expected_loss_dtype
            torch.float32,  # expected_reduce_dtype
        )

    def test_fp32_reduce_scatter_autocast(self):
        self._spawn_test_case(
            {"mixed_precision": True, "fp32_reduce_scatter": True},
            True,  # autocast enabled
            torch.float16,  # expected_input_dtype
            torch.float16,  # expected_param_dtype
            torch.float32,  # expected_loss_dtype
            torch.float32,  # expected_reduce_dtype
        )

    def _spawn_test_case(self, cfg, autocast_enabled, in_dtype, p_dtype, loss_dtype, reduce_dtype, world_size=2):
        """Call test_dtypes inside of torch.multiprocessing.spawn"""
        fn = functools.partial(self._test_dtypes, cfg, autocast_enabled, in_dtype, p_dtype, loss_dtype, reduce_dtype)
        spawn_and_init(fn, world_size=world_size)

    @staticmethod
    def _test_dtypes(cfg: Dict, autocast, in_dtype, p_dtype, loss_dtype, reduce_dtype, rank, group):
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
            spawn_and_init(functools.partial(self._test_identical_outputs, TransformerWithSharedParams, config),)

    def test_cpu_offload_and_cpu_grads(self):
        # We only test True and None (which implies True). We don't test the
        # False condition because that requires the optimizer to internally do
        # the device transfer and PyTorch optimizers don't support this.
        for move_grads_choice in (True, None):
            config = {"mixed_precision": True, "cpu_offload": True, "move_grads_to_cpu": move_grads_choice}
            test_fn = functools.partial(
                self._test_identical_outputs, TransformerWithSharedParams, config, use_cuda=False
            )
            spawn_and_init(test_fn)

    def test_cpu_offload_and_cuda_grads(self):
        # If grads are on gpu, but model and optimizer are on cpu, backward breaks.
        config = {"mixed_precision": True, "cpu_offload": True, "move_grads_to_cpu": False}
        with self.assertRaises(Exception):  # RuntimeError inside spawn
            test_fn = functools.partial(
                self._test_identical_outputs, TransformerWithSharedParams, config, use_cuda=False
            )
            spawn_and_init(test_fn)

    def test_delayed_optim_step(self):
        # We use a model with a long CUDA delay right before the optimizer step.
        # This tests our streams logic, and that we don't start the FP32 -> FP16
        # transfer until after the optimization step completes.
        config = {"mixed_precision": True}
        test_fn = functools.partial(self._test_identical_outputs, self._delayed_optim_step_model, config)
        spawn_and_init(test_fn)

    @classmethod
    def _delayed_optim_step_model(cls, rank, group, config=None):
        def _maybe_wrap(layer):
            if config is not None:
                return ShardParamsDataParallel(layer, group, **config)
            return layer

        torch.manual_seed(0)  # keep everything deterministic
        model = nn.Sequential(
            nn.Linear(8, 4), _maybe_wrap(nn.Linear(4, 16)), _maybe_wrap(nn.Linear(16, 4)), nn.Linear(4, 8),
        )
        return ModuleWithDelay(model, delay_after_loss_ms=250)

    def test_delayed_reduce_scatter(self):
        # We insert a delay in the torch.distributed.reduce_scatter op, so that
        # the post_backward_stream takes much longer than the backward pass.
        # This tests that we properly block at the end of the backward pass for
        # the reductions to finish.
        config = {"mixed_precision": True}
        with mock.patch("torch.distributed.reduce_scatter", wraps=self._delayed_reduce_scatter):
            test_fn = functools.partial(self._test_identical_outputs, TransformerWithSharedParams, config)
        spawn_and_init(test_fn)

    @classmethod
    def _delayed_reduce_scatter(cls, *args, **kwargs):
        torch.cuda._sleep(int(250 * get_cycles_per_ms()))
        return torch.distributed.reduce_scatter(*args, **kwargs)

    @classmethod
    def _test_identical_outputs(cls, model_init_fn, config, rank, group, num_steps=3, use_cuda=True):
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
        model = model_init_fn(rank, group).cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, process_group=group)
        ref_loss = cls._train_for_several_steps(model, num_steps, autocast)
        ref_state_dict = model.module.state_dict()

        # Confirm we get the same behavior using ShardParamsDataParallel.
        model = ShardParamsDataParallel(model_init_fn(rank, group, config), group, **config)
        if use_cuda:
            model = model.cuda()
        else:
            assert next(model.parameters()).device == torch.device("cpu")
        shard_loss = cls._train_for_several_steps(model, num_steps, autocast)
        shard_state_dict = model.state_dict()

        try:
            torch.testing.assert_allclose(ref_loss, shard_loss)
            assert objects_are_equal(ref_state_dict, shard_state_dict, raise_exception=True)
        except (AssertionError, RuntimeError) as e:
            raise Exception(f"ShardParamsDataParallel didn't match PyTorch DDP using config: {config}" "\n\n{e}")


class TestSaveLoadLocalStateDict(DistributedTest):
    def test_load_local_state_dict(self):
        test_fn = functools.partial(self._load_local_and_train, {"flatten_parameters": False})
        spawn_and_init(test_fn)

    def test_local_state_dict_flatten_params_breaks(self):
        test_fn_broken = functools.partial(self._load_local_and_train, {"flatten_parameters": True})
        with self.assertRaises(Exception):
            spawn_and_init(test_fn_broken)
        # RuntimeError: Traceback [1]
        # [1] https://gist.github.com/sshleifer/612d8eb02dbbf357d6133b2700e02f5e

    def test_local_state_dict_odd_vocab_shape_breaks(self):
        test_fn = functools.partial(self._load_local_and_train, {"flatten_parameters": False}, d_model=16, d_vocab=37)
        with self.assertRaises(Exception):
            spawn_and_init(test_fn)

    @classmethod
    def _load_local_and_train(self, config, rank, group, d_model=32, d_vocab=32):
        """Check that local_state_dict can be saved and loaded for a given worker, and that training updates it"""
        model = ShardParamsDataParallel(
            TransformerWithSharedParams(d_model=d_model, d_vocab=d_vocab), group, **config
        ).cuda()
        state_1 = model.local_state_dict()
        state_before_training = {k: v.cpu().clone() for k, v in state_1.items()}
        model.load_local_state_dict(state_1)
        state_1_weight = state_1["embed_tokens.weight"]

        # This weight will be sharded since we access module.state_dict directly
        state_1_module_weight = model.module.state_dict()["embed_tokens.weight"]
        torch.testing.assert_allclose(state_1_weight, state_1_module_weight)
        torch.testing.assert_allclose(state_1_weight, model.module.embed_tokens.weight)
        self._train_for_several_steps(model, 4, False)

        state_2 = model.local_state_dict()
        state_after_training = {k: v.cpu().clone() for k, v in state_2.items()}
        model.load_local_state_dict(state_2)

        assert state_1.keys() == state_2.keys()

        # Assert that parameters were updated since before training
        unchanged = []
        for k in state_1:
            if (state_before_training[k] == state_after_training[k]).all():
                unchanged.append(k)
        if unchanged:
            raise AssertionError(f"params {unchanged} not changed after training")


class TestSaveLoadStateDict(DistributedTest):
    def test_calling_state_dict_twice_breaks(self):
        test_fn = functools.partial(self._test_calling_state_dict_twice_breaks, {"flatten_parameters": False})
        spawn_and_init(test_fn)

    @classmethod
    def _test_calling_state_dict_twice_breaks(self, config, rank, group):
        ddp_model = self.get_wrapped_model(group, cuda_first=False, config=config)
        self._train_for_several_steps(ddp_model, 1, False)
        ddp_model.state_dict()  # Succeeds
        try:
            ddp_model.state_dict()
            assert False, "Second state_dict call succeeded"
        except Exception:
            pass

    def test_state_dict_after_forward(self):
        test_fn = functools.partial(self._test_module_state_dict, {"flatten_parameters": False})
        spawn_and_init(test_fn)

    @classmethod
    def _test_module_state_dict(cls, config, rank, group):
        ddp_model = cls.get_wrapped_model(group, cuda_first=False, config=config)
        try:
            ddp_model.state_dict()
            assert False, "Calling state_dict before forward succeeded"
        except Exception:
            pass
        cls._train_for_several_steps(ddp_model, 2, False)
        state_1 = ddp_model.state_dict()
        # You must make a new ShardParamsDataParallel instance to use module.load_state_dict
        unwrapped_model = TransformerWithSharedParams()
        unwrapped_model.load_state_dict(state_1)
        new_ddp_model = ShardParamsDataParallel(unwrapped_model, group, **config).cuda()
        cls._train_for_several_steps(new_ddp_model, 2, False)
        try:
            ddp_model.load_state_dict(new_ddp_model.state_dict())
            assert False, "ddp_model.load_state_dict(new_ddp_model.state_dict()) succeeded"
        except Exception:
            pass


def get_sharded_model():
    sharded_model = ShardParamsDataParallel(
        nn.Sequential(
            nn.Linear(8, 100),
            ShardParamsDataParallel(nn.Linear(100, 100)),
            ShardParamsDataParallel(nn.Linear(100, 100)),
            nn.Linear(100, 8),
        )
    )
    return sharded_model


class TestHooks(DistributedTest):
    # Feel free to modify these tests as the implementation changes.
    # They aspire to make sure that backward hooks are registered and used

    @parameterized.expand([[True], [False]])
    def test_output_backward_hooks(self, cuda_first):
        fn = functools.partial(self._test_output_backward_hooks, cuda_first=cuda_first)
        spawn_and_init(fn)

    @classmethod
    def _test_output_backward_hooks(self, rank, group, cuda_first=False):
        model = self.get_wrapped_model(group, cuda_first=cuda_first)
        optim = torch.optim.Adam(model.parameters(), lr=0.0001)
        optim.zero_grad()
        # Inputs always cuda regardless of move_grads_cpu, or model.device
        input = model.module.get_input(torch.device("cuda"))
        output = model(*input)
        assert len(output._backward_hooks) == 1  # this is pre-bwd hook
        loss = model.module.get_loss(input, output).cuda()
        for p in model.params:
            assert p.grad is None  # because of pre_backward_hook
        loss.backward()
        assert len(output._backward_hooks) == 1  # It doesn't get removed
        optim.step()
        assert len(output._backward_hooks) == 1

    @parameterized.expand([[True], [False]])
    def test_register_functions_called(self, cuda_first):
        fn = functools.partial(self._test_register_functions_called, cuda_first=cuda_first)
        spawn_and_init(fn)

    @classmethod
    def _test_register_functions_called(self, rank, group, cuda_first=False):
        """Tests that _register_{pre|post}_backward_hooks called during forward."""
        model = self.get_wrapped_model(group, cuda_first=cuda_first)
        input = model.module.get_input(torch.device("cuda"))
        model._register_post_backward_hooks = mock.MagicMock(return_value=None)
        model._register_pre_backward_hooks = mock.MagicMock(return_value=None)
        assert not model._register_post_backward_hooks.called
        assert not model._register_pre_backward_hooks.called
        model(*input)
        assert model._register_post_backward_hooks.called
        assert model._register_pre_backward_hooks.called


class TransformerWithSharedParams(nn.Module):
    def __init__(self, *unused_args, d_vocab=32, d_model=16, **unused_kwargs):
        super().__init__()
        torch.manual_seed(0)  # keep everything deterministic
        self.embed_tokens = nn.Embedding(d_vocab, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=8, dropout=0.1,
        )
        self.output_proj = nn.Linear(d_model, d_vocab)
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


class ModuleWithDelay(nn.Module):
    def __init__(self, module, delay_after_loss_ms):
        super().__init__()
        self.module = module
        self.delay_after_loss_ms = delay_after_loss_ms

    def get_input(self, device):
        torch.manual_seed(1)  # keep everything deterministic
        return (torch.rand(4, 8, device=device),)

    def forward(self, x):
        return self.module(x)

    def get_loss(self, input, output):
        loss = output.sum()
        torch.cuda._sleep(int(self.delay_after_loss_ms * get_cycles_per_ms()))
        return loss


def spawn_and_init(fn, world_size=2, args=None):
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
