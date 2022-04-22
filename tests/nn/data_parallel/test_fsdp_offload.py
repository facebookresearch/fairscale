# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
import itertools
import sys
import tempfile
import time
import unittest

from parameterized import parameterized
import pytest
import torch
from torch import nn
import torch.distributed

try:
    import fairscale.experimental.nn.ssd_offload as so
except ImportError as ie:
    # Note: We need the nightly version for SSD offload to work. Hence I am checking for the next PyTorch release.
    pytestmark = pytest.mark.skipif(True, reason=ie.msg)
    pass

from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from fairscale.nn.data_parallel import FullyShardedDataParallel, OffloadConfig, TrainingState
from fairscale.utils.testing import dist_init, spawn_for_all_world_sizes

# How to use remote-pdb: https://gist.github.com/sshleifer/9d43351957179c13606e015b072927d4
# All helper functions called by spawn must be either @classmethod, @staticmethod


class DistributedTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available, skipping test")
        if sys.platform == "win32":
            raise unittest.SkipTest("NCCL doesn't support Windows, skipping test")
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("distributed tests require 2+ GPUs, skipping")

    @staticmethod
    def _eval_with_config(model, autocast):
        model.eval()
        model_device = torch.device("cuda")
        with torch.cuda.amp.autocast(enabled=autocast):
            # Inputs always cuda regardless of move_grads_cpu, or model.device
            input = model.module.get_input(torch.device("cuda"))
            output = model(*input)
            loss = model.module.get_loss(input, output).to(model_device)
        assert loss.dtype == torch.float32
        if isinstance(model, FullyShardedDataParallel):
            model.assert_state(TrainingState.IDLE)
        return loss.detach()

    @staticmethod
    def _eval_for_several_steps(model, num_steps, autocast, lr=0.01, norm_type=None):
        model.eval()
        # Inputs always cuda regardless of move_grads_cpu, or model.device
        input = model.module.get_input(torch.device("cuda"))

        for _ in range(num_steps):
            with torch.cuda.amp.autocast(enabled=autocast):
                output = model(*input)

    @classmethod
    def _test_identical_outputs_eval(
        cls,
        model_init_fn,
        config,
        rank,
        group,
        num_steps=2,
        use_cuda=True,
        lr=0.01,
        ref_ddp_fn=None,
    ):
        if config.get("mixed_precision", False):
            autocast = True
            # Force the compute dtype to be torch.float32 so that we get
            # identical results as PyTorch DDP when using autocast. Note that
            # this will cause the all-gather to happen in FP32, which is slower
            # than necessary in most cases.
            config["compute_dtype"] = torch.float32
        else:
            autocast = False

        # Establish reference behavior with PyTorch DDP (+ optionally autocast).
        model = model_init_fn(group=group, wrapper_config=None).cuda()
        if ref_ddp_fn is None:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[rank], output_device=rank, process_group=group
            )
        else:
            model = ref_ddp_fn(model, group)
        ref_loss = cls._eval_with_config(model, autocast)
        ref_state_dict = model.module.state_dict()
        if config.get("cpu_offload", False):
            for k in ref_state_dict.keys():
                ref_state_dict[k] = ref_state_dict[k].cpu()

        # Confirm we get the same behavior using FullyShardedDataParallel.
        if config.get("ssd_offload", False):
            config["offload_config"] = OffloadConfig(offload_type="ssd_offload")
            # ssd offload only supports flatten_params ATM
            config["flatten_parameters"] = True

        del config["ssd_offload"]
        model = FullyShardedDataParallel(model_init_fn(group=group, wrapper_config=config), group, **config)
        if not model.ssd_offload and not model.move_params_to_cpu:
            if use_cuda:
                model = model.cuda()
            else:
                assert next(model.parameters()).device == torch.device("cpu")
        shard_loss = cls._eval_with_config(model, autocast)

        try:
            torch.testing.assert_allclose(ref_loss, shard_loss)
        except (AssertionError, RuntimeError) as e:
            raise Exception(f"FullyShardedDataParallel didn't match PyTorch DDP using config: {config}\n\n {e}")
        if config.get("flatten_parameters", True):
            metadata = model.local_metadata_dict()
            assert isinstance(metadata, dict)


keys = ["reshard_after_forward", "mixed_precision", "nested_wrapping"]
CONFIG_OPTIONS = [[dict(zip(keys, config))] for config in itertools.product([True, False], repeat=len(keys))]


def rename_test(testcase_func, param_num, param):
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name(str(param.args)),
    )


class TestSsdMemory(DistributedTest):
    def test_memory_benchmark(self):

        test_fn = functools.partial(self._test_memory_benchmark, config={})
        spawn_and_init(test_fn)

    @classmethod
    def _test_memory_benchmark(self, rank, group, config):
        time_keeper = TimeKeeper()

        SIZE = 8 * 8
        time_keeper.print_time("START", 1.0)
        a = torch.empty(1)
        b = a.cuda()
        # wait for cuda to fully load
        time.sleep(1)
        time_keeper.print_time("INIT_CUDA", 1.0)
        model = SimpleLinear(group, input_size=SIZE, output_size=SIZE, layers=4)
        time_keeper.print_time("CPU_MODEL", 1.0)

        with tempfile.TemporaryDirectory() as current_tempdir:
            config["offload_config"] = OffloadConfig(offload_type="ssd_offload", dir=current_tempdir)

            model = FullyShardedDataParallel(model, **config)
            time_keeper.print_time("FSDP_MODEL", 1.0)

            self._eval_for_several_steps(model, 1, autocast=False)
            time_keeper.print_time("EVAL")


class SimpleLinear(nn.Module):
    def __init__(self, group, input_size, output_size, layers=1, **unused_kwargs):
        super().__init__()
        self.rank = group.rank()
        self.world_size = group.size()
        self.input_size = input_size
        self.output_size = output_size
        torch.manual_seed(0)  # keep everything deterministic
        seq_layers = []
        for i in range(layers):
            seq_layers.append(nn.Linear(input_size, output_size, bias=False))
        self.module = nn.Sequential(*seq_layers)
        self.bs = 2

    def get_input(self, device):
        torch.manual_seed(1 + self.rank)  # keep everything deterministic
        src = torch.rand((self.bs, self.input_size), device=device, dtype=torch.float32)
        tgt = torch.rand((self.bs, self.input_size), device=device, dtype=torch.float32)
        return (src, tgt)

    def forward(self, src_ids, tgt_ids):
        param_devices = [p.device for p in self.module.parameters()]

        return self.module(src_ids)

    def get_loss(self, input, output):
        _, tgt = input

        return nn.functional.binary_cross_entropy_with_logits(output, tgt)

    def run_backward(self, loss):
        loss.backward()


KEYS = ["ssd_offload", "flatten_parameters", "mixed_precision", "move_params_to_cpu"]
CONFIG = [[dict(zip(KEYS, config))] for config in itertools.product([True, False], repeat=len(KEYS))]


class TimeKeeper:
    def __init__(self):
        self.start_time = time.time()

    def print_time(self, s: str, wait_time: float = 1.0):
        cur_time = time.time()
        print(f"@time: {cur_time - self.start_time:0.2f} {s}")
        time.sleep(wait_time)


class TestModuleProperties(DistributedTest):
    @parameterized.expand(CONFIG, name_func=rename_test)
    def test_named_parameters(self, config):

        test_fn = functools.partial(self._test_named_params, config=config)
        spawn_and_init(test_fn)

    @classmethod
    def _test_named_params(self, rank, group, config):
        # Get the named parameters before wrapping.
        before_wrap_model = TransformerWithSharedParams(group)
        before_wrap_params = before_wrap_model.named_parameters()

        with tempfile.TemporaryDirectory() as current_tempdir:
            if config["ssd_offload"]:
                config["offload_config"] = OffloadConfig(offload_type="ssd_offload", dir=current_tempdir)
                # ssd offload only supports flatten_params ATM
                config["flatten_parameters"] = True
            del config["ssd_offload"]

            model = FullyShardedDataParallel(before_wrap_model, **config)
            print(f"model.ssd_offload {model.ssd_offload}")
            if not model.ssd_offload and not model.move_params_to_cpu:
                model = model.cuda()

            self._eval_with_config(model, autocast=config["mixed_precision"])

            # Get the named parameters after wrapping to compare.
            after_wrap_params = model.named_parameters()

            if not config.get("flatten_parameters", False):
                for before_nm, after_nm in zip(before_wrap_params, after_wrap_params):
                    assert before_nm[0] == after_nm[0]
            else:
                named_params_flat = [p for p in after_wrap_params][0][0]
                assert "flat_param_0" in named_params_flat

            after_wrap_params = model.named_parameters()

            for before_nm, after_nm_original in zip(before_wrap_params, after_wrap_params):
                assert before_nm[0] == after_nm_original[0]
                torch.testing.assert_allclose(before_nm[1].shape, after_nm_original[1].shape)


class TestSsdLoading(DistributedTest):
    @parameterized.expand(CONFIG_OPTIONS, name_func=rename_test)
    def test_ssd_offloading_eval(self, config):

        test_fn = functools.partial(self._test_ssd_offload_eval, config=config)
        spawn_and_init(test_fn)

    @parameterized.expand(CONFIG, name_func=rename_test)
    def test_transformer_parameterized(self, config):

        spawn_and_init(functools.partial(self._test_identical_outputs_eval, TransformerWithSharedParams, config))

    @parameterized.expand(CONFIG_OPTIONS, name_func=rename_test)
    def test_ssd_offloading_train_flatten_params_wrapper(self, config):

        test_fn = functools.partial(self._test_ssd_offloading_train_flatten_params_wrapper, config=config)
        spawn_and_init(test_fn)

    @classmethod
    def _test_ssd_offloading_train_flatten_params_wrapper(self, rank, group, config):
        SIZE = 16 * 16
        LR = 0.01
        MOMENTUM = 0.1
        model = SimpleLinear(group, input_size=SIZE, output_size=SIZE, layers=4)

        with tempfile.TemporaryDirectory() as current_tempdir:
            config["offload_config"] = OffloadConfig(offload_type="ssd_offload", dir=current_tempdir)
            config["flatten_parameters"] = True

            nested_wrapping = config["nested_wrapping"]
            del config["nested_wrapping"]

            if nested_wrapping:
                model = FullyShardedDataParallel(
                    NestedWrappedModule(group, wrap_everything=True, wrapper_config=config)
                )
            else:
                model = FullyShardedDataParallel(model, **config)
            model_device = torch.device("cuda")
            model.train()
            optim = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

            checkpoint_file = tempfile.NamedTemporaryFile()
            checkpoint_load_directory = tempfile.TemporaryDirectory(prefix="checkpoint_dir")

            pre_checkpoint_last_output = None
            post_checkpoint_last_output = None

            ITERATIONS = 10

            # Inputs always cuda regardless of move_grads_cpu, or model.device
            with torch.cuda.amp.autocast(enabled=config.get("mixed_precision", False)):
                for i in range(ITERATIONS):
                    optim.zero_grad()
                    input = model.get_input(torch.device("cuda"))
                    output = model(*input)
                    pre_checkpoint_last_output = output
                    """
                    param_itr = iter(model.named_parameters())
                    p_name, p_val = next(param_itr)
                    print(f"i={i} pre_checkpoint {p_name} = {p_val[0].item()}")
                    """
                    loss = model.module.get_loss(input, output).to(model_device)
                    assert loss.dtype == torch.float32
                    model.module.run_backward(loss)
                    optim.step()
                    if i == 0:
                        with so.CheckpointPathContextManager(override_path=checkpoint_load_directory.name):
                            # so.torch_saver.save({"model": model.state_dict(), "optim": optim.state_dict()}, checkpoint_file.name)
                            torch.save({"model": model.state_dict()}, checkpoint_file.name)
                        # reset momentum just after checkpoint save
                        optim = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

                checkpoint = torch.load(checkpoint_file.name)
                model.load_state_dict(checkpoint["model"])
                # reset momentum just after checkpoint load
                optim = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
                # do more iterations after loading checkpoint
                for i in range(ITERATIONS - 1):
                    optim.zero_grad()
                    input = model.get_input(torch.device("cuda"))
                    output = model(*input)
                    post_checkpoint_last_output = output
                    """
                    param_itr = iter(model.named_parameters())
                    p_name, p_val = next(param_itr)
                    print(f"i={i} post_checkpoint {p_name} = {p_val[0].item()}")
                    """
                    loss = model.module.get_loss(input, output).to(model_device)
                    assert loss.dtype == torch.float32

                    model.module.run_backward(loss)
                    optim.step()

            # Verify output of checkpoint load + run is equal to original output
            assert torch.equal(pre_checkpoint_last_output, post_checkpoint_last_output)
            if isinstance(model, FullyShardedDataParallel):
                model.assert_state(TrainingState.IDLE)

    @classmethod
    def _test_ssd_offload_eval(self, rank, group, config):
        model = TransformerWithSharedParams(group)
        state_dict = model.state_dict()

        nested_wrapping = config["nested_wrapping"]
        del config["nested_wrapping"]
        config["flatten_parameters"] = True

        with tempfile.TemporaryDirectory() as current_tempdir:
            config["offload_config"] = OffloadConfig(offload_type="ssd_offload", dir=current_tempdir)
            if nested_wrapping:
                model = FullyShardedDataParallel(
                    NestedWrappedModule(group, wrap_everything=True, wrapper_config=config)
                )
            else:
                model = FullyShardedDataParallel(model, **config)

            self._eval_with_config(model, autocast=config["mixed_precision"])

            # With SSD offload only local_state_dict will work. We can support global
            # state dict if we think it is necessary.
            state_dict = model.local_state_dict()
            model.load_local_state_dict(state_dict)

            self._eval_with_config(model, config["mixed_precision"])


class TransformerWithSharedParams(nn.Module):
    def __init__(self, group, *unused_args, d_vocab=23, d_model=16, add_bn=True, **unused_kwargs):
        super().__init__()
        self.rank = group.rank()
        self.world_size = group.size()
        torch.manual_seed(0)  # keep everything deterministic
        assert d_vocab >= 12  # we use torch.arange(12) as input
        self.embed_tokens = nn.Embedding(d_vocab, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=8,
            dropout=0.1,
        )
        self.output_proj = nn.Linear(d_model, d_vocab)

        # share the embedding and output projection weights
        self.output_proj.weight = self.embed_tokens.weight
        self.register_buffer("vocab_bias", self.embed_tokens.weight.new_ones((d_model,)))
        self.register_buffer("long_buffer", torch.zeros_like(self.vocab_bias, dtype=torch.long))

        self.bs = 2
        self.bn = torch.nn.BatchNorm1d(self.bs) if add_bn else torch.nn.Identity()

    def get_input(self, device):
        torch.manual_seed(1 + self.rank)  # keep everything deterministic
        src = torch.arange(12, device=device).view(6, self.bs)  # T x B
        tgt = torch.arange(self.bs * 4, device=device).view(4, self.bs)  # T x B
        return (src, tgt)

    def forward(self, src_ids, tgt_ids):
        src = self.embed_tokens(src_ids)
        src = src + self.vocab_bias + self.long_buffer.type_as(src)
        tgt = self.embed_tokens(tgt_ids)
        tgt = self.bn(tgt)
        x = self.transformer(src, tgt)
        return self.output_proj(x)

    def get_loss(self, input, output):
        _, tgt = input
        return nn.functional.cross_entropy(output.view(-1, output.size(-1)), tgt.view(-1), reduction="sum")

    def run_backward(self, loss):
        loss.backward()


class NestedWrappedModule(nn.Module):
    def __init__(self, group, wrapper_config, wrap_everything=False, checkpoint=False):
        super().__init__()
        self.rank = group.rank()
        self.world_size = group.size()
        self.wrapper_config = wrapper_config

        def _maybe_wrap(layer):
            if wrapper_config is not None:
                return FullyShardedDataParallel(layer, group, **wrapper_config)
            return layer

        torch.manual_seed(0)  # keep everything deterministic
        self.module = nn.Sequential(
            nn.Linear(8, 4),
            _maybe_wrap(
                nn.Sequential(
                    _maybe_wrap(nn.Linear(4, 16)),
                    nn.Linear(16, 16),
                )
            ),
            _maybe_wrap(nn.Linear(16, 4)),
            nn.Linear(4, 8),
        )

        # Wrap all modules triggers a corner case where root FSDP doesn't have any params.
        # Test it with checkpoint_wrapper as well to validate final backward callback
        # is queued correctly when root FSDP does not have any params and every layer is
        # wrapped as FSDP(checkpoint(module)).
        if wrap_everything:
            if checkpoint:
                self.module = nn.Sequential(
                    _maybe_wrap(checkpoint_wrapper(nn.Linear(8, 4))),
                    _maybe_wrap(checkpoint_wrapper(nn.Linear(4, 16))),
                    _maybe_wrap(checkpoint_wrapper(nn.Linear(16, 4))),
                    _maybe_wrap(checkpoint_wrapper(nn.Linear(4, 8))),
                )
            else:
                self.module = nn.Sequential(
                    _maybe_wrap(nn.Linear(8, 4)),
                    _maybe_wrap(nn.Linear(4, 16)),
                    _maybe_wrap(nn.Linear(16, 4)),
                    _maybe_wrap(nn.Linear(4, 8)),
                )

    def get_input(self, device):
        torch.manual_seed(1 + self.rank)  # keep everything deterministic
        return (torch.rand(4, 8, device=device),)

    def forward(self, x):
        return self.module(x)

    def get_loss(self, input, output):
        loss = output.sum()
        return loss

    def run_backward(self, loss):
        loss.backward()


def spawn_and_init(fn, args=None, **spawn_kwargs):
    if args is None:
        args = ()

    run_fn = functools.partial(init_and_run, fn, args)

    # Below 3 lines are to easily enable single-process debugging
    # _, filename = tempfile.mkstemp()
    # _, filename_rpc = tempfile.mkstemp()
    # run_fn(0, 1, filename, filename_rpc)

    spawn_for_all_world_sizes(run_fn, **spawn_kwargs)


def init_and_run(fn, args, rank, world_size, filename, filename_rpc):
    dist_init(rank, world_size, filename, filename_rpc)
    group = torch.distributed.new_group()
    fn(rank, group, *args)


if __name__ == "__main__":
    unittest.main()
