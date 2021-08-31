# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
import glob
import os
import sys
import time
import unittest

import pytest
import torch
from torch import nn
import torch.distributed

from fairscale.nn.data_parallel import FullyShardedDataParallel, TrainingState
from fairscale.utils import torch_version
from fairscale.utils.testing import dist_init, rmf, spawn_for_all_world_sizes

# How to use remote-pdb: https://gist.github.com/sshleifer/9d43351957179c13606e015b072927d4
# All helper functions called by spawn must be either @classmethod, @staticmethod


class TimeKeeper:
    def __init__(self):
        self.start_time = time.time()

    def print_time(self, s: str, wait_time: float = 5.0):
        cur_time = time.time()
        print(f"@time: {cur_time - self.start_time:0.2f} {s}")
        time.sleep(wait_time)


tk = TimeKeeper()


class DistributedTest(unittest.TestCase):
    def setUp(self):
        if torch_version() < (1, 6, 0):
            raise unittest.SkipTest("Need pytorch version >= 1.6 due to lack of reduce_scatter")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available, skipping test")
        if sys.platform == "win32":
            raise unittest.SkipTest("NCCL doesn't support Windows, skipping test")
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("distributed tests require 2+ GPUs, skipping")

    @staticmethod
    def _train_for_several_steps(model, num_steps, autocast, lr=0.01, norm_type=None):
        model_device = next(model.parameters()).device
        # use SGD with momentum instead of Adam, since Adam is scale invariant
        # and this makes it bad for tests
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        for _ in range(num_steps):
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=autocast):
                # Inputs always cuda regardless of move_grads_cpu, or model.device
                input = model.module.get_input(torch.device("cuda"))
                output = model(*input)
                loss = model.module.get_loss(input, output).to(model_device)
            assert loss.dtype == torch.float32
            model.module.run_backward(loss)
            if norm_type is not None:
                clip_norm = 0.3
                if isinstance(model, FullyShardedDataParallel):
                    model.clip_grad_norm_(clip_norm, norm_type)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, norm_type)
            # for name, param in model.named_parameters():
            # print(f"name {name} param {param.device} {param.storage().size()}")
            # optim.step()
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

            tk.print_time(f"eval step: {_}", 1.0)

    @staticmethod
    def get_wrapped_model(group, cuda_first=False, config={}, **model_kwargs) -> FullyShardedDataParallel:
        if cuda_first:
            model = FullyShardedDataParallel(TransformerWithSharedParams(group, **model_kwargs).cuda(), group, **config)
        else:
            model = FullyShardedDataParallel(TransformerWithSharedParams(group, **model_kwargs), group, **config).cuda()
        return model


class TestSsdLoading(DistributedTest):
    def test_memory_benchmark(self):
        test_fn = functools.partial(self._test_memory_benchmark, config={})
        spawn_and_init(test_fn)

    def test_named_parameter(self):
        test_fn = functools.partial(self._test_named_parameter, config={})
        spawn_and_init(test_fn)

    def test_parameter(self):
        test_fn = functools.partial(self._test_parameter, config={})
        spawn_and_init(test_fn)

    @classmethod
    def _test_named_parameter(self, rank, group, config):
        SIZE = 128
        model_orig = SimpleLinear(group, input_size=SIZE, output_size=SIZE, layers=1)

        config["ssd_offload"] = True
        model = FullyShardedDataParallel(model_orig, **config)

        with pytest.raises(RuntimeError):
            for n, p in model.named_parameters():
                pass

        config["ssd_offload"] = False
        model = FullyShardedDataParallel(model_orig, **config)
        for n, p in model.named_parameters():
            pass

        fileList = glob.glob(os.getcwd() + "/*_rank*")
        for file in fileList:
            rmf(file)

    @classmethod
    def _test_parameter(self, rank, group, config):
        SIZE = 128
        model_orig = SimpleLinear(group, input_size=SIZE, output_size=SIZE, layers=1)

        config["ssd_offload"] = True
        model = FullyShardedDataParallel(model_orig, **config)

        with pytest.raises(RuntimeError):
            for p in model.parameters():
                pass

        config["ssd_offload"] = False
        model = FullyShardedDataParallel(model_orig, **config)
        for p in model.parameters():
            pass

        fileList = glob.glob(os.getcwd() + "/*_rank*")
        for file in fileList:
            rmf(file)

    @classmethod
    def _test_memory_benchmark(self, rank, group, config):

        SIZE = 1024 * 16
        tk.print_time("START", 1.0)
        a = torch.empty(1)
        b = a.cuda()
        # wait for cuda to fully load
        time.sleep(5)
        tk.print_time("INIT_CUDA", 1.0)
        model = SimpleLinear(group, input_size=SIZE, output_size=SIZE, layers=4)
        tk.print_time("CPU_MODEL", 1.0)

        # Train the model for 1 step.

        config["ssd_offload"] = True
        model = FullyShardedDataParallel(model, **config)
        tk.print_time("FSDP_MODEL", 1.0)

        self._eval_for_several_steps(model, 4, autocast=False)
        tk.print_time("TRAIN_1")

        fileList = glob.glob(os.getcwd() + "/*_rank*")
        for file in fileList:
            rmf(file)


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
        return self.module(src_ids)

    def get_loss(self, input, output):
        _, tgt = input

        return nn.functional.binary_cross_entropy_with_logits(output, tgt)

    def run_backward(self, loss):
        loss.backward()


def spawn_and_init(fn, args=None, **spawn_kwargs):
    if args is None:
        args = ()

    run_fn = functools.partial(init_and_run, fn, args)
    spawn_for_all_world_sizes(run_fn, **spawn_kwargs, world_sizes=[1])


def init_and_run(fn, args, rank, world_size, filename, filename_rpc):
    dist_init(rank, world_size, filename, filename_rpc)
    group = torch.distributed.new_group()
    fn(rank, group, *args)


if __name__ == "__main__":
    unittest.main()
