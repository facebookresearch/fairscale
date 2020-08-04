# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing OssDdp class.
"""

import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Linear, Sequential

from fairscale.nn.data_parallel import OssDdp
from fairscale.optim import OSS

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
skip_if_single_gpu = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multiple GPUs required")


def test_on_cpu():
    run_test(backend=dist.Backend.GLOO, device=torch.device("cpu"))


@skip_if_no_cuda
@skip_if_single_gpu
def test_on_gpu():
    run_test(backend=dist.Backend.NCCL, device=torch.device("cuda"))


def run_one_step(rank, world_size, backend, device, temp_file_name):
    url = "file://" + temp_file_name
    dist.init_process_group(init_method=url, backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = Sequential(Linear(2, 3), Linear(3, 4)).to(device)
    optimizer = OSS(model.parameters(), lr=0.1, momentum=0.99)
    ddp = OssDdp(model, optimizer, world_size)
    input_tensor = torch.rand((64, 2)).to(device)
    output = ddp(input_tensor).sum()
    output.backward()
    ddp.reduce()
    optimizer.step()
    # TODO: I need to figure out a way to verify the grads are reduced correctly
    #       between the ranks. I haven't found the best way yet. Will need to come
    #       back here before this is used in real training.


def run_test(backend, device, world_size=2):
    temp_file_name = tempfile.mkstemp()[1]
    mp.spawn(run_one_step, args=(world_size, backend, device, temp_file_name), nprocs=world_size, join=True)
