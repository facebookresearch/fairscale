# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing OssDdp class.
"""

import pytest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fairscale.nn.data_parallel import OssDdp

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
skip_if_single_gpu = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multiple GPUs required")

def test_on_cpu():
    run_test(backend=dist.Backend.GLOO, device=torch.device("cpu"))

@skip_if_no_cuda
@skip_if_single_gpu
def test_on_gpu():
    run_test(backend=dist.Backend.NCCL, device=torch.device("cuda"))

def run_test(backend, device):
    pass
