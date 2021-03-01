# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import unittest
from unittest import mock

import torch
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F

from fairscale.nn import FullyShardedDataParallel as FSDP
from fairscale.nn import auto_wrap, enable_wrap, wrap


class TestAutoWrap(unittest.TestCase):
    def setUp(self) -> None:
        class TestDistributedGroup:
            def size(self):
                return 1

            @property
            def world_size(self):
                return 1

            def rank(self):
                return 0

        self.process_group = TestDistributedGroup()

    def test_wrap(self):
        with enable_wrap(flatten_parameters=False, process_group=self.process_group):
            layer = wrap(nn.Linear(5, 5))
        assert isinstance(layer, FSDP)
        assert layer.flatten_parameters is False

    def test_wrap_disabled_outside_context(self):
        layer = wrap(nn.Linear(5, 5))
        assert isinstance(layer, nn.Linear)

    def test_wrap_override_defaults(self):
        with enable_wrap(flatten_parameters=False, process_group=self.process_group):
            layer = wrap(nn.Linear(5, 5), flatten_parameters=True)
        assert isinstance(layer, FSDP)
        assert layer.flatten_parameters

    def test_auto_wrap(self):
        """
        Test to ensure with auto wrap, we wrap child modules correctly based on the min_num_params.
        ``nn.Linear(5, 5)`` does not exceed the bucket size, but combined they do.
        """
        with enable_wrap(process_group=self.process_group, flatten_parameters=False):
            sequential = nn.Sequential(
                nn.Linear(5, 5), nn.Linear(5, 5), nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5))
            )
            model = auto_wrap(sequential, min_num_params=40)
        assert isinstance(model, FSDP)
        assert isinstance(model.module[0], nn.Linear)
        assert isinstance(model.module[1], nn.Linear)
        assert isinstance(model.module[2], FSDP)
        assert isinstance(model.module[2].module[0], nn.Linear)
        assert isinstance(model.module[2].module[1], nn.Linear)

    # todo: currently complains that address is in use, not sure why since I clear the proc group.
    # def test_auto_wrap_smoke(self):
    #     self._auto_wrap_smoke_test(enable_mixed_precision=False)

    def test_auto_wrap_smoke_autocast(self):
        """
        Ensure we can do a forward/backward through an auto-wrapped model.
        """
        self._auto_wrap_smoke_test(enable_mixed_precision=True)

    @mock.patch.dict(os.environ, {"MASTER_ADDR": "localhost", "MASTER_PORT": "1337"}, clear=True)
    @unittest.skipIf(not torch.cuda.is_available(), "Test Requires CUDA")
    def _auto_wrap_smoke_test(self, enable_mixed_precision):
        device = torch.device("cuda")
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

        with enable_wrap(mixed_precision=enable_mixed_precision):
            sequential = nn.Sequential(
                nn.Linear(5, 5), nn.Linear(5, 5), nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5))
            )
            model = auto_wrap(sequential, min_num_params=40)
        model.to(device)
        input = torch.rand((1, 5), dtype=torch.float).to(device)

        with autocast(enabled=enable_mixed_precision):
            output = model(input)
            loss = F.mse_loss(input, output)
        loss.backward()
        torch.distributed.destroy_process_group()
