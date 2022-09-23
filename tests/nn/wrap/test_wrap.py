# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import os
import random
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairscale.fair_dev.testing.testing import DummyProcessGroup
from fairscale.nn import FullyShardedDataParallel as FSDP
from fairscale.nn import auto_wrap, default_auto_wrap_policy, enable_wrap, wrap

try:
    from torch.cuda.amp import autocast
except ImportError:
    autocast = None  # type: ignore


class TestAutoWrap(unittest.TestCase):
    def setUp(self) -> None:
        # For all the tests here, we use a fake group and flatten being False since those should
        # not affect how wrapping work.
        self.process_group = DummyProcessGroup(rank=0, size=1)

    def test_wrap(self):
        with enable_wrap(wrapper_cls=FSDP, flatten_parameters=False, process_group=self.process_group):
            layer = wrap(nn.Linear(5, 5))
        assert isinstance(layer, FSDP)
        assert layer.flatten_parameters is False

    def test_wrap_disabled_outside_context(self):
        layer = wrap(nn.Linear(5, 5))
        assert isinstance(layer, nn.Linear)

    def test_wrap_override_defaults(self):
        with enable_wrap(wrapper_cls=FSDP, flatten_parameters=False, process_group=self.process_group):
            layer = wrap(nn.Linear(5, 5), flatten_parameters=True)
        assert isinstance(layer, FSDP)
        assert layer.flatten_parameters

    def test_auto_wrap(self):
        """
        Test to ensure with auto wrap, we wrap child modules correctly based on the min_num_params.
        ``nn.Linear(5, 5)`` does not exceed the bucket size, but combined they do.
        Root is not wrapped given there are not enough unwrapped params left and skip_params_check_for_root
        is not set.
        """
        with enable_wrap(wrapper_cls=FSDP, process_group=self.process_group, flatten_parameters=False):
            sequential = nn.Sequential(nn.Linear(5, 5), nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5)))
            my_auto_wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=60)
            model = auto_wrap(sequential, auto_wrap_policy=my_auto_wrap_policy)
        assert isinstance(model, nn.Sequential)
        assert isinstance(model[0], nn.Linear)
        assert isinstance(model[1], FSDP)
        assert isinstance(model[1].module[0], nn.Linear)
        assert isinstance(model[1].module[1], nn.Linear)

    def test_auto_wrap_skip_root_checks(self):
        """
        Similar test as before but this time we set skip_params_check_for_root=True in the wrap policy.
        So in this case the root is wrapped even without enough remaining unwrapped params.
        """
        with enable_wrap(wrapper_cls=FSDP, process_group=self.process_group, flatten_parameters=False):
            sequential = nn.Sequential(nn.Linear(5, 5), nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5)))
            my_auto_wrap_policy = functools.partial(
                default_auto_wrap_policy, min_num_params=60, skip_params_check_for_root=True
            )
            model = auto_wrap(sequential, auto_wrap_policy=my_auto_wrap_policy)
        assert isinstance(model, FSDP)
        assert isinstance(model.module[0], nn.Linear)
        assert isinstance(model.module[1], FSDP)
        assert isinstance(model.module[1].module[0], nn.Linear)
        assert isinstance(model.module[1].module[1], nn.Linear)

    def test_auto_wrap_preset_exclude_wrap(self):
        """
        Test to ensure excluded modules are not wrapped, regardless if the total param size is greater than the
        min_num_params.
        """
        with enable_wrap(wrapper_cls=FSDP, process_group=self.process_group, flatten_parameters=False):
            sequential = nn.ModuleList([nn.Linear(5, 5), nn.Linear(5, 5)])
            my_auto_wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=40)
            model = auto_wrap(sequential, auto_wrap_policy=my_auto_wrap_policy)
        assert isinstance(model, nn.ModuleList)
        assert isinstance(model[0], nn.Linear)
        assert isinstance(model[1], nn.Linear)

    def test_auto_wrap_preset_exclude_wrap_include_children(self):
        """
        Test to ensure excluded modules are not wrapped, but children are if param size is greater than
        min_num_params
        """
        with enable_wrap(wrapper_cls=FSDP, process_group=self.process_group, flatten_parameters=False):
            sequential = nn.ModuleList([nn.Linear(10, 10)])
            my_auto_wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=40)
            model = auto_wrap(sequential, auto_wrap_policy=my_auto_wrap_policy)
        assert isinstance(model, nn.ModuleList)
        assert isinstance(model[0], FSDP)

    def test_auto_wrap_preset_force_leaf(self):
        """
        Test to ensure force-leaf modules are not wrapped, and children are not wrapped.
        """
        with enable_wrap(wrapper_cls=FSDP, process_group=self.process_group, flatten_parameters=False):
            sequential = nn.Sequential(nn.Linear(10, 10), nn.MultiheadAttention(100, 1))
            my_auto_wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=40)
            model = auto_wrap(sequential, auto_wrap_policy=my_auto_wrap_policy)
        assert isinstance(model.module[0], FSDP)
        # Assert children of multihead attention are not wrapped
        assert isinstance(model.module[1], nn.MultiheadAttention)
        assert isinstance(model.module[1].out_proj, nn.Linear)

    def test_auto_wrap_preset_force_leaf_custom(self):
        """
        Test to ensure force-leaf modules are not wrapped.
        """
        my_auto_wrap_policy = functools.partial(
            default_auto_wrap_policy,
            min_num_params=40,
            force_leaf_modules=default_auto_wrap_policy.FORCE_LEAF_MODULES.union({nn.Linear}),
        )
        with enable_wrap(
            auto_wrap_policy=my_auto_wrap_policy,
            wrapper_cls=FSDP,
            process_group=self.process_group,
            flatten_parameters=False,
        ):
            sequential = nn.Sequential(nn.Linear(10, 10), nn.ModuleList([nn.Linear(10, 10)]))
            model = auto_wrap(sequential)
        # Model was wrapped in FSDP as no inner modules were wrapped.
        assert isinstance(model, FSDP)
        assert isinstance(model.module[0], nn.Linear)
        assert isinstance(model.module[1], nn.ModuleList)

    # todo: currently complains that address is in use, not sure why since I clear the proc group.
    # def test_auto_wrap_smoke(self):
    #     self._auto_wrap_smoke_test(enable_mixed_precision=False)

    def test_auto_wrap_smoke_autocast(self):
        """
        Ensure we can do a forward/backward through an auto-wrapped model.
        """
        self._auto_wrap_smoke_test(enable_mixed_precision=True)

    @unittest.skipIf(not torch.cuda.is_available(), "Test Requires CUDA")
    @unittest.skipIf(autocast is None, "Test Requires autocast")
    def _auto_wrap_smoke_test(self, enable_mixed_precision):
        device = torch.device("cuda")
        torch.cuda.set_device(0)

        # Random port in case the next test run quickly, same port would cause conflict.
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(random.randint(2000, 3000))
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

        try:
            with enable_wrap(wrapper_cls=FSDP, mixed_precision=enable_mixed_precision):
                sequential = nn.Sequential(
                    nn.Linear(5, 5), nn.Linear(5, 5), nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5))
                )
                my_auto_wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=40)
                model = auto_wrap(sequential, auto_wrap_policy=my_auto_wrap_policy)
            model.to(device)
            input = torch.rand((1, 5), dtype=torch.float).to(device)

            with autocast(enabled=enable_mixed_precision):
                output = model(input)
                loss = F.mse_loss(input, output)
            loss.backward()
        finally:
            torch.distributed.destroy_process_group()
            del os.environ["MASTER_ADDR"]
            del os.environ["MASTER_PORT"]
