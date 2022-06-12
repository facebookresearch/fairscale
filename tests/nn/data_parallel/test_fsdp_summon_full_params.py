# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
import gc
import unittest

from parameterized import parameterized
import pytest
import torch

from fairscale.internal.version import torch_version

from .test_fsdp import CONFIG_OPTIONS, DistributedTest, rename_test, spawn_and_init


def get_cuda_mem():
    torch.cuda.synchronize()
    gc.collect()
    return torch.cuda.memory_allocated()


@pytest.mark.skipif(torch_version() < (1, 8, 0), reason="pytorch version >= 1.8.0 required")
class TestMemory(DistributedTest):
    @parameterized.expand(CONFIG_OPTIONS, name_func=rename_test)
    def test_memory(self, config):
        spawn_and_init(functools.partial(self._test_memory, config))

    @parameterized.expand(CONFIG_OPTIONS, name_func=rename_test)
    def test_memory_volatile(self, config):
        spawn_and_init(functools.partial(self._test_memory, config, volatile=True))

    @classmethod
    def _test_memory(self, config, rank, group, volatile=False):
        model = self.get_wrapped_model(group, cuda_first=False, config=config)
        self._train_for_several_steps(model, 1, autocast=model.mixed_precision)

        mems = [get_cuda_mem()]

        with model.summon_full_params(volatile=volatile):
            mems.append(get_cuda_mem())
            assert mems[1] >= mems[0]

            state_dict = model.state_dict()
            mems.append(get_cuda_mem())
            assert mems[2] >= mems[1]

        mems.append(get_cuda_mem())
        assert mems[3] <= mems[2]

        del state_dict
        mems.append(get_cuda_mem())
        # Any value other than `==` indicates a memory leak. If mems[4] >
        # mems[0], that indicates we're not cleaning up params properly in
        # summon_full_params. If mems[4] < mems[0], that indicates there's a
        # memory leak in _train_for_several_steps.
        assert mems[4] == mems[0], f"memory leak detected, {mems[4]} != {mems[0]}"


class TestPersistence(DistributedTest):
    @parameterized.expand(CONFIG_OPTIONS, name_func=rename_test)
    def test_non_volatile(self, config):
        spawn_and_init(functools.partial(self._test_persistence, config))

    @classmethod
    def _test_persistence(self, config, rank, group, volatile=False):
        model = self.get_wrapped_model(group, cuda_first=False, config=config)

        with model.summon_full_params(volatile=False):
            model.module.embed_tokens.weight.data.fill_(42)
        with model.summon_full_params():
            # non-volatile changes are persisted
            assert torch.all(model.module.embed_tokens.weight.data == 42.0)


if __name__ == "__main__":
    unittest.main()
