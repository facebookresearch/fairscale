# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
import itertools
import unittest
from unittest.mock import patch

from parameterized import parameterized
import torch

from fairscale.nn.data_parallel import FullyShardedDataParallel
from fairscale.utils.testing import DummyProcessGroup, objects_are_equal

from .test_fsdp import DistributedTest, NestedWrappedModule, rename_test, spawn_and_init


class TestNoSync(DistributedTest):
    def test_transformer(self):
        fn = functools.partial(self._test_transformer, config={})
        spawn_and_init(fn)

    def test_transformer_no_flat_params(self):
        config = {"flatten_parameters": False}
        fn = functools.partial(self._test_transformer, config=config)
        spawn_and_init(fn)

    def test_nested_wrapper(self):
        fn = functools.partial(self._test_nested_wrapper, config={})
        spawn_and_init(fn)

    def test_no_sync_before_first_forward(self):
        group = DummyProcessGroup(rank=0, size=1)
        model = self.get_wrapped_model(group, config={}, add_bn=False)
        batch = model.module.get_input(torch.device("cuda"))
        with model.no_sync():
            output = model(*batch)
            loss = model.module.get_loss(batch, output)
            loss.backward()
        output = model(*batch)
        loss = model.module.get_loss(batch, output)
        loss.backward()

    @classmethod
    def _test_transformer(self, rank, group, config):
        model = self.get_wrapped_model(group, config=config, add_bn=False)
        model.eval()  # turn off dropout for the test
        self._test_no_sync(model, batch_dim=1)

    @classmethod
    def _test_nested_wrapper(self, rank, group, config):
        model = NestedWrappedModule(group, config)
        model = FullyShardedDataParallel(model, group, **config).cuda()
        self._test_no_sync(model, batch_dim=0)

    @classmethod
    def _test_no_sync(self, model, batch_dim):
        # Generate two input batches. We'll test that we get the same grads if
        # we train on them sequentially while accumulating grads (with no_sync)
        # vs. concatenating the batches and training in one go.
        batch1 = model.module.get_input(torch.device("cuda"))
        assert isinstance(batch1, tuple)
        batch2 = tuple(
            # This randomly permutes the values in a multi-dim tensor.
            x.view(-1)[torch.randperm(x.numel())].view_as(x)
            for x in batch1
        )
        for x, y in zip(batch1, batch2):
            assert not torch.all(x == y)

        # Concat the batches along batch dimension.
        concat_batch = tuple(torch.cat((x, y), dim=batch_dim) for (x, y) in zip(batch1, batch2))

        # Establish reference behavior on the concat batch.
        model.zero_grad()
        output = model(*concat_batch)
        ref_loss = model.module.get_loss(concat_batch, output)
        ref_loss.backward()
        ref_grads = [p.grad.detach().clone() for p in model.parameters()]

        # Test that we get the same results by accumulating grads.
        model.zero_grad()
        with model.no_sync():  # accumulate gradients from the first batch
            output = model(*batch1)
            loss1 = model.module.get_loss(batch1, output)
            loss1.backward()
        output = model(*batch2)
        loss2 = model.module.get_loss(batch2, output)
        loss2.backward()
        accumulated_loss = loss1 + loss2
        accumulated_grads = [p.grad.detach().clone() for p in model.parameters()]

        torch.testing.assert_allclose(ref_loss, accumulated_loss)
        assert objects_are_equal(ref_grads, accumulated_grads, raise_exception=True)


keys = ["reshard_after_forward", "mixed_precision"]
COMM_CONFIG_OPTIONS = [[dict(zip(keys, config))] for config in itertools.product([True, False], repeat=len(keys))]


class TestNoSyncCommunication(DistributedTest):
    @parameterized.expand(COMM_CONFIG_OPTIONS, name_func=rename_test)
    def test_communication(self, config):
        fn = functools.partial(self._test_communication, config=config)
        spawn_and_init(fn)

    @parameterized.expand(COMM_CONFIG_OPTIONS, name_func=rename_test)
    def test_communication_nested(self, config):
        fn = functools.partial(self._test_communication, config=config, nested_model=True)
        spawn_and_init(fn)

    @classmethod
    def _test_communication(self, rank, group, config, nested_model=False):
        if group.size() == 1:
            return

        # Turn off bucketing to accurately count number of reduce_scatters.
        config["bucket_cap_mb"] = 0

        if nested_model:
            model = NestedWrappedModule(group, config)
            model = FullyShardedDataParallel(model, group, **config).cuda()
        else:
            model = self.get_wrapped_model(group, config=config)

        num_fsdp = 0
        for child in model.modules():  # includes self
            if isinstance(child, FullyShardedDataParallel) and len(child.params) > 0:
                num_fsdp += 1

        if config.get("reshard_after_forward", True):
            # inside no_sync:
            #   num_fsdp all-gathers in the forward
            #   num_fsdp-1 all-gathers in the backward (except root)
            # outside no_sync:
            #   num_fsdp-1 all-gathers in the forward (except root)
            #   num_fsdp-1 all-gathers in the backward (except root)
            expected_all_gather1 = 2 * num_fsdp - 1
            expected_all_gather2 = expected_all_gather1 + (2 * num_fsdp - 2)
        else:
            # inside no_sync:
            #   num_fsdp all-gathers in the forward
            # outside no_sync:
            #   none
            expected_all_gather1 = num_fsdp
            expected_all_gather2 = num_fsdp

        expected_reduce_scatter = num_fsdp

        batch = model.module.get_input(torch.device("cuda"))
        with patch("torch.distributed.all_gather") as mock_all_gather:
            with patch("torch.distributed.reduce_scatter") as mock_reduce_scatter:
                with model.no_sync():
                    output = model(*batch)
                    loss = model.module.get_loss(batch, output)
                    loss.backward()

                assert (
                    mock_all_gather.call_count == expected_all_gather1
                ), f"{mock_all_gather.call_count} != {expected_all_gather1}"
                assert mock_reduce_scatter.call_count == 0, f"{mock_reduce_scatter.call_count} != 0"

                output = model(*batch)
                loss = model.module.get_loss(batch, output)
                loss.backward()

                assert (
                    mock_all_gather.call_count == expected_all_gather2
                ), f"{mock_all_gather.call_count} != {expected_all_gather2}"
                assert (
                    mock_reduce_scatter.call_count == expected_reduce_scatter
                ), f"{mock_reduce_scatter.call_count} != {expected_reduce_scatter}"


if __name__ == "__main__":
    unittest.main()
