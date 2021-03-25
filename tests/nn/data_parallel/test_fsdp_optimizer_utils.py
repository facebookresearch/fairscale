# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import functools
from time import time

from parameterized import parameterized
import torch
from torch.optim import SGD, Adadelta, Adam  # type: ignore

from fairscale.nn import FullyShardedDataParallel
from fairscale.optim.utils import recursive_copy_to_device
from fairscale.utils.testing import objects_are_equal

from .test_fsdp import (
    DistributedTest,
    DummyProcessGroup,
    NestedWrappedModule,
    TransformerWithSharedParams,
    rename_test,
    spawn_and_init,
)


def first_tensor_numel(dct):
    for k, v in dct.items():
        if torch.is_tensor(v):
            return v.numel()
    return 0


def assert_equal(a, b):
    assert a == b, f"{a} != {b}"


class TestOptimizerUtils(DistributedTest):
    @parameterized.expand(
        [[functools.partial(SGD, momentum=0.9), True], [SGD, False], [Adam, False], [Adadelta, True]],
        name_func=rename_test,
    )
    def test_consolidate_optimizer(self, optim_fn, transformer):
        config = {"mixed_precision": True, "flatten_parameters": True}
        test_fn = functools.partial(
            self._test_consolidated_optimizer, config, optim_fn=optim_fn, transformer=transformer
        )

        spawn_and_init(test_fn, world_sizes=[min(torch.cuda.device_count(), 4)])

    @classmethod
    def _test_consolidated_optimizer(self, config, rank, group, optim_fn=torch.optim.SGD, transformer=False):
        """FSDP.gather_full_optim_state_dict() should return something very similar to optimizer.state_dict()"""
        # Establish reference behavior.

        if transformer:
            fsdp = self.get_wrapped_model(group, config=config).cuda()
            unwrapped_model = TransformerWithSharedParams(group).cuda()
        else:
            fsdp = FullyShardedDataParallel(NestedWrappedModule(group, wrapper_config=config), group, **config).cuda()
            unwrapped_model = NestedWrappedModule(group, wrapper_config=None).cuda()

        try:
            fsdp_optim = optim_fn(fsdp.parameters(), lr=0.01,)
            optim_unwrapped = optim_fn(unwrapped_model.parameters(), lr=0.01)
        except TypeError:  # Adadelta
            fsdp_optim = optim_fn(fsdp.parameters())
            optim_unwrapped = optim_fn(unwrapped_model.parameters())

        fsdp_optim.zero_grad()
        optim_unwrapped.zero_grad()

        x = fsdp.module.get_input(torch.device("cuda"))
        output = fsdp(*x)
        loss = fsdp.module.get_loss(x, output).to("cuda")
        fsdp.module.run_backward(loss)
        fsdp_optim.step()

        output = unwrapped_model(*x)
        loss = unwrapped_model.get_loss(x, output)
        unwrapped_model.run_backward(loss)
        optim_unwrapped.step()
        unwrapped_sd = optim_unwrapped.state_dict()

        tstart = time()
        sd = fsdp.gather_full_optim_state_dict(fsdp_optim, recipient_rank=0)
        duration = time() - tstart
        # Switching from fairscale.optim.utils.broadcast_object to torch.broadcast_object_list will cause this to raise
        assert duration < fsdp.world_size, f"gather optim state took {duration} seconds, suspect change in _consolidate"

        if fsdp.rank > 0:
            return

        assert_equal(len(sd["state"]), len(unwrapped_sd["state"]))
        assert_equal(len(sd["param_groups"][0]["params"]), len(unwrapped_sd["param_groups"][0]["params"]))
        assert_equal(
            sum([first_tensor_numel(v) for k, v in sd["state"].items()]),
            sum([first_tensor_numel(v) for k, v in unwrapped_sd["state"].items()]),
        )

        shard_sd = fsdp.get_shard_from_optim_state_dict(sd)

        original_shard_sd = fsdp_optim.state_dict()
        assert_equal(len(shard_sd["state"]), len(original_shard_sd["state"]))
        assert_equal(shard_sd.keys(), original_shard_sd.keys())
        original_shard_sd = recursive_copy_to_device(original_shard_sd, non_blocking=False, device="cpu")

        assert_equal(
            sum([first_tensor_numel(v) for k, v in shard_sd["state"].items()]),
            sum([first_tensor_numel(v) for k, v in original_shard_sd["state"].items()]),
        )
        assert objects_are_equal(shard_sd, original_shard_sd)

    def test_named_params_ordering(self):
        """Test assumption of consolidate_optimizer_state_dict"""
        group = DummyProcessGroup(0, 1)
        model = TransformerWithSharedParams(group)
        named_pars = [p for n, p in model.named_parameters()]
        for i, p in enumerate(model.parameters()):
            assert objects_are_equal(p, named_pars[i])
