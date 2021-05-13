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
from fairscale.nn.data_parallel.fsdp_optim_utils import is_singleton_tensor
from fairscale.utils.params import recursive_copy_to_device
from fairscale.utils.testing import objects_are_equal

from .test_fsdp import (
    DistributedTest,
    DummyProcessGroup,
    MixtureOfExperts,
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
        [[functools.partial(SGD, momentum=0.9), True], [SGD, False], [Adam, False], [Adadelta, True], [Adam, True]],
        name_func=rename_test,
    )
    def test_consolidate_optimizer(self, optim_fn, transformer):
        config = {"mixed_precision": True, "flatten_parameters": True}
        config["fp32_compute_dtype"] = True
        test_fn = functools.partial(
            self._test_consolidated_optimizer, config, optim_fn=optim_fn, transformer=transformer
        )

        spawn_and_init(test_fn, world_sizes=[min(torch.cuda.device_count(), 4)])

    @classmethod
    def _test_consolidated_optimizer(self, config, rank, group, optim_fn=torch.optim.SGD, transformer=False):
        """FSDP.gather_full_optim_state_dict() should return something very similar to optimizer.state_dict()"""
        # Establish reference behavior.

        if transformer:
            unwrapped_model = TransformerWithSharedParams(group, wrapper_config=config).cuda()
            fsdp = self.get_wrapped_model(group, config=config).cuda()
        else:
            unwrapped_model = MixtureOfExperts(group, wrapper_config=None).cuda()
            fsdp = FullyShardedDataParallel(MixtureOfExperts(group, wrapper_config=config)).cuda()

        try:
            fsdp_optim = optim_fn(fsdp.parameters(), lr=0.01,)
            optim_unwrapped = optim_fn(unwrapped_model.parameters(), lr=0.01)
        except TypeError:  # Adadelta
            fsdp_optim = optim_fn(fsdp.parameters())
            optim_unwrapped = optim_fn(unwrapped_model.parameters())

        fsdp_optim.zero_grad()
        optim_unwrapped.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
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

        if not transformer:
            no_broadcast_children = [x for x in fsdp._fsdp_instances if x.no_broadcast_optim_state]
            assert len(no_broadcast_children) == 1
            assert fsdp._fsdp_instances[-1].no_broadcast_optim_state
        torch.cuda.empty_cache()
        cuda_gb_before = torch.cuda.memory_stats(fsdp.rank)["allocated_bytes.all.current"] / 1024 ** 3
        tstart = time()
        sd = fsdp.gather_full_optim_state_dict(fsdp_optim, recipient_rank=0)
        duration = time() - tstart
        # Switching from fairscale.utils.params.broadcast_object to torch.broadcast_object_list will cause this to raise
        assert duration < fsdp.world_size, f"gather optim state took {duration} seconds, suspect change in _consolidate"

        cuda_gb_after = torch.cuda.memory_stats(fsdp.rank)["allocated_bytes.all.current"] / 1024 ** 3
        mem_usg_gb = cuda_gb_after - cuda_gb_before
        assert mem_usg_gb == 0, f"gather_full_optim_state_dict used {mem_usg_gb:.2f} CUDA GB, max allowed is 0"
        assert cuda_gb_after > 0, "got 0 memory usage, logging is broken"

        if fsdp.rank > 0:
            assert sd is None
            return

        # assert whole state dict on CPU
        for k, v in sd["state"].items():
            for buffer_name, t in v.items():
                if torch.is_tensor(t):
                    msg = f"got device {t.device} for {k}: {buffer_name}. expected CPU"
                    assert t.device == torch.device("cpu"), msg

        unflat_state = sd["state"]
        assert "uncollected_local_ids" in sd
        shard_sd = fsdp.get_shard_from_optim_state_dict(sd)
        shard_sd = recursive_copy_to_device(shard_sd, non_blocking=False, device="cpu")
        state_after_get_shard = sd["state"]
        assert objects_are_equal(unflat_state, state_after_get_shard)  # no side effects.

        assert_equal(len(sd["state"]), len(unwrapped_sd["state"]))
        assert_equal(len(sd["param_groups"][0]["params"]), len(unwrapped_sd["param_groups"][0]["params"]))
        assert_equal(
            sum([first_tensor_numel(v) for k, v in sd["state"].items()]),
            sum([first_tensor_numel(v) for k, v in unwrapped_sd["state"].items()]),
        )

        original_shard_sd = fsdp_optim.state_dict()
        assert_equal(len(shard_sd["state"]), len(original_shard_sd["state"]))
        assert_equal(shard_sd.keys(), original_shard_sd.keys())
        original_shard_sd = recursive_copy_to_device(original_shard_sd, non_blocking=False, device="cpu")
        # Before asserting that the dicts are equal, we check keys individually to allow nice tracebacks.
        assert_equal(
            [first_tensor_numel(v) for k, v in shard_sd["state"].items()],
            [first_tensor_numel(v) for k, v in original_shard_sd["state"].items()],
        )
        assert_equal(
            [v for k, v in shard_sd["param_groups"][0].items()],
            [v for k, v in original_shard_sd["param_groups"][0].items()],
        )
        assert objects_are_equal(shard_sd["state"], original_shard_sd["state"])
        assert objects_are_equal({k: shard_sd[k] for k in original_shard_sd}, original_shard_sd)

    def test_named_params_ordering(self):
        """Test assumption of consolidate_optimizer_state_dict"""
        group = DummyProcessGroup(0, 1)
        model = TransformerWithSharedParams(group)
        named_pars = [p for n, p in model.named_parameters()]
        for i, p in enumerate(model.parameters()):
            assert objects_are_equal(p, named_pars[i])

    def test_is_singleton_tensor(self):
        assert is_singleton_tensor(torch.tensor(4.0))
        assert not is_singleton_tensor(torch.tensor([4.0]))
        assert not is_singleton_tensor(torch.tensor([4.0, 5.0]))
        assert not is_singleton_tensor([4.0])
        assert not is_singleton_tensor(4.0)
