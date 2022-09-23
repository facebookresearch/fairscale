# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import copy
import functools
from time import time
import unittest

from parameterized import parameterized
import torch
from torch import nn
from torch.optim import SGD, Adadelta, Adam  # type: ignore

from fairscale.fair_dev.testing.testing import dist_init, objects_are_equal, spawn_for_all_world_sizes
from fairscale.internal.params import recursive_copy_to_device
from fairscale.nn.data_parallel import FullyShardedDataParallel, get_fsdp_instances
from fairscale.nn.data_parallel.fsdp_optim_utils import is_singleton_tensor

from .test_fsdp import (
    DistributedTest,
    DummyProcessGroup,
    MixtureOfExperts,
    TransformerWithSharedParams,
    rename_test,
    spawn_and_init,
)


def all_tensors_numel_except_for_step(dct):
    """Compute the sum of numel from all tensors from a dict, except when the key is `step`."""
    ret = 0
    for k, v in dct.items():
        if k != "step" and torch.is_tensor(v):
            ret += v.numel()
    return ret


def assert_equal(a, b):
    assert a == b, f"{a} != {b}"


def spawn_and_init_multiple_groups(fn, args=None, **spawn_kwargs):
    if args is None:
        args = ()

    run_fn = functools.partial(init_and_run, fn, args)
    spawn_for_all_world_sizes(run_fn, **spawn_kwargs)


def _find_my_group_index(grouped_ranks):
    """Return the index corresponding to the MoE group of the current process."""
    my_rank = torch.distributed.get_rank()
    for i, group in enumerate(grouped_ranks):
        if my_rank in group:
            return i
    raise RuntimeError(f"Unable to find process rank {my_rank} in the set of grouped ranks {grouped_ranks}.")


def get_moe_group(moe_expert_count=2):
    """Return a process group for initializing a MoE layer."""
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()

        # If you have more experts than the world size.
        if world_size <= moe_expert_count:
            assert moe_expert_count % world_size == 0
            moe_groups = [[i] for i in range(world_size)]

        # If you have a larger world size than experts.
        else:
            assert world_size % moe_expert_count == 0
            ranks_per_group = world_size // moe_expert_count
            moe_groups = [[i + j * moe_expert_count for j in range(ranks_per_group)] for i in range(moe_expert_count)]

        moe_pgs = [torch.distributed.new_group(g) for g in moe_groups]

        # Find the index in the set of moe_groups which contains the current rank.
        my_group_idx = _find_my_group_index(moe_groups)
        return moe_pgs[my_group_idx]
    else:
        return torch.distributed.new_group([torch.distributed.get_rank()])


def init_and_run(fn, args, rank, world_size, filename, filename_rpc):
    """Initialize and run the unit test for testing replicated MoE groups."""
    dist_init(rank, world_size, filename, filename_rpc)
    torch.cuda.set_device(rank)
    group = torch.distributed.new_group()
    # Specify the moe_group used to initialize the MoE layers with.
    fn(rank, group, *args, expert_group=get_moe_group())


class TestOptimizerUtils(DistributedTest):
    @parameterized.expand(
        [[functools.partial(SGD, momentum=0.9), True], [SGD, False], [Adam, False], [Adadelta, True], [Adam, True]],
        name_func=rename_test,
    )
    def test_consolidate_optimizer(self, optim_fn, transformer):
        config = {"mixed_precision": True, "flatten_parameters": True}
        config["compute_dtype"] = torch.float32
        test_fn = functools.partial(
            self._test_consolidated_optimizer, config, optim_fn=optim_fn, transformer=transformer
        )

        spawn_and_init(test_fn, world_sizes=[min(torch.cuda.device_count(), 4)])

    @parameterized.expand(
        [[SGD, False], [Adam, False]],
        name_func=rename_test,
    )
    def test_consolidate_optimizer_diff_world_size(self, optim_fn, transformer):
        if torch.cuda.device_count() < 4:
            raise unittest.SkipTest("This test requires at least 4 GPUs.")
        config = {"mixed_precision": True, "flatten_parameters": True}
        config["compute_dtype"] = torch.float32
        test_fn = functools.partial(self._test_consolidated_optimizer, config, optim_fn=Adam, transformer=transformer)

        spawn_and_init_multiple_groups(test_fn, world_sizes=[min(torch.cuda.device_count(), 4)])

    @classmethod
    def _test_consolidated_optimizer(
        self, config, rank, group, optim_fn=torch.optim.SGD, transformer=False, expert_group=None
    ):
        """FSDP.gather_full_optim_state_dict() should return something very similar to optimizer.state_dict()"""
        # Establish reference behavior.
        if transformer:
            unwrapped_model = TransformerWithSharedParams(group, wrapper_config=config).cuda()
            fsdp = self.get_wrapped_model(group, config=config).cuda()
        else:
            unwrapped_model = MixtureOfExperts(group, wrapper_config=None, expert_group=expert_group).cuda()
            fsdp = FullyShardedDataParallel(
                MixtureOfExperts(group, wrapper_config=config, expert_group=expert_group)
            ).cuda()

        try:
            fsdp_optim = optim_fn(
                fsdp.parameters(),
                lr=0.01,
            )
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

        if not transformer and not expert_group:
            no_broadcast_children = [x for x in get_fsdp_instances(fsdp) if x.no_broadcast_optim_state]
            assert len(no_broadcast_children) == 1, f"Length of non shared params {len(no_broadcast_children)}"
            assert get_fsdp_instances(fsdp)[-1].no_broadcast_optim_state
        torch.cuda.empty_cache()
        cuda_gb_before = torch.cuda.memory_stats(fsdp.rank)["allocated_bytes.all.current"] / 1024**3
        tstart = time()
        sd = fsdp.gather_full_optim_state_dict(fsdp_optim, recipient_rank=0)
        duration = time() - tstart
        assert duration < fsdp.world_size, f"gather optim state took {duration} seconds, suspect change in _consolidate"

        cuda_gb_after = torch.cuda.memory_stats(fsdp.rank)["allocated_bytes.all.current"] / 1024**3
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

        if expert_group:
            sd_state = recursive_copy_to_device(sd["state"], non_blocking=False, device="cpu")
            orig_state = recursive_copy_to_device(unwrapped_sd["state"], non_blocking=False, device="cpu")

            assert_equal(len(sd_state.keys()), len(orig_state.keys()))

            assert_equal(
                sum([all_tensors_numel_except_for_step(v) for k, v in sd_state.items()]),
                sum([all_tensors_numel_except_for_step(v) for k, v in orig_state.items()]),
            )
            return

        assert "uncollected_local_ids" in sd
        sd_copy = copy.deepcopy(sd)
        unflat_state = sd_copy["state"]
        shard_sd = fsdp.get_shard_from_optim_state_dict(sd_copy)
        shard_sd = recursive_copy_to_device(shard_sd, non_blocking=False, device="cpu")
        state_after_get_shard = sd_copy["state"]
        # sd is changed in-place in case there are extra states.
        assert not objects_are_equal(unflat_state, state_after_get_shard)
        del sd_copy

        assert_equal(len(sd["state"]), len(unwrapped_sd["state"]))
        assert_equal(len(sd["param_groups"][0]["params"]), len(unwrapped_sd["param_groups"][0]["params"]))
        assert_equal(
            sum([all_tensors_numel_except_for_step(v) for k, v in sd["state"].items()]),
            sum([all_tensors_numel_except_for_step(v) for k, v in unwrapped_sd["state"].items()]),
        )

        original_shard_sd = fsdp_optim.state_dict()
        assert_equal(len(shard_sd["state"]), len(original_shard_sd["state"]))
        assert_equal(shard_sd.keys(), original_shard_sd.keys())
        original_shard_sd = recursive_copy_to_device(original_shard_sd, non_blocking=False, device="cpu")
        # Before asserting that the dicts are equal, we check keys individually to allow nice tracebacks.
        assert_equal(
            [all_tensors_numel_except_for_step(v) for k, v in shard_sd["state"].items()],
            [all_tensors_numel_except_for_step(v) for k, v in original_shard_sd["state"].items()],
        )
        assert_equal(
            [v for k, v in shard_sd["param_groups"][0].items()],
            [v for k, v in original_shard_sd["param_groups"][0].items()],
        )
        objects_are_equal(shard_sd["state"], original_shard_sd["state"], raise_exception=True)
        objects_are_equal({k: shard_sd[k] for k in original_shard_sd}, original_shard_sd, raise_exception=True)

    @parameterized.expand(
        [(True,), (False,)],
        name_func=rename_test,
    )
    def test_model_with_unused_params(self, wrap_l2):
        """Test handling of model with unused params by gather_full_optim_state_dict()"""
        test_fn = functools.partial(self._test_model_with_unused_params, wrap_l2=wrap_l2)
        spawn_and_init(test_fn, world_sizes=[2])

    @classmethod
    def _test_model_with_unused_params(self, rank, pg, wrap_l2):
        model = ModelWithUnusedParams(wrap_l2).cuda()
        data = torch.rand(4).cuda().requires_grad_(True)
        model = FullyShardedDataParallel(model)
        optim = SGD(model.parameters(), momentum=0.9, lr=0.1)
        out = model(data).sum()
        out.backward()
        optim.step()
        model.zero_grad(set_to_none=True)
        sd = model.gather_full_optim_state_dict(optim)
        if rank == 0:
            shard_sd = model.get_shard_from_optim_state_dict(sd)
            orig_sd = optim.state_dict()
            orig_sd = recursive_copy_to_device(orig_sd, non_blocking=False, device="cpu")
            objects_are_equal(shard_sd, orig_sd, raise_exception=True)
        else:
            assert sd is None, sd

    def test_named_params_ordering(self):
        """Test assumption of consolidate_optimizer_state_dict"""
        group = DummyProcessGroup(0, 1)
        model = TransformerWithSharedParams(group)
        named_pars = [p for n, p in model.named_parameters()]
        for i, p in enumerate(model.parameters()):
            objects_are_equal(p, named_pars[i], raise_exception=True)

    def test_is_singleton_tensor(self):
        """Test is_singleton_tensor function"""
        assert is_singleton_tensor(torch.tensor(4.0))
        assert not is_singleton_tensor(torch.tensor([4.0]))
        assert not is_singleton_tensor(torch.tensor([4.0, 5.0]))
        assert not is_singleton_tensor([4.0])
        assert not is_singleton_tensor(4.0)


class ModelWithUnusedParams(nn.Module):
    def __init__(self, wrap_l2):
        super().__init__()
        self.l = nn.Linear(4, 4)
        # unused param must be wrapped, otherwise, due to flatten, it
        # is always used.
        self.not_trained = nn.Linear(4, 4).requires_grad_(False)
        self.not_trained = FullyShardedDataParallel(self.not_trained)
        # optionally testing a used param after the unused one by
        # wrapping it.
        self.l2 = nn.Linear(4, 4)
        if wrap_l2:
            # When wrapping happens, the unused param will be in the middle
            # of the param list (for optimizer state dict), not at the
            # end. This way, we can test the handling code in more corner
            # cases.
            self.l2 = FullyShardedDataParallel(self.l2)

    def forward(self, x):
        with torch.no_grad():
            y = self.not_trained(x)
        return self.l2(self.l(x)) - y
