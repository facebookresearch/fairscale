import functools

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


def first_tensor_shape(dct):
    for k, v in dct.items():
        if torch.is_tensor(v):
            return v.numel()
    raise ValueError("found no tensors")


def assert_equal(a, b):
    assert a == b, f"{a} != {b}"


class TestOptimizerUtils(DistributedTest):
    @parameterized.expand(
        [[functools.partial(SGD, momentum=0.9), False], [SGD, False], [Adam, False], [Adadelta, True]],
        name_func=rename_test,
    )
    def test_consolidate_optimizer(self, optim_fn, transformer):
        config = {"mixed_precision": True, "flatten_parameters": True}
        test_fn = functools.partial(
            self._test_consolidated_optimizer, config, optim_fn=optim_fn, transformer=transformer
        )
        spawn_and_init(test_fn)

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
        except TypeError:  # AdaScale
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

        # first_key = unwrapped_sd['state'][0].keys()
        sd = fsdp.gather_full_optim_state_dict(fsdp_optim, recipient_rank=None)

        assert_equal(len(sd["state"]), len(unwrapped_sd["state"]))
        assert_equal(len(sd["param_groups"][0]["params"]), len(unwrapped_sd["param_groups"][0]["params"]))
        assert_equal(
            sum([first_tensor_shape(v) for k, v in sd["state"].items()]),
            sum([first_tensor_shape(v) for k, v in unwrapped_sd["state"].items()]),
        )

        shard_sd = fsdp.get_shard_from_optim_state_dict(sd)

        original_shard_sd = fsdp_optim.state_dict()
        assert_equal(len(shard_sd["state"]), len(original_shard_sd["state"]))
        assert_equal(shard_sd.keys(), original_shard_sd.keys())
        torch.save(shard_sd, f"new_shard_{fsdp.world_size}.pt")
        original_shard_sd = recursive_copy_to_device(original_shard_sd, non_blocking=False, device="cpu")

        assert_equal(
            sum([first_tensor_shape(v) for k, v in shard_sd["state"].items()]),
            sum([first_tensor_shape(v) for k, v in original_shard_sd["state"].items()]),
        )
        if shard_sd["state"]:
            assert objects_are_equal(shard_sd["state"][0], original_shard_sd["state"][0])
        assert objects_are_equal(shard_sd["state"], original_shard_sd["state"])

    def test_named_params_ordering(self):
        """Test assumption of consolidate_optimizer_state_dict"""
        group = DummyProcessGroup(0, 1)
        model = TransformerWithSharedParams(group)
        named_pars = [p for n, p in model.named_parameters()]
        for i, p in enumerate(model.parameters()):
            assert p.shape == named_pars[i].shape
