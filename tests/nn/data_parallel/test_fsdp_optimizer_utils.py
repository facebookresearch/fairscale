import functools

from parameterized import parameterized
import torch

from fairscale.optim.utils import recursive_copy_to_device
from fairscale.utils.testing import objects_are_equal

from .test_fsdp import DistributedTest, TransformerWithSharedParams, assert_equal, rename_test, spawn_and_init


class TestOptimizerUtils(DistributedTest):
    @parameterized.expand(
        [
            [functools.partial(torch.optim.SGD, momentum=0.9)],
            [torch.optim.SGD],
            [torch.optim.Adam],
            [torch.optim.Adadelta],
        ],
        name_func=rename_test,
    )
    def test_consolidate_optimizer(self, optim_fn):
        config = {"mixed_precision": True}
        test_fn = functools.partial(self._test_consolidated_optimizer, config, optim_fn=optim_fn)
        spawn_and_init(test_fn)

    @classmethod
    def _test_consolidated_optimizer(self, config, rank, group, optim_fn=torch.optim.SGD):
        """FSDP.gather_full_optim_state_dict() should return something very similar to optimizer.state_dict()"""
        # Establish reference behavior.
        fsdp = self.get_wrapped_model(group, cuda_first=False, config=config)
        unwrapped_model = TransformerWithSharedParams(group).cuda()
        try:
            fsdp_optim = optim_fn(fsdp.parameters(), lr=0.01,)
            optim_unwrapped = optim_fn(unwrapped_model.parameters(), lr=0.01)
        except TypeError:  # AdaScale
            fsdp_optim = optim_fn(fsdp.parameters())
            optim_unwrapped = optim_fn(unwrapped_model.parameters())

        fsdp_optim.zero_grad()
        optim_unwrapped.zero_grad()

        src_ids, tgt_ids = fsdp.module.get_input(torch.device("cuda"))
        output = fsdp(src_ids, tgt_ids)
        loss = fsdp.module.get_loss((src_ids, tgt_ids), output).to("cuda")
        fsdp.module.run_backward(loss)
        fsdp_optim.step()
        # fsdp.consolidate_optim_state_dict(fsdp_optim, recipient_rank=0)

        output = unwrapped_model(src_ids, tgt_ids)
        loss = unwrapped_model.get_loss((src_ids, tgt_ids), output)
        unwrapped_model.run_backward(loss)
        optim_unwrapped.step()
        unwrapped_sd = optim_unwrapped.state_dict()

        n_pars = len(list(unwrapped_model.parameters()))

        # torch.save(fsdp._all_optimizer_states, f"all_optim_states_world_size_{fsdp.world_size}.pt")
        sd = fsdp.gather_full_optim_state_dict(fsdp_optim, recipient_rank=-1)
        # assert_equal(len(fsdp._all_optimizer_states), fsdp.world_size)
        torch.save(sd, f"fsdp_consolidated_{fsdp.world_size}.pt")

        assert_equal(len(sd["state"]), len(unwrapped_sd["state"]))
        assert_equal(len(sd["param_groups"][0]["params"]), len(unwrapped_sd["param_groups"][0]["params"]))

        shard_sd = fsdp.get_shard_from_optim_state_dict(sd)

        original_shard_sd = fsdp_optim.state_dict()
        assert_equal(len(shard_sd["state"]), len(original_shard_sd["state"]))
        assert objects_are_equal(
            shard_sd, recursive_copy_to_device(original_shard_sd, non_blocking=False, device="cpu")
        )
