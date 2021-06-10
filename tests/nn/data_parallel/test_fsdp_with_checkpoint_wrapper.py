""" Test FSDP with an submodule that is FSDP(checkpoint_wrapper()). """

import tempfile

import pytest
import torch
from torch import nn
import torch.distributed
import torch.multiprocessing as mp

from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from fairscale.nn.data_parallel import FullyShardedDataParallel
from fairscale.utils.testing import dist_init, skip_if_single_gpu, teardown, torch_version


@skip_if_single_gpu
def test_train_and_eval_with_checkpointing():
    if torch_version() < (1, 6, 0):
        pytest.skip("older pytorch doesn't support reduce_scatter")

    temp_file_name = tempfile.mkstemp()[1]
    unused = tempfile.mkstemp()[1]
    world_size = 2

    mp.spawn(
        _test_func, args=(world_size, temp_file_name, unused), nprocs=world_size, join=True,
    )


def _test_func(rank, world_size, tempfile_name, unused):
    result = dist_init(rank, world_size, tempfile_name, unused)
    assert result, "Dist init failed"

    # Keep initialization deterministic.
    torch.manual_seed(0)

    model = FullyShardedDataParallel(SimpleModuleWithCheckpointing().cuda())
    optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Collect parameter sizes to ensure these stay consistent through the steps below.
    expected_param_shapes = {name: tuple(param.shape) for name, param in model.named_parameters()}

    # For clarify, this is what `expected_param_shapes` should look like depending on world size:
    if world_size == 1:
        assert expected_param_shapes == {
            "_fsdp_wrapped_module.flat_param": (24,),
            "_fsdp_wrapped_module._fpw_module.ffn.1._fsdp_wrapped_module.flat_param": (12,),
        }, expected_param_shapes
    else:
        assert expected_param_shapes == {
            "_fsdp_wrapped_module.flat_param": (12,),
            "_fsdp_wrapped_module._fpw_module.ffn.1._fsdp_wrapped_module.flat_param": (6,),
        }, expected_param_shapes

    torch.manual_seed(1 + rank)

    # Train for a step.
    _train_step(model, optim, expected_param_shapes)

    # Now do an eval step.
    _eval_step(model, optim, expected_param_shapes)

    # And finally do another train step.
    _train_step(model, optim, expected_param_shapes)

    teardown()


def _train_step(model, optim, expected_param_shapes):
    # Prepare for training step.
    optim.zero_grad(set_to_none=True)  # type: ignore
    model.train()

    # Create input and run forward pass.
    input = torch.randn(2, 3).cuda()
    loss = model(input).sum()
    _check_fwd_counter(model, 1)
    _check_params_and_grads(model, expected_param_shapes, grads_are_none=True)

    # Run backward pass.
    loss.backward()
    _check_fwd_counter(model, 0)
    _check_params_and_grads(model, expected_param_shapes, grads_are_none=False)

    # Finally, take a step.
    optim.step()
    _check_params_and_grads(model, expected_param_shapes, grads_are_none=False)


def _eval_step(model, optim, expected_param_shapes):
    optim.zero_grad(set_to_none=True)  # type: ignore
    model.eval()
    with torch.no_grad():
        input = torch.randn(2, 3).cuda()
        model(input).sum()
    _check_fwd_counter(model, 0)
    _check_params_and_grads(model, expected_param_shapes, grads_are_none=True)


def _check_params_and_grads(model, expected_param_shapes, grads_are_none=False):
    current_param_shapes = {name: tuple(param.shape) for name, param in model.named_parameters()}
    assert set(current_param_shapes.keys()) == set(expected_param_shapes.keys())
    for key, current_shape in current_param_shapes.items():
        expected_shape = expected_param_shapes[key]
        assert (
            current_shape == expected_shape
        ), f"Parameter {key} should have shape {expected_shape}, but found shape {current_shape}"

    for key, param in model.named_parameters():
        if grads_are_none:
            assert param.grad is None, f"Parameter {key} should not have a gradient in this context"
        else:
            assert param.grad is not None, f"Parameter {key} should have a gradient"
            current_shape = tuple(param.grad.shape)
            expected_shape = expected_param_shapes[key]
            assert (
                current_shape == expected_shape
            ), f"Parameter {key} should have gradient with shape {expected_shape}, but found shape {current_shape}"


def _check_fwd_counter(model, expected_value):
    current_value = model._fpw_module.ffn[1]._fsdp_wrapped_module.module._checkpoint_fwd_counter  # type: ignore
    assert (
        current_value == expected_value
    ), f"forward counter of checkpointed submodule should be {expected_value}, but found {current_value}"


class SimpleModuleWithCheckpointing(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(3, 3),
            FullyShardedDataParallel(checkpoint_wrapper(nn.Linear(3, 3), maintain_forward_counter=True)),
            nn.Linear(3, 3),
        )

    def forward(self, x):
        return self.ffn(x)
