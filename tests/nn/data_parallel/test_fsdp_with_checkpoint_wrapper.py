# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

""" Test FSDP with an submodule that is FSDP(checkpoint_wrapper()) or checkpoint_wrapper(FSDP()). """

import contextlib

import pytest
import torch
from torch import nn
import torch.distributed
import torch.multiprocessing as mp

from fairscale.fair_dev.testing.testing import dist_init, skip_if_single_gpu, teardown, temp_files_ctx
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP


@skip_if_single_gpu
@pytest.mark.parametrize("flatten", ["flat", "nonflat"])
@pytest.mark.parametrize("mixed_precision", ["fp16", "fp32"])
@pytest.mark.parametrize("amp_context", ["autocast", "noautocast"])
@pytest.mark.parametrize("half_input", ["halfin", "fullin"])
@pytest.mark.parametrize("fsdp_wrap_ckpt", ["F->C", "C->F"])
def test_train_and_eval_with_checkpointing(flatten, mixed_precision, amp_context, half_input, fsdp_wrap_ckpt):

    flatten = flatten == "flat"
    mixed_precision = mixed_precision == "fp16"
    amp_context = amp_context == "autocast"
    half_input = half_input == "halfin"
    fsdp_wrap_ckpt = fsdp_wrap_ckpt == "F->C"

    # Expecting an known bug in 4 out of 32 cases.
    if fsdp_wrap_ckpt and mixed_precision and not flatten:
        pytest.skip("known bug")

    world_size = 2

    with temp_files_ctx(2) as (temp_file_name, unused):
        mp.spawn(
            _test_func,
            args=(
                world_size,
                temp_file_name,
                unused,
                flatten,
                mixed_precision,
                amp_context,
                half_input,
                fsdp_wrap_ckpt,
            ),
            nprocs=world_size,
            join=True,
        )


def _test_func(
    rank, world_size, tempfile_name, unused, flatten, mixed_precision, amp_context, half_input, fsdp_wrap_ckpt
):
    result = dist_init(rank, world_size, tempfile_name, unused)
    assert result, "Dist init failed"

    # Keep initialization deterministic.
    torch.manual_seed(0)

    model = FSDP(
        SimpleModuleWithCheckpointing(flatten, mixed_precision, fsdp_wrap_ckpt).cuda(),
        flatten_parameters=flatten,
        mixed_precision=mixed_precision,
    )
    optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Collect parameter sizes to ensure these stay consistent through the steps below.
    expected_param_shapes = {name: tuple(param.shape) for name, param in model.named_parameters()}

    # For clarity, this is what `expected_param_shapes` should look like depending on world size:
    if not flatten:
        assert expected_param_shapes == {
            "ffn.0.weight": (5,),
            "ffn.0.bias": (2,),
            "ffn.1.weight": (5,),
            "ffn.1.bias": (2,),
            "ffn.2.weight": (5,),
            "ffn.2.bias": (2,),
        }
    else:
        assert expected_param_shapes == {
            "_fsdp_wrapped_module.flat_param_0": (12,),
            "_fsdp_wrapped_module._fpw_module.ffn.1._fsdp_wrapped_module.flat_param_0": (6,),
        }, expected_param_shapes

    torch.manual_seed(1 + rank)

    # Train for a step.
    _train_step(model, optim, expected_param_shapes, amp_context, mixed_precision, half_input)

    # Now do an eval step.
    _eval_step(model, optim, expected_param_shapes, amp_context, mixed_precision, half_input)

    # And finally do another train step.
    _train_step(model, optim, expected_param_shapes, amp_context, mixed_precision, half_input)

    teardown()


def _train_step(model, optim, expected_param_shapes, amp_context, mixed_precision, half_input):
    # Prepare for training step.
    optim.zero_grad()
    model.train()

    # Create input and run forward pass.
    input = torch.randn(2, 3).cuda()

    # Make it FP16 when it is OK to do so.
    if (amp_context and half_input) or (mixed_precision and half_input):
        input = input.half()

    context = contextlib.suppress()
    if amp_context:
        context = torch.cuda.amp.autocast(True)

    with context:
        loss = model(input).sum()

    _check_params(model, expected_param_shapes)

    # Run backward pass.
    loss.backward()
    _check_params(model, expected_param_shapes)

    # Finally, take a step.
    optim.step()
    _check_params(model, expected_param_shapes)


def _eval_step(model, optim, expected_param_shapes, amp_context, mixed_precision, half_input):
    optim.zero_grad()
    model.eval()
    with torch.no_grad():
        input = torch.randn(2, 3).cuda()
        if (amp_context and half_input) or (mixed_precision and half_input):
            input = input.half()
        context = contextlib.suppress()
        if amp_context:
            context = torch.cuda.amp.autocast(True)
        with context:
            model(input).sum()
    _check_params(model, expected_param_shapes)


def _check_params(model, expected_param_shapes):
    current_param_shapes = {name: tuple(param.shape) for name, param in model.named_parameters()}
    assert set(current_param_shapes.keys()) == set(expected_param_shapes.keys())
    for key, current_shape in current_param_shapes.items():
        expected_shape = expected_param_shapes[key]
        assert (
            current_shape == expected_shape
        ), f"Parameter {key} should have shape {expected_shape}, but found shape {current_shape}"


class SimpleModuleWithCheckpointing(nn.Module):
    def __init__(self, flatten, mixed_precision, fsdp_wrap_ckpt):
        super().__init__()
        if fsdp_wrap_ckpt:
            middle_module = FSDP(
                checkpoint_wrapper(nn.Linear(3, 3)), flatten_parameters=flatten, mixed_precision=mixed_precision
            )
        else:
            middle_module = checkpoint_wrapper(
                FSDP(nn.Linear(3, 3), flatten_parameters=flatten, mixed_precision=mixed_precision)
            )

        self.ffn = nn.Sequential(nn.Linear(3, 3), middle_module, nn.Linear(3, 3))

    def forward(self, x):
        return self.ffn(x)
