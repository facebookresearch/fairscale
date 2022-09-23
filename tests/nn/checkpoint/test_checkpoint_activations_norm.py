# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test checkpoint_wrapper with normalization layers. """

import pytest
import torch
from torch.nn import BatchNorm2d, LayerNorm, Linear, Sequential
from torch.optim import SGD

from fairscale.fair_dev.testing.testing import objects_are_equal
from fairscale.internal import torch_version
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

NORM_TYPES = [LayerNorm, BatchNorm2d]
MP_TYPES = ["fp32", "fp16", "call_half"]


def get_model(norm_type, checkpointed, mixed_precision):
    assert norm_type in NORM_TYPES, norm_type
    assert checkpointed in [True, False], checkpointed
    assert mixed_precision in MP_TYPES

    model = Sequential(Linear(3, 2), norm_type(2))

    if mixed_precision == "fp16":
        # Set param.data and buffers as fp16
        for p in model.parameters():
            p.data = p.data.half()
        for m in model:
            for n, b in m.named_buffers():
                setattr(m, n, b.half())
    elif mixed_precision == "call_half":
        model.half()

    if checkpointed:
        model = checkpoint_wrapper(model)

    return model


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("norm_type", NORM_TYPES)
@pytest.mark.parametrize("mixed_precision", MP_TYPES)
def test_norm(device, norm_type, mixed_precision):
    """Test checkpoint_wrapper with different norm layers."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Skip due to lack of GPU")

    # Get input, ref, checkpoint models and make them equal.
    in_data = torch.rand(2, 2, 3, 3).to(device)
    m_ref = get_model(norm_type, False, mixed_precision).to(device)
    m_cpt = get_model(norm_type, True, mixed_precision).to(device)
    m_cpt.load_state_dict(m_ref.state_dict())

    if torch_version() >= (1, 6, 0):
        # This assert fails on 1.5.1.
        assert objects_are_equal(m_ref.state_dict(), m_cpt.state_dict())

    if mixed_precision != "fp32":
        in_data = in_data.half()

    # Needed due to checkpointing.
    in_data.requires_grad = True
    for model in (m_ref, m_cpt):
        optim = SGD(model.parameters(), lr=0.1)
        if device == "cpu" and mixed_precision != "fp32":
            # Got: RuntimeError: "batch_norm"/"layer_norm" not implemented for 'Half'.
            with pytest.raises(RuntimeError):
                out = model(in_data)
            return
        else:
            # Everything else work.
            out = model(in_data)
        out.sum().backward()
        optim.step()

    if torch_version() >= (1, 6, 0):
        assert objects_are_equal(m_ref.state_dict(), m_cpt.state_dict())
