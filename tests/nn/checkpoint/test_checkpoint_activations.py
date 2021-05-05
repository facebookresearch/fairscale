# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Test fairscale.nn.misc.checkpoint_activations API."""

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint_wrapper

from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from fairscale.utils.testing import skip_if_no_cuda, torch_version
from fairscale.nn.misc import checkpoint_wrapper as deprecated_checkpoint_wrapper

def get_cuda_mem_allocated():
    """Helper to get cuda memory allocated if possible."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    else:
        return 0


def get_loss_and_gnorm(model, input):
    """Helper to run a forward/backward pass and return results in a dict."""
    ret = {}

    ret["mem_0"] = get_cuda_mem_allocated()
    ret["mem_peak"] = 0
    if ret["mem_0"] > 0:
        torch.cuda.reset_peak_memory_stats()

    model.zero_grad()
    loss = model(input).sum()
    ret["mem_after_fwd"] = get_cuda_mem_allocated()

    loss.backward()
    ret["mem_after_bwd"] = get_cuda_mem_allocated()

    gnorm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.parameters()]))
    ret["loss"] = loss.item()
    ret["gnorm"] = gnorm.item()

    if ret["mem_0"] > 0:
        ret["mem_peak"] = torch.cuda.max_memory_allocated()

    return ret


class BasicModel(nn.Module):
    """Basic model with a single FFN being checkpointed.

       Used for extensive checkings: equivalency with non-checkpoint, torch-checkpoint, etc.
    """

    def __init__(self, use_pytorch_checkpoint=False, use_fairscale_checkpoint=False, **kwargs):
        super().__init__()
        torch.manual_seed(0)  # make sure weights are deterministic.
        assert not (
            use_pytorch_checkpoint and use_fairscale_checkpoint
        ), "Cannot use both pytorch and fairscale checkpointing mechanisms."
        self.use_pytorch_checkpoint = use_pytorch_checkpoint
        self.ffn = nn.Sequential(
            nn.Linear(32, 128),
            # add a Dropout layer to test RNG save/restore
            nn.Dropout(p=0.5),
            nn.Linear(128, 32),
        )
        if use_fairscale_checkpoint:
            self.ffn = checkpoint_wrapper(self.ffn, **kwargs)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        if self.use_pytorch_checkpoint:
            x = torch_checkpoint_wrapper(self.ffn, x)
        else:
            x = self.ffn(x)
        return self.out(x)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_basic(device):
    if "cuda" in device and not torch.cuda.is_available():
        pytest.skip("test requires a GPU")

    input = torch.rand(2, 16, 32).requires_grad_(True)
    model = BasicModel().to(device)
    no_cpt = get_loss_and_gnorm(model, input.to(device))

    model = BasicModel(use_pytorch_checkpoint=True).to(device)
    pyt_cpt = get_loss_and_gnorm(model, input.to(device))

    model = BasicModel(use_fairscale_checkpoint=True).to(device)
    fairscale_cpt = get_loss_and_gnorm(model, input.to(device))

    model = BasicModel(use_fairscale_checkpoint=True, offload_to_cpu=True).to(device)
    fairscale_cpt_offload = get_loss_and_gnorm(model, input.to(device))

    # Check for correctness.
    for key in "loss", "gnorm":
        if not (no_cpt[key] == pyt_cpt[key] == fairscale_cpt[key] == fairscale_cpt_offload[key]):
            print(no_cpt, pyt_cpt, fairscale_cpt, fairscale_cpt_offload)
            assert 0
        del no_cpt[key]
        del pyt_cpt[key]
        del fairscale_cpt[key]
        del fairscale_cpt_offload[key]

    # Check for memory usage for cuda only.
    if "cpu" in device:
        return

    mem_peaks = [98816, 103424, 103424, 107520]
    if torch_version() < (1, 7, 0):
        # Older torch behaves slightly differently
        mem_peaks = [102400, 103424, 103424, 107520]

    assert no_cpt == {"mem_0": 38912, "mem_peak": mem_peaks[0], "mem_after_fwd": 64000, "mem_after_bwd": 74240}, no_cpt
    assert pyt_cpt == {
        "mem_0": 38912,
        "mem_peak": mem_peaks[1],
        "mem_after_fwd": 43520,
        "mem_after_bwd": 74240,
    }, pyt_cpt
    assert fairscale_cpt == {
        "mem_0": 38912,
        "mem_peak": mem_peaks[2],
        "mem_after_fwd": 43520,
        "mem_after_bwd": 74240,
    }, fairscale_cpt
    assert fairscale_cpt_offload == {
        "mem_0": 38912,
        "mem_peak": mem_peaks[3],
        "mem_after_fwd": 43520,
        "mem_after_bwd": 74240,
    }, fairscale_cpt_offload


class CpuOffloadModel(nn.Module):
    """Model used to check cpu offload memory saving"""

    def __init__(self, enable_checkpoint=False, cpu_offload=False):
        super().__init__()

        torch.manual_seed(0)  # make sure weights are deterministic.

        # These numbers are picked to show cpu_offload memory saving.
        # Inner (recomputed) activation sizes need to be just right
        # to show the benefit.
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 8)),
            nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 4), nn.Linear(4, 4)),
            nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 8), nn.Linear(8, 2)),
        )

        if enable_checkpoint:
            for i, layer in enumerate(self.layers):
                # Only middle layer needs to have offloading
                self.layers[i] = checkpoint_wrapper(layer, cpu_offload if i == 1 else False)

    def forward(self, x):
        return self.layers(x)


@skip_if_no_cuda
def test_offload_memory():
    device = "cuda"

    input = torch.rand(60, 24, 4).requires_grad_(True)

    model = CpuOffloadModel().to(device)
    base = get_loss_and_gnorm(model, input.to(device))

    model = CpuOffloadModel(True).to(device)
    cpt = get_loss_and_gnorm(model, input.to(device))

    model = CpuOffloadModel(True, True).to(device)
    offload = get_loss_and_gnorm(model, input.to(device))

    for key in "loss", "gnorm":
        if not (base[key] == cpt[key] == offload[key]):
            # Use print to collect all debugging info.
            print(base, cpt, offload)
            assert 0
        del base[key]
        del cpt[key]
        del offload[key]

    ref_base = {"mem_0": 32256, "mem_peak": 334336, "mem_after_fwd": 274944, "mem_after_bwd": 41984}
    ref_cpt = {"mem_0": 32256, "mem_peak": 253952, "mem_after_fwd": 101888, "mem_after_bwd": 41984}
    ref_offload = {"mem_0": 32256, "mem_peak": 207872, "mem_after_fwd": 55808, "mem_after_bwd": 41984}

    if not (base == ref_base and cpt == ref_cpt and offload == ref_offload):
        # Use print to collect all debugging info.
        print(base, cpt, offload)
        assert 0


class MultiinMultioutModel(nn.Module):
    """Model used to check different inputs and outputs"""

    def __init__(self, multiout=False, checkpoint_config=0):
        super().__init__()
        torch.manual_seed(0)  # make sure weights are deterministic.

        self.multiout = multiout

        self.conv1 = nn.Sequential(nn.Conv2d(1, 5, 3), nn.ReLU(), nn.Conv2d(5, 5, 3))
        self.conv2 = nn.Sequential(nn.Conv2d(3, 5, 3), nn.ReLU(), nn.Conv2d(5, 5, 3))

        assert 0 <= checkpoint_config <= 3
        if checkpoint_config & 1:
            self.conv1 = checkpoint_wrapper(self.conv1)
        if checkpoint_config & (1 << 1):
            self.conv2 = checkpoint_wrapper(self.conv2)

    def forward(self, x1, x2=None):
        out1 = self.conv1(x1)
        out2 = self.conv2(x2)
        if self.multiout:
            return out1, out2
        return out1 + out2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("multiout", [True, False])
@pytest.mark.parametrize("checkpoint_config", [1, 2, 3])
def test_multiin_multiout(device, multiout, checkpoint_config):
    if "cuda" in device and not torch.cuda.is_available():
        pytest.skip("test requires a GPU")

    def train(model, in1, in2):
        out = model(in1, x2=in2)
        if isinstance(out, tuple):
            out = torch.cat(out)
        loss = out.sum()
        loss.backward()
        gnorm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.parameters()]))
        return {"loss": loss.item(), "gnorm": gnorm.item()}

    in1 = torch.rand(4, 1, 32, 32).requires_grad_(True)
    in2 = torch.rand(4, 3, 32, 32).requires_grad_(True)

    model = MultiinMultioutModel(multiout, 0).to(device)
    no_cpt = train(model, in1.to(device), in2.to(device))

    model = MultiinMultioutModel(multiout, checkpoint_config).to(device)
    cpt = train(model, in1.to(device), in2.to(device))

    for key in ["loss", "gnorm"]:
        if no_cpt[key] != cpt[key]:
            print(no_cpt, cpt)
            assert 0

def test_deprecated_path():

    ffn = nn.Sequential(
        nn.Linear(32, 128),
        nn.Dropout(p=0.5),
        nn.Linear(128, 32),
    )
    ffn = deprecated_checkpoint_wrapper(ffn, {})


