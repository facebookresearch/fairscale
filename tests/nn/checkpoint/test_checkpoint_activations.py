# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Test fairscale.nn.misc.checkpoint_activations API."""

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint_wrapper

from fairscale.fair_dev.testing.testing import skip_if_no_cuda
from fairscale.internal import torch_version
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper, disable_checkpointing
from fairscale.nn.misc import FlattenParamsWrapper
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
@pytest.mark.skipif(
    torch_version() >= (1, 13, 0),
    reason="mem_peak behavior changed for torch 1.13 and above",
)
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
    if torch_version() >= (1, 12, 0):
        pytest.skip("to be fixed")

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

    # Check if import works as before.
    # from fairscale.nn.misc.checkpoint_activations import checkpoint_wrapper
    from fairscale.nn import checkpoint_wrapper

    ffn = nn.Sequential(
        nn.Linear(32, 128),
        nn.Dropout(p=0.5),
        nn.Linear(128, 32),
    )
    ffn = checkpoint_wrapper(ffn, {})

    # Check if direct import works as before.
    ffn = nn.Sequential(
        nn.Linear(32, 128),
        nn.Dropout(p=0.5),
        nn.Linear(128, 32),
    )
    ffn = deprecated_checkpoint_wrapper(ffn, {})


@skip_if_no_cuda
def test_list_input():
    """Test checkpointing with input argument type being a list.

    Note: Testing shows that PyTorch's torch.utils.checkpoint function does not pass this test.
    """
    count = 0

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Linear(2, 2)

        def forward(self, x):
            nonlocal count
            count += 1
            y = []
            for i in x:
                y.append(self.conv(i))
            return y

    model = nn.Sequential(checkpoint_wrapper(Model()), Model()).cuda()
    in_data1 = torch.rand(4, 2).cuda()
    in_data2 = torch.rand(4, 2).cuda()

    # Forward. Count should be 2 for 2 modules.
    out = model([in_data1, in_data2])
    loss = sum(x.sum() for x in out)
    assert count == 2, f"Incorrect count {count}"

    # Backward. Adds 1 more forward call due to checkpoint.
    loss.backward()
    assert count == 3, f"Incorrect count {count}"


def test_checkpoint_disabling():
    """Test to check new disable_checkpoint() API added to checkpoint_wrapper."""

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnt = 0
            self.linear = nn.Linear(2, 2)

        def forward(self, x):
            self.cnt += 1
            y = []
            for i in x:
                y.append(self.linear(i))
            return y

    x = torch.rand(4, 2)
    model1 = checkpoint_wrapper(TestModel())
    model2 = checkpoint_wrapper(TestModel())

    # Forward. cnt += 1
    y = model1(x)
    y = sum(i.sum() for i in y)
    # Backward. cnt += 1
    y.backward()
    assert model1.cnt == 2

    with disable_checkpointing():
        # Forward. cnt += 1
        y = model2(x)
        y = sum(i.sum() for i in y)
        # Backward. cnt remains same as checkpointing is disabled
        y.backward()
    assert model2.cnt == 1


def test_checkpoint_requires_grad():
    """Test to check checkpointing when outputs do not require gradient."""

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnt = 0
            self.linear = nn.Linear(2, 2)

        def forward(self, x):
            self.cnt += 1
            return self.linear(x)

    x = torch.rand(4, 2)
    model = nn.Sequential(
        checkpoint_wrapper(TestModel()),
        checkpoint_wrapper(TestModel()),
        checkpoint_wrapper(TestModel()),
        checkpoint_wrapper(TestModel()),
    )
    model[0].requires_grad_(False)
    model[1].requires_grad_(False)
    model[2].requires_grad_(False)

    y = model(x)
    y = y.sum()
    y.backward()

    # Since only last model needs grad, we only run forward twice for it
    assert model[0].cnt == 1
    assert model[1].cnt == 1
    assert model[2].cnt == 1
    assert model[3].cnt == 2

    # Now test with first model needing grad
    model = nn.Sequential(
        checkpoint_wrapper(TestModel()),
        checkpoint_wrapper(TestModel()),
        checkpoint_wrapper(TestModel()),
        checkpoint_wrapper(TestModel()),
    )
    model[0].requires_grad_(True)
    model[1].requires_grad_(False)
    model[2].requires_grad_(False)

    y = model(x)
    y = y.sum()
    y.backward()

    # Since first model needs grad, all models need grad, so we run forward twice for all
    assert model[0].cnt == 2
    assert model[1].cnt == 2
    assert model[2].cnt == 2
    assert model[3].cnt == 2

    # Stress test with multiple inputs/outputs, of which some are not Tensor
    class TestModel2(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnt = 0
            self.linear = nn.Linear(2, 2)

        def forward(self, x, y, z):
            self.cnt += 1
            z = z + [self.cnt]
            return self.linear(x + y), z, ["hi"]

    model1 = checkpoint_wrapper(TestModel())
    model2 = checkpoint_wrapper(TestModel())
    model3 = checkpoint_wrapper(TestModel2())
    model4 = checkpoint_wrapper(TestModel())
    model1.requires_grad_(False)
    model2.requires_grad_(False)

    y = model4(model3(model1(x), model2(x), ["bye"])[0])
    y = y.sum()
    y.backward()

    assert model1.cnt == 1
    assert model2.cnt == 1
    assert model3.cnt == 2
    assert model4.cnt == 2

    model1 = checkpoint_wrapper(TestModel())
    model2 = checkpoint_wrapper(TestModel())
    model3 = checkpoint_wrapper(TestModel2())
    model4 = checkpoint_wrapper(TestModel())
    model2.requires_grad_(False)

    y = model4(model3(model1(x), model2(x), ["bye"])[0])
    y = y.sum()
    y.backward()

    assert model1.cnt == 2
    assert model2.cnt == 1
    assert model3.cnt == 2
    assert model4.cnt == 2

    # Test flattened pararameters
    model = nn.Sequential(
        FlattenParamsWrapper(checkpoint_wrapper(TestModel())),
        FlattenParamsWrapper(checkpoint_wrapper(TestModel())),
        FlattenParamsWrapper(checkpoint_wrapper(TestModel())),
        FlattenParamsWrapper(checkpoint_wrapper(TestModel())),
    )
    model[0].requires_grad_(False)
    model[1].requires_grad_(False)

    y = model(x)
    y = y.sum()
    y.backward()

    assert model[0].cnt == 1
    assert model[1].cnt == 1
    assert model[2].cnt == 2
    assert model[3].cnt == 2
