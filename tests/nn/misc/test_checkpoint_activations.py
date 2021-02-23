# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Test fairscale.nn.misc.checkpoint_activations API."""

import unittest

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from fairscale.nn.misc.checkpoint_activations import checkpoint_wrapper


def get_cuda_mem_allocated():
    """Helper to get cuda memory allocated if possible."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    else:
        return 0


class Model(nn.Module):
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
            x = torch_checkpoint(self.ffn, x)
        else:
            x = self.ffn(x)
        return self.out(x)


class TestComparisonToPyTorch(unittest.TestCase):
    def _test_checkpoint_wrapper(self, device):
        def get_loss_and_gnorm(model, input):
            ret = {}
            ret["mem_0"] = get_cuda_mem_allocated()
            model.zero_grad()
            loss = model(input).sum()
            ret["mem_after_fwd"] = get_cuda_mem_allocated()
            loss.backward()
            ret["mem_after_bwd"] = get_cuda_mem_allocated()
            gnorm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.parameters()]))
            ret["loss"] = loss.item()
            ret["gnorm"] = gnorm.item()
            return ret

        input = torch.rand(2, 16, 32).requires_grad_(True)
        model = Model().to(device)
        no_cpt = get_loss_and_gnorm(model, input.to(device))

        model = Model(use_pytorch_checkpoint=True).to(device)
        pyt_cpt = get_loss_and_gnorm(model, input.to(device))

        model = Model(use_fairscale_checkpoint=True).to(device)
        fairscale_cpt = get_loss_and_gnorm(model, input.to(device))

        model = Model(use_fairscale_checkpoint=True, offload_to_cpu=True).to(device)
        fairscale_cpt_offload = get_loss_and_gnorm(model, input.to(device))

        # Check for correctness.
        torch.testing.assert_allclose(no_cpt["loss"], pyt_cpt["loss"])
        torch.testing.assert_allclose(no_cpt["gnorm"], pyt_cpt["gnorm"])

        torch.testing.assert_allclose(no_cpt["loss"], fairscale_cpt["loss"])
        torch.testing.assert_allclose(no_cpt["gnorm"], fairscale_cpt["gnorm"])

        torch.testing.assert_allclose(no_cpt["loss"], fairscale_cpt_offload["loss"])
        torch.testing.assert_allclose(no_cpt["gnorm"], fairscale_cpt_offload["gnorm"])

        # Check for memory usage for cuda only.
        if device == torch.device("cpu"):
            return
        for d in [no_cpt, pyt_cpt, fairscale_cpt, fairscale_cpt_offload]:
            del d["loss"]
            del d["gnorm"]
        assert no_cpt == {"mem_0": 38912, "mem_after_fwd": 64000, "mem_after_bwd": 74240}, no_cpt
        assert pyt_cpt == {"mem_0": 38912, "mem_after_fwd": 43520, "mem_after_bwd": 74240}, pyt_cpt
        assert fairscale_cpt == {"mem_0": 38912, "mem_after_fwd": 43520, "mem_after_bwd": 74240}, fairscale_cpt
        assert fairscale_cpt_offload == {
            "mem_0": 38912,
            "mem_after_fwd": 43520,
            "mem_after_bwd": 74240,
        }, fairscale_cpt_offload

    def test_checkpoint_wrapper_cpu(self):
        self._test_checkpoint_wrapper(device=torch.device("cpu"))

    @unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
    def test_checkpoint_wrapper_cuda(self):
        self._test_checkpoint_wrapper(device=torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
