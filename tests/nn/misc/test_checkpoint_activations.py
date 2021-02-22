# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Test fairscale.nn.misc.checkpoint_activations
"""

import unittest

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from fairscale.nn.misc.checkpoint_activations import checkpoint_wrapper


class Model(nn.Module):
    def __init__(self, use_pytorch_checkpoint=False, use_fairscale_checkpoint=False, **kwargs):
        super().__init__()
        torch.manual_seed(0)  # make sure weights are deterministic.
        assert not (use_pytorch_checkpoint and use_fairscale_checkpoint), "Doesn't make sense to use both"
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
            x = checkpoint(self.ffn, x)
        else:
            x = self.ffn(x)
        return self.out(x)


class TestComparisonToPyTorch(unittest.TestCase):
    def _test_checkpoint_wrapper(self, device):
        def get_loss_and_gnorm(model, input):
            ret = {}
            ret["mem_0"] = torch.cuda.memory_allocated()
            model.zero_grad()
            loss = model(input).sum()
            ret["mem_after_fwd"] = torch.cuda.memory_allocated()
            loss.backward()
            ret["mem_after_bwd"] = torch.cuda.memory_allocated()
            gnorm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.parameters()]))
            ret["loss"] = loss.item()
            ret["gnorm"] = gnorm.item()
            return ret

        input = torch.rand(2, 16, 32).requires_grad_(True)
        model = Model().to(device)
        no_cpt = get_loss_and_gnorm(model, input.to(device))
        print(no_cpt)

        model = Model(use_pytorch_checkpoint=True).to(device)
        pyt_cpt = get_loss_and_gnorm(model, input.to(device))
        print(pyt_cpt)
        torch.testing.assert_allclose(no_cpt["loss"], pyt_cpt["loss"])
        torch.testing.assert_allclose(no_cpt["gnorm"], pyt_cpt["gnorm"])

        model = Model(use_fairscale_checkpoint=True).to(device)
        fairscale_cpt = get_loss_and_gnorm(model, input.to(device))
        print(fairscale_cpt)
        torch.testing.assert_allclose(no_cpt["loss"], fairscale_cpt["loss"])
        torch.testing.assert_allclose(no_cpt["gnorm"], fairscale_cpt["gnorm"])

        model = Model(use_fairscale_checkpoint=True, offload_to_cpu=True).to(device)
        fairscale_cpt_offload = get_loss_and_gnorm(model, input.to(device))
        print(fairscale_cpt_offload)
        torch.testing.assert_allclose(no_cpt["loss"], fairscale_cpt_offload["loss"])
        torch.testing.assert_allclose(no_cpt["gnorm"], fairscale_cpt_offload["gnorm"])
        assert 0

    def test_checkpoint_wrapper_cpu(self):
        self._test_checkpoint_wrapper(device=torch.device("cpu"))

    @unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
    def test_checkpoint_wrapper_cuda(self):
        self._test_checkpoint_wrapper(device=torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
