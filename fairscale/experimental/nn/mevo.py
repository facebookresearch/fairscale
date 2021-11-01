# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Tuple

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

DEBUG = False


def next_power_of_2_or_max(n: int, max_n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    if n > max_n:
        return max_n
    return n


def get_data(
    shape: Tuple[Tuple[int, int], Tuple[int, int]], dtype: torch.dtype = torch.float16, device: str = "cuda"
) -> Tuple[torch.Tensor, nn.Parameter, torch.Tensor]:
    """ Utility function for getting some tensors for testing and benchmarking."""
    (tokens, d1), (d2, vocabs) = shape
    assert d1 == d2
    input = torch.rand(tokens, d1, device=device, dtype=dtype).requires_grad_(True)
    weight = nn.Linear(d2, vocabs, bias=False, device=device, dtype=dtype).weight
    target = (torch.rand(tokens, device=device) * vocabs).long()
    return input, weight, target


class BaselineSoftmax(nn.Module):
    """ Baseline softmax that does an output projection and a softmax. """

    def __init__(
        self, proj_weight: torch.nn.Parameter, k: int = 0, tile_factor: int = 0, log: bool = True
    ):  # k, tile_factor are ignored.
        super().__init__()
        out_dim, in_dim = proj_weight.shape
        self.fc = nn.Linear(in_dim, out_dim, bias=False, device="cuda", dtype=proj_weight.dtype)
        self.fc.weight = proj_weight
        assert self.fc.weight.dtype in [torch.float16, torch.float32], self.fc.weight.dtype
        self.fp16 = self.fc.weight.dtype == torch.float16
        self.log = log

    def forward(self, *input: Any, **kwargs: Any) -> Any:
        assert kwargs == {}
        input, target = input
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        if self.fp16:
            assert input.dtype == torch.float16
        x = self.fc(input)
        if self.log:
            x = F.log_softmax(x, dim=-1, dtype=torch.float32)
        else:
            x = F.softmax(x, dim=-1, dtype=torch.float32)
        assert x.dtype == torch.float32
        return x


class BaselineSoftmaxNllLoss(BaselineSoftmax):
    """ Baseline that does an output projection, a softmax NLL loss. """

    def __init__(
        self, proj_weight: nn.Parameter, k: int = 0, tile_factor: int = 0, log: bool = True
    ):  # k, tile_factor are ignored.
        super().__init__(proj_weight, k, tile_factor, log)

    def forward(self, *input: Any, **kwargs: Any) -> Any:
        assert kwargs == {}
        input, target = input
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        if len(input.shape) == 3:
            input = input.reshape(-1, input.shape[2])
        if len(target.shape) == 2:
            target = target.reshape(-1)
        if self.fp16:
            assert input.dtype == torch.float16
        x = self.fc(input)
        if self.log:
            x = F.log_softmax(x, dim=-1, dtype=torch.float32)
        else:
            x = F.softmax(x, dim=-1, dtype=torch.float32)
        assert x.dtype == torch.float32
        x = F.nll_loss(x, target, reduction="sum")
        return x


class GetMaxFunction(torch.autograd.Function):
    @staticmethod
    def get_max(i: torch.Tensor, w: torch.Tensor, fp: bool) -> torch.Tensor:
        _m = torch.matmul(i, w.T)
        if fp:
            _m = _m.float()
        _m = _m.max(dim=1)[0]
        return _m

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        i: torch.Tensor,
        w: torch.Tensor,
        kernel_obj: "TorchFuseAllTiled",
        w_idx: int,
        w_split_size: int,
        split_dim: int,
    ) -> torch.Tensor:
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            print("DEBUG max fwd")
        ctx.save_for_backward(i, w)
        ctx.kernel_obj = kernel_obj
        ctx.w_idx = w_idx
        ctx.w_split_size = w_split_size
        ctx.args = {}
        assert split_dim == 0
        with torch.no_grad():
            return GetMaxFunction.get_max(i, w, kernel_obj.fp_max)

    @staticmethod
    def backward(ctx: Any, *args: Any) -> Any:
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            print("DEBUG max bwd")
        assert len(args) == 1
        # Gradients should already exist due to TargetScoreFunction's backward.
        assert ctx.kernel_obj.proj_weight.grad is not None

        # Get saved i and w.
        i, w = ctx.saved_tensors
        assert i.requires_grad
        assert w.requires_grad
        i = i.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)

        # Forward + backward again.
        with torch.enable_grad():
            maxs = GetMaxFunction.get_max(i, w, ctx.kernel_obj.fp_max)
        torch.autograd.backward(maxs, *args)
        # Accumulate the grads.
        assert w.grad is not None
        with torch.no_grad():
            grads = torch.split(ctx.kernel_obj.proj_weight.grad, ctx.w_split_size)
            grads[ctx.w_idx].add_(w.grad)
        return i.grad, None, None, None, None, None


class GetSumFunction(torch.autograd.Function):
    @staticmethod
    def get_sum(i: torch.Tensor, w: torch.Tensor, maxs: torch.Tensor, fp: bool) -> torch.Tensor:
        _s = torch.matmul(i, w.T)
        if fp:
            _s = _s.float()
        _s = (_s - maxs.reshape(-1, 1)).exp().sum(dim=1)
        return _s

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        i: torch.Tensor,
        w: torch.Tensor,
        maxs: torch.Tensor,
        kernel_obj: "TorchFuseAllTiled",
        w_idx: int,
        w_split_size: int,
        split_dim: int,
    ) -> torch.Tensor:
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            print("DEBUG sum fwd")
        ctx.save_for_backward(i, w, maxs)
        ctx.kernel_obj = kernel_obj
        ctx.w_idx = w_idx
        ctx.w_split_size = w_split_size
        assert split_dim == 0
        with torch.no_grad():
            return GetSumFunction.get_sum(i, w, maxs, kernel_obj.fp_sum)

    @staticmethod
    def backward(ctx: Any, *args: Any) -> Any:
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            print("DEBUG sum bwd")
        assert len(args) == 1
        # Gradients should already exist due to TargetScoreFunction's backward.
        assert ctx.kernel_obj.proj_weight.grad is not None
        i, w, maxs = ctx.saved_tensors
        assert i.requires_grad
        assert w.requires_grad
        assert maxs.requires_grad
        i = i.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)
        maxs = maxs.detach().requires_grad_(True)
        # Forward + backward again.
        with torch.enable_grad():
            sums = GetSumFunction.get_sum(i, w, maxs, ctx.kernel_obj.fp_sum)
        torch.autograd.backward(sums, *args)
        # Accumulate the grads.
        assert w.grad is not None
        with torch.no_grad():
            grads = torch.split(ctx.kernel_obj.proj_weight.grad, ctx.w_split_size)
            grads[ctx.w_idx].add_(w.grad)
        return i.grad, None, maxs.grad, None, None, None, None


class TargetScoreFunction(torch.autograd.Function):
    @staticmethod
    def get_target_score(i: torch.Tensor, w: torch.Tensor, target: torch.Tensor, fp: bool) -> torch.Tensor:
        tokens, d_model = i.shape
        assert d_model == w.shape[1]
        tw = w.gather(dim=0, index=target.reshape(target.shape[0], 1).expand(target.shape[0], d_model))
        assert tw.shape == (tokens, d_model)
        target_score = i * tw
        if fp:
            target_score = target_score.float()
        target_score = target_score.sum(dim=1)  # sum into target scores with shape (tokens,)
        return target_score

    @staticmethod
    def forward(  # type: ignore
        ctx: Any, i: torch.Tensor, w: torch.Tensor, target: torch.Tensor, kernel_obj: "TorchFuseAllTiled"
    ) -> torch.Tensor:
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            print("DEBUG target fwd")
        ctx.save_for_backward(i, w, target)
        ctx.kernel_obj = kernel_obj
        with torch.no_grad():
            x = TargetScoreFunction.get_target_score(i, w, target, kernel_obj.fp_target)
        return x

    @staticmethod
    def backward(ctx: Any, *args: Any) -> Any:
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            print("DEBUG target bwd")
        assert len(args) == 1
        i, w, target = ctx.saved_tensors
        assert i.requires_grad
        assert w.requires_grad
        assert not target.requires_grad
        i = i.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)
        with torch.enable_grad():
            scores = TargetScoreFunction.get_target_score(i, w, target, ctx.kernel_obj.fp_target)
        torch.autograd.backward(scores, *args)
        if ctx.kernel_obj.proj_weight.grad is not None:
            # This means we accumulate full grad between iters. Not memory efficient.
            ctx.kernel_obj.proj_weight.grad.add_(w.grad)
        else:
            ctx.kernel_obj.proj_weight.grad = w.grad
        return i.grad, None, None, None


class BackwardTriggerFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx: Any, w: torch.Tensor, trigger_tensor: torch.Tensor
    ) -> torch.Tensor:
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            print("DEBUG trigger fwd")
        ctx.save_for_backward(w, trigger_tensor)
        return w

    @staticmethod
    def backward(ctx: Any, *args: Any) -> Any:
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            print("DEBUG trigger bwd")
        assert len(args) == 1
        w, trigger = ctx.saved_tensors
        assert w.requires_grad
        assert trigger.requires_grad
        return None, torch.zeros_like(trigger)


class BackwardTrigger(nn.Module):
    def __init__(self, linked_param: torch.Tensor):
        super().__init__()
        assert isinstance(linked_param, nn.Parameter)
        self.trigger = nn.Parameter(torch.rand(1, dtype=linked_param.dtype))
        self.trigger._linked_param = linked_param

    def forward(self, *input: Any, **kwargs: Any) -> Any:
        return BackwardTriggerFn.apply(self.trigger._linked_param, self.trigger)


class TorchFuseAllTiled(nn.Module):
    """ Torch fuse fc + softmax + nll_loss in a tiled fashion.

        This uses less memory but is quite a bit slower.
    """

    def __init__(self, proj_weight: nn.Parameter, k: int = 0, tile_factor: int = 16, reduction: str = "sum"):
        # k is ignored.
        super().__init__()
        self.proj_weight = proj_weight
        self.tf_in, self.tf_w = tile_factor, tile_factor
        self.fp_max = True
        self.fp_sum = True  # This is esp. important when tensors are large. Otherwise, you get inf.
        self.fp_target = True
        self.log_softmax = True
        self.reduction = reduction
        assert self.reduction in ["sum", "mean"]
        self.trigger = BackwardTrigger(self.proj_weight)
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            print(
                f"DEBUG cfg tf_in={self.tf_in} tf_w={self.tf_w} fp_max={self.fp_max} "
                f"fp_sum={self.fp_sum} fp_target={self.fp_target} log_softmax={self.log_softmax} "
                f"reduction={self.reduction}"
            )

    def get_max(self, i: torch.Tensor, w: torch.Tensor, w_idx: int, w_split_size: int, split_dim: int) -> torch.Tensor:
        return GetMaxFunction.apply(i, w, self, w_idx, w_split_size, split_dim)

    def get_sum(
        self, i: torch.Tensor, w: torch.Tensor, maxs_at_idx: torch.Tensor, w_idx: int, w_split_size: int, split_dim: int
    ) -> torch.Tensor:
        return GetSumFunction.apply(i, w, maxs_at_idx, self, w_idx, w_split_size, split_dim)

    def get_target_nlprob(
        self, i: torch.Tensor, w: torch.Tensor, target: torch.Tensor, debase_max: torch.Tensor, exp_sums: torch.Tensor
    ) -> torch.Tensor:
        target_score = TargetScoreFunction.apply(i, w, target, self)
        prob = (target_score - debase_max).exp() / exp_sums
        if self.log_softmax:
            # lprob
            prob = prob.log()
        # nlprob, then sum over all tokens.
        return -prob.sum()

    def forward(self, *input: Any, **kwargs: Any) -> Any:
        cur_mem = round(torch.cuda.memory_allocated() / 1024 / 1024)
        mem = round(torch.cuda.max_memory_allocated() / 1024 / 1024)
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            print("DEBUG cur, peak", cur_mem, mem)
        assert kwargs == {}
        input, target = input
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert input.requires_grad
        if len(input.shape) == 3:
            input = input.reshape(-1, input.shape[2])
        if len(target.shape) == 2:
            target = target.reshape(-1)

        tokens, d_model = input.shape
        vocab, d2 = self.proj_weight.shape
        assert d_model == d2
        split_dim = 0
        input_split_size = next_power_of_2_or_max(tokens // self.tf_in, tokens)
        weight_split_size = next_power_of_2_or_max(vocab // self.tf_w, vocab)
        inputs = torch.split(input, input_split_size, split_dim)
        weight = self.trigger()
        weights = torch.split(weight, weight_split_size, split_dim)

        # Get maxs
        maxs = []
        for i in inputs:
            m = None  # max with (tokens_tile,) shape
            for w_idx, w in enumerate(weights):
                _m = self.get_max(i, w, w_idx, weight_split_size, split_dim)
                if m is None:
                    m = _m
                else:
                    m = torch.max(m, _m)
            assert m is not None
            maxs.append(m)  # (tokens_tile,)
        maxs_tensor = torch.cat(maxs)  # (tokens,)
        assert maxs_tensor.shape == (tokens,)

        # Get sums.
        sums = []
        for idx, i in enumerate(inputs):
            s = None  # sum with (tokens_tile,) shape
            for w_i, w in enumerate(weights):
                _s = self.get_sum(i, w, maxs[idx], w_i, weight_split_size, split_dim)
                if s is None:
                    s = _s
                else:
                    s += _s
            assert s is not None
            sums.append(s)  # (tokens_tile,)
        sums = torch.cat(sums)  # (tokens,)
        assert sums.shape == (tokens,)

        # select weights for targets
        result = self.get_target_nlprob(input, self.proj_weight, target, maxs_tensor, sums)
        if self.reduction == "mean":
            result /= tokens
        return result
