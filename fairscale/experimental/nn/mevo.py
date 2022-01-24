# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Optional, Tuple

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

# Debugging flag to enable some prints. Useful to debug with FSDP.
DEBUG = False


def _next_power_of_2_or_max(n: int, max_n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n, with a limit.

    Useful when used in splitting a tensor into chunks with power-of-2 sizes.
    """
    # special case, just split to 1 element chunks.
    if n == 0:
        return 1
    orig_n = n
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    assert n >= orig_n, f"{n} vs. {orig_n}"
    assert bin(n).count("1") == 1, bin(n)  # Catch the case n is too large for this function.
    if n > max_n:
        return max_n
    return n


def _reshape_inputs(input: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert 3D inputs to 2D for this kernel"""
    if len(input.shape) == 3:
        input = input.reshape(-1, input.shape[2])
    if len(target.shape) == 2:
        target = target.reshape(-1)
    return input, target


def get_data(
    shape: Tuple[Tuple[int, int], Tuple[int, int]], dtype: torch.dtype = torch.float16, device: str = "cuda"
) -> Tuple[torch.Tensor, nn.Parameter, torch.Tensor]:
    """Utility function for getting some tensors for testing and benchmarking."""
    (tokens, d1), (d2, vocabs) = shape
    assert d1 == d2
    input = torch.rand(tokens, d1, device=device, dtype=dtype).requires_grad_(True)
    # Before pytorch 1.9, nn.Linear does not support device and dtype init option. So we use to()
    # and an if condition.
    layer = nn.Linear(d2, vocabs, bias=False).to(device)
    assert dtype in [torch.float16, torch.float32]
    if dtype == torch.float16:
        layer = layer.half()
    weight = layer.weight
    target = (torch.rand(tokens, device=device) * vocabs).long()
    return input, weight, target


class BaselineSoftmax(nn.Module):
    """Baseline softmax that does an output linear projection and a softmax.


        We also support LMCL (Large Margin Cosine Loss) from the CosFace paper. See
        more detailed comment in the MEVO class below.

        This is intended to be used with an embedding layer with shared weights.

    Args:
        proj_weight (nn.Parameter):
            The shared weight.
        tile_factor (int):
            Unused. It is here to make kernel init easier with MEVO.
        log_softmax (bool):
            If True, use log_softmax instead of softmax.
        margin (float):
            Used in LMCL (when scale != None). See MEVO comments for
            more details.
        scale (Optional[float]):
            Used in LMCL. If scale is None, LMCL is turned off. See
            MEVO comments for more details.

    """

    def __init__(
        self,
        proj_weight: nn.Parameter,
        tile_factor: int = 0,
        log_softmax: bool = True,
        margin: float = 0.35,
        scale: Optional[float] = None,
    ):
        super().__init__()
        out_dim, in_dim = proj_weight.shape
        assert "cuda" in str(proj_weight.device), "weight should be on GPU"
        self.fc = nn.Linear(in_dim, out_dim, bias=False).to("cuda")
        assert proj_weight.dtype in [torch.float16, torch.float32]
        if proj_weight.dtype == torch.float16:
            self.fc = self.fc.half()
        self.fc.weight = proj_weight
        assert self.fc.weight.dtype in [torch.float16, torch.float32], self.fc.weight.dtype
        self.fp16 = self.fc.weight.dtype == torch.float16
        self.log_softmax = log_softmax
        self.margin = margin
        self.scale = scale

    def lmcl_pre_softmax(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # normalize feature and fc layer before multiplication
        #   n: number of features (tokens)
        #   k: number of classes (vocab size)
        #   c: hidden dimension (d_model)
        x = F.normalize(input, dim=1)
        w = F.normalize(self.fc.weight, dim=1)
        logits = torch.einsum("nc,kc->nk", x, w)

        # add margin
        row_ind = torch.arange(x.shape[0], dtype=torch.long).to(x.device)
        col_ind = target
        logits[row_ind, col_ind] -= self.margin

        # add scale
        logits *= self.scale

        return logits

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward function that computes softmax output with the input and target."""
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        input, target = _reshape_inputs(input, target)
        if self.fp16:
            assert input.dtype == torch.float16
        if self.scale is not None:
            x = self.lmcl_pre_softmax(input, target)
        else:
            x = self.fc(input)
        # Note that we do softmax in FP32, which is important for numerical stability.
        if self.log_softmax:
            x = F.log_softmax(x, dim=-1, dtype=torch.float32)
        else:
            x = F.softmax(x, dim=-1, dtype=torch.float32)
        assert x.dtype == torch.float32
        return x


class BaselineSoftmaxNllLoss(BaselineSoftmax):
    """Baseline that does an output projection, a softmax & a NLL loss (cross-entropy).

    See BaselineSoftmax above. Constructor is the same. Only difference is in the
    forward function.

    This class is used for testing and benchmarking.
    """

    def __init__(
        self,
        proj_weight: nn.Parameter,
        tile_factor: int = 0,
        log_softmax: bool = True,
        margin: float = 0.35,
        scale: Optional[float] = None,
    ):
        super().__init__(proj_weight, tile_factor, log_softmax, margin, scale)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward that directly compute the loss."""
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        input, target = _reshape_inputs(input, target)
        x = super().forward(input, target)
        return F.nll_loss(x, target, reduction="sum")


def lmcl_matmul(
    i: torch.Tensor, w: torch.Tensor, tgt: torch.Tensor, w_idx: int, margin: float, scale: Optional[float]
) -> torch.Tensor:
    """LMCL variation of matmul with normalization, margin and scale."""
    # normalize and matmul
    logits = torch.matmul(F.normalize(i, dim=1), F.normalize(w, dim=1).T)

    # add margin using a mask since tgt might be out of the the weight split's range.
    mask = torch.arange(w_idx * w.shape[0], (w_idx + 1) * w.shape[0], dtype=torch.long, device=i.device).expand(
        i.shape[0], -1
    )
    logits[mask == tgt.reshape(-1, 1)] -= margin

    # add scale
    logits *= scale

    return logits


class GetMaxFunction(torch.autograd.Function):
    """Custom checkpointed function to get max-per-token from an input and a weight"""

    @staticmethod
    def get_max(
        i: torch.Tensor,
        w: torch.Tensor,
        tgt: torch.Tensor,
        w_idx: int,
        full_precision: bool,
        margin: float,
        scale: Optional[float],
    ) -> torch.Tensor:
        """
        Throughout this code:

          i: input data with shape = (split-of-tokens, d_model)
          w: weight data with shape = (split-of-vocabs, d_model)
          tgt: target prediction data with shape = (split-of-tokens,)
        """
        if scale is not None:
            _m = lmcl_matmul(i, w, tgt, w_idx, margin, scale)
        else:
            _m = torch.matmul(i, w.T)
        if full_precision:
            _m = _m.float()
        _m = _m.max(dim=1)[0]
        return _m

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        i: torch.Tensor,
        w: torch.Tensor,
        tgt: torch.Tensor,
        kernel_obj: "MemoryEfficientVocabOutput",
        w_idx: int,
        w_split_size: int,
        split_dim: int,
    ) -> torch.Tensor:
        """Forward function that computes the max, without saving activations."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            print("DEBUG max fwd")
        ctx.save_for_backward(i, w, tgt)
        ctx.kernel_obj = kernel_obj
        ctx.w_idx = w_idx
        ctx.w_split_size = w_split_size
        ctx.args = {}
        assert split_dim == 0
        # During forward, we use ``no_grad'' to avoid saving the activations.
        # The activations will be recomputed in backward below and freed
        # immediately after use. This saves the overall GPU peak memory of this layer.
        with torch.no_grad():
            return GetMaxFunction.get_max(i, w, tgt, w_idx, kernel_obj.fp_max, kernel_obj.margin, kernel_obj.scale)

    @staticmethod
    def backward(ctx: Any, *args: Any) -> Any:
        """Recompute the forward max and backward grad.

        Accumulate the grad to the right split of the full grad.
        """
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            print("DEBUG max bwd")
        assert len(args) == 1
        # Gradients should already exist due to TargetScoreFunction's backward.
        assert ctx.kernel_obj.proj_weight.grad is not None

        # Get saved i and w.
        i, w, tgt = ctx.saved_tensors
        assert i.requires_grad
        assert w.requires_grad
        # We use ``detach()'' to ensure the backward call below does not
        # trigger backward computation that produced i and w here. Otherwise,
        # the backward call below would trigger backward all the way to
        # the batch input.
        i = i.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)

        # Forward + backward again.
        with torch.enable_grad():
            # This saves the activations.
            maxs = GetMaxFunction.get_max(
                i, w, tgt, ctx.w_idx, ctx.kernel_obj.fp_max, ctx.kernel_obj.margin, ctx.kernel_obj.scale
            )
        # This will use the activations and free them immediately.
        torch.autograd.backward(maxs, *args)

        # Accumulate the computed gradients into the bigger weight tensor's gradient tensor.
        assert w.grad is not None
        with torch.no_grad():
            grads = torch.split(ctx.kernel_obj.proj_weight.grad, ctx.w_split_size)
            grads[ctx.w_idx].add_(w.grad)
        return i.grad, None, None, None, None, None, None


class GetSumFunction(torch.autograd.Function):
    """Custom checkpointed function to get sum-per-token from an input and a weight."""

    @staticmethod
    def get_sum(
        i: torch.Tensor,
        w: torch.Tensor,
        tgt: torch.Tensor,
        maxs: torch.Tensor,
        w_idx: int,
        full_precision: bool,
        margin: float,
        scale: Optional[float],
    ) -> torch.Tensor:
        if scale is not None:
            _s = lmcl_matmul(i, w, tgt, w_idx, margin, scale)
        else:
            _s = torch.matmul(i, w.T)
        if full_precision:
            _s = _s.float()
        _s = (_s - maxs.reshape(-1, 1)).exp().sum(dim=1)
        return _s

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        i: torch.Tensor,
        w: torch.Tensor,
        tgt: torch.Tensor,
        maxs: torch.Tensor,
        kernel_obj: "MemoryEfficientVocabOutput",
        w_idx: int,
        w_split_size: int,
        split_dim: int,
    ) -> torch.Tensor:
        """Forward function that computes the sum, without saving activations."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            print("DEBUG sum fwd")
        ctx.save_for_backward(i, w, tgt, maxs)
        ctx.kernel_obj = kernel_obj
        ctx.w_idx = w_idx
        ctx.w_split_size = w_split_size
        assert split_dim == 0
        with torch.no_grad():
            return GetSumFunction.get_sum(
                i, w, tgt, maxs, w_idx, kernel_obj.fp_sum, kernel_obj.margin, kernel_obj.scale
            )

    @staticmethod
    def backward(ctx: Any, *args: Any) -> Any:
        """Recompute the forward sum and backward grad.

        Accumulate the grad to the right split of the full grad.
        """
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            print("DEBUG sum bwd")
        assert len(args) == 1
        # Gradients should already exist due to TargetScoreFunction's backward.
        assert ctx.kernel_obj.proj_weight.grad is not None

        # Get saved i, w, and maxs.
        i, w, tgt, maxs = ctx.saved_tensors
        assert i.requires_grad
        assert w.requires_grad
        assert maxs.requires_grad
        i = i.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)
        maxs = maxs.detach().requires_grad_(True)

        # Forward + backward again.
        with torch.enable_grad():
            sums = GetSumFunction.get_sum(
                i, w, tgt, maxs, ctx.w_idx, ctx.kernel_obj.fp_sum, ctx.kernel_obj.margin, ctx.kernel_obj.scale
            )
        torch.autograd.backward(sums, *args)

        # Accumulate the grads.
        assert w.grad is not None
        with torch.no_grad():
            grads = torch.split(ctx.kernel_obj.proj_weight.grad, ctx.w_split_size)
            grads[ctx.w_idx].add_(w.grad)
        return i.grad, None, None, maxs.grad, None, None, None, None


class TargetScoreFunction(torch.autograd.Function):
    """Custom checkpointed function to compute the target score."""

    @staticmethod
    def get_target_score(
        i: torch.Tensor,
        w: torch.Tensor,
        target: torch.Tensor,
        full_precision: bool,
        margin: float,
        scale: Optional[float],
    ) -> torch.Tensor:
        tokens, d_model = i.shape
        assert d_model == w.shape[1]
        tw = w.gather(dim=0, index=target.reshape(target.shape[0], 1).expand(target.shape[0], d_model))
        assert tw.shape == (tokens, d_model)
        if scale is not None:
            target_score = F.normalize(i, dim=1) * F.normalize(tw, dim=1)
        else:
            target_score = i * tw
        if full_precision:
            target_score = target_score.float()
        target_score = target_score.sum(dim=1)  # sum into target scores with shape (tokens,)
        if scale is not None:
            target_score -= margin
            target_score *= scale
        return target_score

    @staticmethod
    def forward(  # type: ignore
        ctx: Any, i: torch.Tensor, w: torch.Tensor, target: torch.Tensor, kernel_obj: "MemoryEfficientVocabOutput"
    ) -> torch.Tensor:
        """Forward, without activations."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            print("DEBUG target fwd")
        ctx.save_for_backward(i, w, target)
        ctx.kernel_obj = kernel_obj
        with torch.no_grad():
            x = TargetScoreFunction.get_target_score(
                i, w, target, kernel_obj.fp_target, kernel_obj.margin, kernel_obj.scale
            )
        return x

    @staticmethod
    def backward(ctx: Any, *args: Any) -> Any:
        """Forward and backward again, assign or accumulate the gradients."""
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
            scores = TargetScoreFunction.get_target_score(
                i, w, target, ctx.kernel_obj.fp_target, ctx.kernel_obj.margin, ctx.kernel_obj.scale
            )
        torch.autograd.backward(scores, *args)
        if ctx.kernel_obj.proj_weight.grad is not None:
            # This means we accumulate full grad between iters. Not memory efficient.
            ctx.kernel_obj.proj_weight.grad.add_(w.grad)
        else:
            ctx.kernel_obj.proj_weight.grad = w.grad
        return i.grad, None, None, None


class BackwardTriggerFn(torch.autograd.Function):
    """A backward trigger function."""

    @staticmethod
    def forward(ctx: Any, w: torch.Tensor, trigger_tensor: torch.Tensor) -> torch.Tensor:  # type: ignore
        """We take a weight tensor and the trigger as inputs and output the weight directly."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            print("DEBUG trigger fwd")
        ctx.save_for_backward(w, trigger_tensor)
        return w

    @staticmethod
    def backward(ctx: Any, *args: Any) -> Any:
        """We return zero grad for the trigger only."""
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            print("DEBUG trigger bwd")
        assert len(args) == 1
        w, trigger = ctx.saved_tensors
        assert w.requires_grad
        assert trigger.requires_grad
        return None, torch.zeros_like(trigger)


class BackwardTrigger(nn.Module):
    """A backward trigger module.

    This module takes a parameter as an input and create a linked parameter
    from a newly created trigger parameter.

    The way to use it in a module's ``__init__'' and ``forward'' functions:

    ```
    def __init__():
      ...
      self.trigger = BackwardTrigger(some_layer.weight)
      ...

    def forward():
      w = self.trigger()
      ... continue to use w ...
    ```

    As a resule, the trigger's backward hook will be called at the end of
    the backward for the module that uses this trigger.
    """

    def __init__(self, linked_param: torch.Tensor):
        super().__init__()
        assert isinstance(linked_param, nn.Parameter)
        self.trigger = nn.Parameter(torch.rand(1, dtype=linked_param.dtype, device=linked_param.device))
        self.trigger._linked_param = linked_param

    def forward(self) -> torch.Tensor:  # type: ignore
        return BackwardTriggerFn.apply(self.trigger._linked_param, self.trigger)


class MemoryEfficientVocabOutput(nn.Module):  # AKA. MEVO
    """Fused fc + softmax + nll_loss in a tiled fashion.

        MEVO uses much less memory but is quite a bit slower.

        MEVO also implements the LMCL (Large Margin Cosine Loss) function introduced by
        highly cited
        `CosFace: Large Margin Cosine Loss for Deep Face Recognition [Wang et al.]`_.

        .. _`CosFace: Large Margin Cosine Loss for Deep Face Recognition [Wang et al.]`: https://arxiv.org/abs/1801.09414

        LMCL can be turned on using the ``margin`` and ``scale`` parameters below. These
        hyperparameters most likely require tuning, depending on the number of classes etc.

        MEVO LMCL can be suitable for face recognition and image retrieval tasks, esp. when
        the number prediction target classes is large. MEVO is slower but can use much
        less GPU memory in that case, which enables training with larger batches. We
        hope this is helpful but we strongly recommend users (AI researchers
        and engineers) to carefully consider their applications of this technology. This
        types of technology should not be used by small group of people exclusively to
        potentially harm the general public.

    Args:
        proj_weight (nn.Parameter):
            Sharing this weight with an embedding layer.
        tile_factor (int):
            Number of splits to use on the input sequence and vocab dimensions.
            Default: 16
        reduction (str):
            Reduction OP (sum or mean).
            Default: sum
        margin (float):
            Hyperparameter of the separation margin between classes. See the
            appendix of the CosFace paper for a formula on how to compute its
            value properly. The default value is unlikely to be suitable in all
            cases.
            Default: 0.35
        scale (Optional[float]):
            Hyperparameter of the feature-vector-scaling for LMCL. When not
            supplied, LMCL is turned off. See the appendix of the CosFace paper for
            a formula on how to compute its value properly.
            Default: None
    """

    def __init__(
        self,
        proj_weight: nn.Parameter,
        tile_factor: int = 16,
        reduction: str = "sum",
        margin: float = 0.35,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.proj_weight = proj_weight
        # TODO (Min): these two factors doesn't have to be the same. More tuning can be done.
        self.tf_in, self.tf_w = tile_factor, tile_factor
        self.fp_max = True
        self.fp_sum = True  # This is esp. important when tensors are large. Otherwise, you get inf.
        self.fp_target = True
        self.log_softmax = True
        self.reduction = reduction
        assert self.reduction in ["sum", "mean"]
        self.margin = margin
        self.scale = scale
        self.trigger = BackwardTrigger(self.proj_weight)
        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            print(
                f"DEBUG cfg tf_in={self.tf_in} tf_w={self.tf_w} fp_max={self.fp_max} "
                f"fp_sum={self.fp_sum} fp_target={self.fp_target} log_softmax={self.log_softmax} "
                f"reduction={self.reduction} margin={self.margin} scale={self.scale}"
            )

    def get_target_nlprob(
        self, i: torch.Tensor, w: torch.Tensor, target: torch.Tensor, debase_max: torch.Tensor, exp_sums: torch.Tensor
    ) -> torch.Tensor:
        """Get target's negative log probability."""
        target_score = TargetScoreFunction.apply(i, w, target, self)
        prob = (target_score - debase_max).exp() / exp_sums
        if self.log_softmax:
            # lprob
            prob = prob.log()
        # nlprob, then sum over all tokens.
        return -prob.sum()

    def eval_forward(self, input: torch.Tensor) -> torch.Tensor:
        """Eval time forward that doesn't fuse the softmax and NLL Loss kernels."""
        # Margin, scaling and normalization of LMCL does not apply to eval time as far as
        # I can tell. Therefore, we just do a matmul like the standard output layer.
        return torch.matmul(input, self.proj_weight.T)

    def forward(self, input: torch.Tensor, target: Optional[torch.Tensor]) -> torch.Tensor:  # type: ignore
        if not self.training and target is None:
            return self.eval_forward(input)

        if DEBUG and dist.is_initialized() and dist.get_rank() == 0:
            cur_mem = round(torch.cuda.memory_allocated() / 1024 / 1024)
            mem = round(torch.cuda.max_memory_allocated() / 1024 / 1024)
            print("DEBUG cur, peak", cur_mem, mem)
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        if torch.is_grad_enabled():
            assert input.requires_grad
        input, target = _reshape_inputs(input, target)

        tokens, d_model = input.shape
        (t2,) = target.shape
        vocab, d2 = self.proj_weight.shape
        assert d_model == d2, f"incorrect shape {d_model} vs {d2}"
        assert tokens == t2, f"incorrect shape {tokens} vs {t2}"
        split_dim = 0
        input_split_size = _next_power_of_2_or_max(tokens // self.tf_in, tokens)
        weight_split_size = _next_power_of_2_or_max(vocab // self.tf_w, vocab)
        inputs = torch.split(input, input_split_size, split_dim)
        weight = self.trigger()
        weights = torch.split(weight, weight_split_size, split_dim)

        targets = tuple([torch.Tensor()] * len(inputs))
        if self.scale is not None:
            targets = torch.split(target, input_split_size, split_dim)

        # Get maxs
        maxs = []
        for i, tgt in zip(inputs, targets):
            m = None  # max with (tokens_tile,) shape
            for w_idx, w in enumerate(weights):
                _m = GetMaxFunction.apply(i, w, tgt, self, w_idx, weight_split_size, split_dim)
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
        for i, tgt, debase_max in zip(inputs, targets, maxs):
            s = None  # sum with (tokens_tile,) shape
            for w_idx, w in enumerate(weights):
                _s = GetSumFunction.apply(i, w, tgt, debase_max, self, w_idx, weight_split_size, split_dim)
                if s is None:
                    s = _s
                else:
                    s += _s
            assert s is not None
            sums.append(s)  # (tokens_tile,)
        sums_tensor = torch.cat(sums)  # (tokens,)
        assert sums_tensor.shape == (tokens,)

        # select weights for targets
        result = self.get_target_nlprob(input, self.proj_weight, target, maxs_tensor, sums_tensor)
        if self.reduction == "mean":
            result /= tokens
        return result
