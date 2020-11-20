# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of Petuum, Inc.  nor the names of its contributors may be
#    used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import functools
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.autograd import Variable
import torch.distributed as dist


class AdaScale(object):
    """
    Implements the AdaScale_ algorithm for scaling the learning rate for
    distributed and large batch size training. Can be used in combination with
    ``torch.nn.parallel.DistributedDataParallel`` and ``torch.optim.SGD``.

    .. _AdaScale: https://proceedings.icml.cc/static/paper_files/icml/2020/4682-Supplemental.pdf

    .. code-block:: python

        optim = torch.optim.SGD(model.parameters(), lr=0.001)
        model = DistributedDataParallel(model)
        adascale = AdaScale(optim)

        for epoch in ...:
            for batch in ...:
                optim.zero_grad()
                loss = ...
                loss.backward()
                adascale.step()

    Args:
        optimizer (torch.optim.Optimizer):
            Optimizer to apply AdaScale to.
        world_size (int):
            Number of world_size for distributed training. If
            None, defaults to ``dist.get_world_size()``.
        scale (float):
            Scaling factor of the batch size, e.g. using a 10x
            larger batch size (summed across all world_size) means a scale of
            10. If None, defaults to ``world_size``.
        smoothing (float):
            Smoothing factor between batches. Default value: 0.9999
        num_gradients_to_accumulate (int):
            Number of passes that we accumulate gradients locally.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        world_size: Optional[int] = None,
        scale: Optional[float] = None,
        smoothing: float = 0.999,
        num_gradients_to_accumulate: int = 1,
    ):
        self._optimizer = optimizer
        self._local_grad_sqr: Optional[torch.Tensor] = None
        self._world_size: int = (
            world_size if world_size is not None else dist.get_world_size() if dist.is_initialized() else 1
        )
        self._smoothing = smoothing
        self._num_backward_calls = 0
        self._num_grads_to_accum = num_gradients_to_accumulate

        if self._world_size * self._num_grads_to_accum <= 1:
            # gain will be NaN since we will be dividing by zero in paper's B.3 where (S-1) == 0.
            raise RuntimeError("AdaScale does not support a single worker without grad accumulation.")

        # Per-param-group sqr & var states (sigma^2 & mu^2 in the paper).
        self._optimizer.state.setdefault(
            "adascale",
            {
                "grad_sqr_avg": np.ones(len(optimizer.param_groups)),
                "grad_var_avg": np.zeros(len(optimizer.param_groups)),
            },
        )

        self.set_scale(self._world_size * self._num_grads_to_accum if scale is None else scale)

        # Register the gradient hooks. Note, don't assume every param will generate
        # a gradient (i.e. triggering the hook) in every backward pass.
        for idx, param_group in enumerate(self._optimizer.param_groups):
            for param in param_group["params"]:
                param.register_hook(functools.partial(self._backward_hook, idx))

    @property
    def state(self) -> Dict[str, np.ndarray]:
        """
        Return the states of AdaScale.
        """
        return self._optimizer.state["adascale"]

    @property
    def scale(self) -> float:
        """
        The scaling factor of the current batch size, relative to the baseline
        batch size when training with a single worker. For example, if the
        baseline batch size is 32, but using a scaled-up batch size of 80, then
        then the scaling factor is 2.5.

        Returns:
            (float):
                The current scaling factor.
        """
        return self._scale

    def set_scale(self, scale: float) -> None:
        """
        Set the scaling factor of the current batch size. It is up to the
        application to invoke this function to make sure that AdaScale's
        scaling factor matches the actual batch size used during training.

        Args:
            scale (float):
                New scaling factor to be applied to AdaScale.
        """
        self._scale = scale

    def grad_sqr_avg(self, pg_idx: Optional[int] = None) -> float:
        """
        Current estimate of the squared l2-norm of the true gradient
        (sigma squared in the AdaScale paper).

        Args:
            pg_idx (Optional[int]):
                Optional index for a parameter group.

        Returns:
            (float):
                Estimate of squared l2-norm.
        """
        if pg_idx is not None:
            return self.state["grad_sqr_avg"][pg_idx]
        else:
            return np.sum(self.state["grad_sqr_avg"])

    def grad_var_avg(self, pg_idx: Optional[int] = None) -> float:
        """
        Current estimate of the trace of the covariance of the true gradient
        (mu squared in the AdaScale paper).

        Args:
            pg_idx (Optional[int]):
                Optional index for a parameter group.

        Returns:
            (float):
                Estimate of trace of the covariance.
        """
        if pg_idx is not None:
            return self.state["grad_var_avg"][pg_idx]
        else:
            return np.sum(self.state["grad_var_avg"])

    def gain(self, scale: Optional[float] = None, pg_idx: Optional[int] = None) -> float:
        """
        Current estimate of the AdaScale gain ratio (r_t in the paper).

        Args:
            scale (float):
                Optional batch size scale to estimate the gain ratio for.
            pg_idx (int):
                Optional index of a parameter group.

        Returns:
            (float):
                Estimate of gain ratio.
        """
        scale = self._scale if scale is None else scale
        var = self.grad_var_avg(pg_idx)
        sqr = self.grad_sqr_avg(pg_idx)
        return (var + sqr) / (var / scale + sqr)

    def _update_avg(self, name: str, value: torch.Tensor, factor: float) -> None:
        # This function computes and stores the moving average of a vector
        # using a smoothing factor.
        biased = self.state.get(name + "_biased", 0.0)
        unbias = self.state.get(name + "_unbias", 0.0)
        biased = factor * biased + (1.0 - factor) * value
        unbias = factor * unbias + (1.0 - factor)
        self.state[name + "_biased"] = biased
        self.state[name + "_unbias"] = unbias
        self.state[name] = biased / unbias

    def _backward_hook(self, pg_idx: int, grad: torch.Tensor) -> None:
        # This method should be invoked once for each parameter during the
        # backward pass, before gradients are synchronized between world_size.

        # Store the local gradient square sums in a vector.
        if self._local_grad_sqr is None:
            self._local_grad_sqr = torch.zeros(len(self._optimizer.param_groups), device=grad.device)
        self._local_grad_sqr[pg_idx] += grad.pow(2).sum()

        # Now, ensure we queue a callback at the end of the callback queue.
        # This will fire after all gradient callbacks are done (esp. those
        # queued by DDP.
        self._final_callback_queued = False
        Variable._execution_engine.queue_callback(self._queue_callback)

    def _queue_callback(self) -> None:
        # This method should be invoked after the entire backward pass. We want
        # to make sure self._final_callback is invoked once, only after all
        # gradients have been synchronized between each worker. However, the
        # synchronization code in DistributedDataParallel is also done in a
        # callback, which might not yet be executed. Therefore, we enqueue
        # self._final_callback from this method, which should ensure it is
        # invoked after the gradient synchronization callback.
        if self._final_callback_queued:
            return
        self._final_callback_queued = True
        Variable._execution_engine.queue_callback(self._final_callback)

    def _final_callback(self) -> None:
        # This method should be invoked once for each backward pass, after
        # gradients have been synchronized between each worker, unless we
        # are in gradient accumulation mode, where grads are not all_reduced
        # between the GPUs.
        self._final_callback_queued = False
        assert isinstance(self._local_grad_sqr, torch.Tensor)

        # Keep track of number of backward calls for gradient accumulation.
        self._num_backward_calls += 1

        # TODO (min): We need to have a way to check that training loop & DDP
        #             is doing the right thing where the gradient is reduced
        #             in this backward pass.
        #             Longer term, we may compute the gain and then inform
        #             the training loop when it is a good time to step().
        if self._num_backward_calls % self._num_grads_to_accum != 0:
            return

        # Since self._local_grad_sqr is FP32, sum shouldn't overflow.
        # This vector has length of # of param_groups, so it is small, but we
        # use async to hide the all_reduce latency, esp when # of nodes is large.
        work = None
        if self._world_size > 1:
            work = dist.all_reduce(self._local_grad_sqr, async_op=True)  # SUM

        # Compute the sums of squares for reduced gradients.
        # Divide by _num_grads_to_accum since the gradients are accumulated.
        #
        # Note: we are mutating the gradients here!!!
        total_grad_sqr = np.array(
            [
                sum(param.grad.div_(self._num_grads_to_accum).pow(2).sum().item() for param in group["params"])
                for group in self._optimizer.param_groups
            ]
        )

        # Wait for all_reduce to be done and move it to cpu & np.
        if work:
            work.wait()
        local_grad_sqr = self._local_grad_sqr.cpu().numpy()
        self._local_grad_sqr = None

        # See appendix B.3 of the paper.
        #
        # local_grad_sqr is \sigma_{i=1}^{S}\norm{g_t_i}
        # total_grad_sqr is \norm{g_t}
        S = self._scale
        grad_var = local_grad_sqr / (S - 1) - total_grad_sqr * S / (S - 1)
        grad_sqr = total_grad_sqr - grad_var / S
        grad_var = np.maximum(grad_var, 1e-6)
        grad_sqr = np.maximum(grad_sqr, 0.0)
        theta = self._smoothing ** S
        self._update_avg("grad_sqr_avg", grad_sqr, theta)
        self._update_avg("grad_var_avg", grad_var, theta)

    def step(self, *args: Any, **kwargs: Any) -> Optional[float]:
        """
        Run one optimizer step using Adascale. Essentially just invokes
        ``optimizer.step(*args, **kwargs)`` with a scaled learning rate.

        Args:
            args (Any):
                Positional arguments passed to ``optimizer.step``.
            kwargs (Any):
                Keyword arguments passed to ``optimizer.step``.

        Returns:
            (Tensor):
                The loss tensor if a closure if used to re-evaluate the model.
        """
        # Set original LR and set new LR.
        original_lr = []
        for idx, param_group in enumerate(self._optimizer.param_groups):
            original_lr.append(param_group["lr"])
            param_group["lr"] = self.gain(pg_idx=idx) * param_group["lr"]

        # Step it.
        res = self._optimizer.step(*args, **kwargs)

        # Restore the original LR.
        for lr, param_group in zip(original_lr, self._optimizer.param_groups):
            param_group["lr"] = lr

        return res

    def zero_grad(self) -> None:
        """Proxy function to optimizer"""
        self._optimizer.zero_grad()
