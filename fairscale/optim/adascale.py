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
from torch.autograd import Variable
import torch.distributed


class AdaScale(object):
    """
    Implements the AdaScale_ algorithm for scaling the learning rate for
    distributed and large batch size training. Can be used in combination with
    ``torch.nn.parallel.DistributedDataParallel`` and ``torch.optim.SGD``.

    .. _AdaScale: https://proceedings.icml.cc/static/paper_files/icml/2020/4682-Supplemental.pdf

    .. code-block:: python

        optim = torch.optim.SGD(model, lr=0.001)
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
            None, defaults to ``torch.distributed.get_world_size()``.
        scale (float):
            Scaling factor of the batch size, e.g. using a 10x
            larger batch size (summed across all world_size) means a scale of
            10. If None, defaults to ``world_size``.
        patch_optimizer (bool):
            If True, monkey-patches the ``step`` method of
            the optimizer with the AdaScale's ``step`` method.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        world_size: Optional[int] = None,
        scale: Optional[float] = None,
        smoothing: float = 0.999,
        patch_optimizer: bool = False,
    ):
        self._optimizer = optimizer
        self._optimizer_step = optimizer.step
        self._local_grad_sqr: Optional[torch.Tensor] = None
        self._world_size: int = (world_size if world_size is not None else torch.distributed.get_world_size())

        if self._world_size <= 1:
            raise RuntimeError("AdaScale does not support a single worker.")

        self._optimizer.state.setdefault(
            "adascale",
            {
                "grad_sqr_avg": np.ones(len(optimizer.param_groups)),
                "grad_var_avg": np.zeros(len(optimizer.param_groups)),
            },
        )

        self.set_scale(self._world_size if scale is None else scale)

        for idx, param_group in enumerate(self._optimizer.param_groups):
            for param in param_group["params"]:
                param.register_hook(functools.partial(self._backward_hook, idx))

        if patch_optimizer:
            self.patch_optimizer()

        self._smoothing = smoothing

    @property
    def state(self) -> Dict[str, np.ndarray]:
        return self._optimizer.state["adascale"]

    @property
    def scale(self) -> float:
        """
        The scaling factor of the current batch size, relative to the baseline
        batch size when training with a single worker. For example, if the
        baseline batch size is 32, but using a scaled-up batch size of 80, then
        then the scaling factor is 2.5.
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

    def grad_sqr_avg(self) -> float:
        """
        Current estimate of the squared l2-norm of the true gradient (sigma
        squared in the AdaScale paper).

        Returns
            (float):
                Estimate of squared l2-norm.
        """
        return np.sum(self.state["grad_sqr_avg"])

    def grad_var_avg(self) -> float:
        """
        Current estimate of the trace of the covariance of the true gradient
        (mu squared in the AdaScale paper).

        Returns
            (float):
                Estimate of trace of the covariance.
        """
        return np.sum(self.state["grad_var_avg"])

    def gain(self, scale: Optional[float] = None) -> float:
        """
        Current estimate of the AdaScale gain ratio (r_t).

        Args:
            scale (float):
                The batch size scale to estimate the gain ratio for.

        Returns
            :(float):
                Estimate of gain ratio.
        """
        scale = self._scale if scale is None else scale
        var = self.grad_var_avg()
        sqr = self.grad_sqr_avg()
        return (var + sqr) / (var / scale + sqr)

    def _update_avg(self, name: str, value: float, factor: float) -> None:
        biased = self.state.get(name + "_biased", 0.0)
        unbias = self.state.get(name + "_unbias", 0.0)
        biased = factor * biased + (1.0 - factor) * value
        unbias = factor * unbias + (1.0 - factor)
        self.state[name + "_biased"] = biased
        self.state[name + "_unbias"] = unbias
        self.state[name] = biased / unbias

    def _backward_hook(self, idx: int, grad: torch.Tensor) -> None:
        # This method should be invoked once for each parameter during the
        # backward pass, before gradients are synchronized between world_size.
        if self._local_grad_sqr is None:
            self._local_grad_sqr = torch.zeros(len(self._optimizer.param_groups), device=grad.device)
        self._local_grad_sqr[idx] += grad.pow(2).sum()
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
        # gradients have been synchronized between each worker.
        self._final_callback_queued = False
        assert isinstance(self._local_grad_sqr, torch.Tensor)

        # self._local_grad_sqr is FP32, sum then div shouldn't overflow.
        torch.distributed.all_reduce(self._local_grad_sqr)  # SUM
        self._local_grad_sqr.div_(self._world_size)

        local_grad_sqr = self._local_grad_sqr.cpu().numpy()
        total_grad_sqr = np.array(
            [sum(param.grad.pow(2).sum().item() for param in group["params"]) for group in self._optimizer.param_groups]
        )
        grad_sqr = (self._world_size * total_grad_sqr - local_grad_sqr) / (self._world_size - 1)
        grad_var = (local_grad_sqr - total_grad_sqr) * self._scale / (self._world_size - 1)
        grad_sqr = np.maximum(grad_sqr, 0.0)
        grad_var = np.maximum(grad_var, 1e-6)
        theta = self._smoothing ** self._scale
        self._update_avg("grad_sqr_avg", grad_sqr, theta)
        self._update_avg("grad_var_avg", grad_var, theta)
        self._local_grad_sqr = None

    def step(self, *args: Any, **kwargs: Any) -> Optional[float]:
        """
        Run one optimizer step using Adascale. Essentially just invokes
        ``optimizer.step(*args, **kwargs)`` with a scaled learning rate.

        Args:
            args:
                Positional arguments passed to ``optimizer.step``.
            kwargs:
                Keyword arguments passed to ``optimizer.step``.
        Returns:
            (Tensor):
                loss if a closure is passed to the optimizer to reevaluate the model.
        """
        initial_lr = [pg["lr"] for pg in self._optimizer.param_groups]
        for idx, param_group in enumerate(self._optimizer.param_groups):
            grad_sqr = float(self.state["grad_sqr_avg"][idx])
            grad_var = float(self.state["grad_var_avg"][idx])
            gain = (grad_var + grad_sqr) / (grad_var / self._scale + grad_sqr)
            param_group["lr"] = gain * param_group["lr"]
        res = self._optimizer_step(*args, **kwargs)
        for lr, param_group in zip(initial_lr, self._optimizer.param_groups):
            param_group["lr"] = lr
        return res

    def patch_optimizer(self) -> None:
        """
        Monkey-patch the optimizer's step function with :meth:`AdaScale.step`.
        """

        @functools.wraps(self._optimizer.step)
        def wrapper(*args: Any, **kwargs: Any) -> Optional[float]:
            return self.step(*args, **kwargs)

        setattr(self._optimizer, "step", wrapper)

    def zero_grad(self) -> None:
        """Proxy function to optimizer"""
        self._optimizer.zero_grad()
