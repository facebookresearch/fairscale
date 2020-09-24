# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO (Min): enable mypy
# type: ignore


import functools

import numpy as np
from torch.autograd import Variable
import torch.distributed
import torch.optim

__all__ = ["AdaScale"]


def _normsq(params):
    """
    Returns the square of the L2 norm for each elem of params
    as an np array
    """
    return np.asarray([p.pow(2).sum().item() for p in params])


class AdaScale(object):
    """
    Implements the AdaScale_ algorithm for scaling the learning rate for
    distributed and large batch size training. Can be used in combination with
    ``torch.nn.parallel.DistributedDataParallel`` and ``torch.optim.SGD``.

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

    Arguments:
        optimizer (torch.optim.Optimizer): Optimizer to apply AdaScale to.
        scale (float): Scaling factor of the batch size, e.g. using a 10x
            larger batch size (summed across all replicas) means a scale of
            10. If None, defaults to ``num_replicas``.
        num_replicas (int): Number of replicas for distributed training. If
            None, defaults to ``torch.distributed.get_world_size()``.
        patch_optimizer (bool): If True, monkey-patches the ``step`` method of
            the optimizer with the AdaScale ``step`` method.

    .. _AdaScale: https://proceedings.icml.cc/static/paper_files/icml/2020/4682-Supplemental.pdf
    """  # noqa: E501

    def __init__(self, optimizer, scale=None, num_replicas=None, patch_optimizer=False):
        self._optimizer = optimizer
        self._optimizer_step = optimizer.step
        self._sum_local_norm = None
        self._var_future = None
        self._num_replicas = num_replicas if num_replicas is not None else torch.distributed.get_world_size()
        self._num_params = sum(len(pg["params"]) for pg in optimizer.param_groups)

        self._optimizer.state.setdefault(
            "adascale",
            {
                # TODO: prev_grad should not be in state, since it is large and
                # doesn't need to be saved in checkpoints.
                "prev_grad": None,
                "norm": np.zeros(self._num_params),
                "replicas": 0.0,
                # Averages of n and v
                "norm_avg": np.ones(self._num_params),
                "var_avg": np.zeros(self._num_params),
            },
        )

        self.set_scale(self._num_replicas if scale is None else scale)

        idx = 0
        for param_group in self._optimizer.param_groups:
            for param in param_group["params"]:
                param.register_hook(functools.partial(self._backward_hook, idx))
                idx += 1

        if patch_optimizer:
            self.patch_optimizer()

        self._smoothing = 0.997

    @property
    def _state(self):
        return self._optimizer.state["adascale"]

    @property
    def scale(self):
        """
        The scaling factor of the current batch size, relative to the baseline
        batch size when training with a single replica. For example, if the
        baseline batch size is 32, but using a scaled-up batch size of 80, then
        then the scaling factor is 2.5.
        """
        return self._scale

    def set_scale(self, scale):
        """
        Set the scaling factor of the current batch size. It is up to the
        application to invoke this function to make sure that AdaScale's
        scaling factor matches the actual batch size used during training.

        Arguments:
            scale (float): New scaling factor to be applied to AdaScale.
        """
        if self._state["replicas"] == 1 and self._num_replicas > 1:
            # TODO: when to reset running averages should be decided outside of
            # the AdaScale object.
            self._reset_avg("norm")
            self._reset_avg("norm_avg")
            self._reset_avg("var_avg")
        self._scale = scale
        self._state["replicas"] = self._num_replicas

    def norm_avg(self):
        """
        Current estimate of the squared l2-norm of the true gradient (sigma
        squared).

        Returns (float): Estimate of squared l2-norm.
        """
        return np.sum(self._state["norm_avg"])

    def var_avg(self):
        """
        Current estimate of the trace of the covariance of the true gradient
        (mu squared).

        Returns (float): Estimate of trace of the covariance.
        """
        return np.sum(self._state["var_avg"])

    def gain(self, scale=None):
        """
        Current estimate of the AdaScale gain ratio.

        Arguments:
            scale (float): The batch size scale to estimate the gain ratio for.

        Returns (float): Estimate of gain ratio.
        """
        scale = self._scale if scale is None else scale
        var = self.var_avg()
        norm = self.norm_avg()
        return (var + norm) / (var / scale + norm)

    def _update_avg(self, param_name, value, factor):
        biased = self._state.get(param_name + "_biased", 0.0)
        unbias = self._state.get(param_name + "_unbias", 0.0)
        biased = factor * biased + (1.0 - factor) * value
        unbias = factor * unbias + (1.0 - factor)
        self._state[param_name + "_biased"] = biased
        self._state[param_name + "_unbias"] = unbias
        self._state[param_name] = biased / unbias

    def _reset_avg(self, param_name):
        self._state.pop(param_name + "_biased", None)
        self._state.pop(param_name + "_unbias", None)

    def _backward_hook(self, idx, grad):
        # This method should be invoked once for each parameter during the
        # backward pass, before gradients are synchronized between replicas.
        if self._num_replicas > 1:
            if self._sum_local_norm is None:
                self._sum_local_norm = torch.zeros(self._num_params, device=grad.device)
            self._sum_local_norm[idx] = grad.pow(2).sum()
        self._final_callback_queued = False
        Variable._execution_engine.queue_callback(self._queue_callback)

    def _queue_callback(self):
        # This method should be invoked after the entire backward pass. We want
        # to make sure self._final_callback is invoked once, only after all
        # gradients have been synchronized between each replica. However, the
        # synchronization code in DistributedDataParallel is also done in a
        # callback, which might not yet be executed. Therefore, we enqueue
        # self._final_callback from this method, which should ensure it is
        # invoked after the gradient synchronization callback.
        if self._final_callback_queued:
            return
        self._final_callback_queued = True
        Variable._execution_engine.queue_callback(self._final_callback)

    def _final_callback(self):
        # This method should be invoked once for each backward pass, after
        # gradients have been synchronized between each replica.
        self._final_callback_queued = False
        grad = []
        for group in self._optimizer.param_groups:
            grad.extend([p.grad.clone() for p in group["params"]])
        theta = self._smoothing ** self._scale
        replicas = self._state["replicas"]
        if replicas > 1:
            if self._var_future is not None:
                self._var_future[0].wait()
                n = _normsq(self._state["prev_grad"])
                var = self._var_future[1].cpu().numpy() / (replicas - 1)
                var -= n * (replicas / (replicas - 1))
                var *= self._scale / replicas
                var = np.maximum(var, 1e-6)
                norm = n - var / self._scale
                norm = np.maximum(norm, 0.0)
                self._update_avg("norm_avg", norm, theta)
                self._update_avg("var_avg", var, theta)
            self._var_future = (torch.distributed.all_reduce(self._sum_local_norm, async_op=True), self._sum_local_norm)
            self._sum_local_norm = None
        else:  # Single replica, use difference estimation.
            prev_grad = self._state["prev_grad"]
            if prev_grad is not None:
                n = _normsq([(g1 + g2) / 2 for g1, g2 in zip(prev_grad, grad)])
                var = np.array([(g1.pow(2).sum() + g2.pow(2).sum()).item() for g1, g2 in zip(prev_grad, grad)])
                var -= 2 * n
                var *= self._scale
                var = np.maximum(var, 1e-6)
                norm = n - var / (2 * self._scale)
                norm = np.maximum(norm, 0.0)
                self._update_avg("norm_avg", norm, theta)
                self._update_avg("var_avg", var, theta)
        self._state["prev_grad"] = grad

    def step(self, *args, **kwargs):
        """
        Run one optimizer step using Adascale. Essentially just invokes
        ``optimizer.step(*args, **kwargs)`` with a scaled learning rate.

        Arguments:
            args: Positional arguments passed to ``optimizer.step``.
            kwargs: Keyword arguments passed to ``optimizer.step``.
        """
        initial_lr = [pg["lr"] for pg in self._optimizer.param_groups]
        offset = 0
        for param_group in self._optimizer.param_groups:
            size = len(param_group["params"])
            grad_sqr = self._state["norm_avg"][offset : offset + size].sum()
            grad_var = self._state["var_avg"][offset : offset + size].sum()
            gain = (grad_var + grad_sqr) / (grad_var / self._scale + grad_sqr)
            param_group["lr"] = gain * param_group["lr"]
            offset += size
        self._optimizer_step(*args, **kwargs)
        for lr, param_group in zip(initial_lr, self._optimizer.param_groups):
            param_group["lr"] = lr

    def patch_optimizer(self):
        """
        Monkey-patch the optimizer's step function with :meth:`AdaScale.step`.
        """
        # TODO: detect if the optimizer has already been patched.

        @functools.wraps(self._optimizer.step)
        def wrapper(*args, **kwargs):
            return self.step(*args, **kwargs)

        self._optimizer.step = wrapper
