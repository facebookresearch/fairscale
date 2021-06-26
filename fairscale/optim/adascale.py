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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import numpy as np
import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.optim import SGD, Optimizer

if TYPE_CHECKING:  # pragma: no cover
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


class AdaScale(Optimizer):
    """
    Implements the AdaScale_ algorithm for scaling the learning rate for
    distributed and large batch size training. Can be used in combination with
    ``torch.nn.parallel.DistributedDataParallel`` and ``torch.optim.SGD``.

    .. _AdaScale: https://proceedings.icml.cc/static/paper_files/icml/2020/4682-Supplemental.pdf

    This class subclasses `Optimizer` so that `torch.optim.lr_scheduler` can
    work with it. In other words, AdaScale is intended to be a complete wrapper of an
    torch Optimizer.

    Note that, AdaScale does *not* help increase per-GPU batch size.

    There are several ways to integrate AdaScale with your training loop.
    We show two examples below.

    Example 1: using PyTorch's `lr_scheduler` classes.

    .. code-block:: python

        optim = AdaScale(SGD(model.parameters(), lr=0.001))
        model = DistributedDataParallel(model)
        scheduler = LambdaLR(optim, lr_lambda=...)

        last_epoch = 0
        done = False
        step = 0
        while not done:
            for batch in dataset:
                optim.zero_grad()
                logits = model()
                loss = criterion(logits, ...)
                loss.backward()
                step += optim.gain()
                optim.step()
                epoch = step // len(dataset)
                if epoch > last_epoch:
                    scheduler.step()
                    last_epoch = epoch
                if epoch >= MAX_EPOCHS:
                    done = True

    Example 2: using a custom `update_lr()` function that update the learning
    rate based on the current step count per epoch.

    .. code-block:: python

        optim = AdaScale(SGD(model.parameters(), lr=0.001))
        model = DistributedDataParallel(model)

        step = 0
        while step < max_steps:
            for batch in ...:
                optim.zero_grad()
                logits = model()
                loss = criterion()
                loss.backward()
                step += optim.gain()
                optim.step()
                update_lr(step)

    Args:
        optimizer (torch.optim.Optimizer):
            Optimizer to apply AdaScale to.
        world_size (int):
            Number of world_size for distributed training.
            If None, defaults to ``dist.get_world_size()``.
        scale (float):
            Scaling factor of the batch size from scale equals 1, e.g. using a 10x
            larger batch size (summed across all ranks with gradient accumulation)
            means a scale of 10.
            If None, defaults to ``world_size * num_gradients_to_accumulate``.
        smoothing (float):
            Smoothing factor for moving average.
            If None, it defaults to ``max(1 - (world_size * num_gradients_to_accumulate)/1000, 0)``.
            Note, for very high scale training, higher smoothing value might be needed,
            esp at the begining of the training. Therefore, if your scale is close to or larger
            than 1000, try experimenting with smoothing value > 0 if the final accuracy is poor.
        num_gradients_to_accumulate (int):
            Number of passes that we accumulate gradients locally
            between each optimizer step. This can be changed during
            training as long as the train loop changes gradient accumulation
            accordingly.
            Default to 1, which does not accumulate gradients.
        debias_ewma (bool):
            (experimental) Use debias exponential moving average
            for smoothing and mu and sigma variables. False will
            use the method in the paper's Appendix B.3.
            Default: True, which is what have been validated so far.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        world_size: Optional[int] = None,
        scale: Optional[float] = None,
        smoothing: float = None,
        num_gradients_to_accumulate: int = 1,
        debias_ewma: bool = True,
    ):
        self._optimizer = optimizer
        self._local_grad_sqr: Optional[torch.Tensor] = None
        self._world_size: int = (
            world_size if world_size is not None else dist.get_world_size() if dist.is_initialized() else 1
        )
        self._num_backward_calls = 0
        self._last_final_backward_call = 0
        self._num_grads_to_accum = num_gradients_to_accumulate
        self._debias_ewma = debias_ewma

        # Proxy the param_groups so that `torch.optim.lr_scheduler` can work.
        self.param_groups = self._optimizer.param_groups

        self.set_num_gradients_to_accumulate(num_gradients_to_accumulate, update_smoothing=True)

        # The previous function call sets smoothing to its default value.
        # Override that here if smoothing was passed as an argument.
        if smoothing is not None:
            self._smoothing = smoothing

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

        self._scale = 1.0  # Assign to inform mypy about the typing of this variable.
        self.set_scale(self._world_size * self._num_grads_to_accum if scale is None else scale)

        self._hook_handles: List[Any] = []
        self._hook()

    def _hook(self) -> None:
        """ Internal function to register the gradient hooks.

            Note, don't assume every parameter will generate a gradient (i.e. triggering the hook)
            in every backward pass, which is the reason that we have ``find_unused_params`` flag
            in the DDP class in ``torch.nn.parallel``.
        """
        assert self._hook_handles == [], "Must run unhook first"
        for idx, param_group in enumerate(self._optimizer.param_groups):
            for param in param_group["params"]:
                h = param.register_hook(functools.partial(self._backward_hook, idx))
                self._hook_handles.append(h)

    def __del__(self) -> None:
        """ Unhook in case caller forgets to call unhook.

            This however may not "work" since there would be circular reference
            between the hook objects and this objects. In that case, neither will
            get GC'ed. Calling unhook explicitly if you really want to delete
            AdaScale from memory.
        """
        self.unhook()

    def unhook(self) -> None:
        """ Unregister hook handles.

            This is public because caller may need to call this to ensure all GPU
            memory are released. Otherwise, the hook may prevent parameters from being
            released from the GPU memory pool.

            Internally, we use this to support ``add_param_group()`` API.
        """
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    @property
    def _state(self) -> Dict[str, np.ndarray]:
        """
        Return the states of AdaScale.
        """
        return self._optimizer.state["adascale"]

    @property
    def scale(self) -> float:
        """
        The scaling factor of the current batch size, relative to the baseline
        batch size, which could be a DDP training. For example, if the
        baseline batch size is 32 on 2 GPUs, but using a scaled-up batch size
        of 80 on 4 GPUs, then then the scaling factor is 80 * 4 / 32 / 2 = 5.

        This is exposed API mainly for logging purpose. Note, this is different
        from ``self.gain()``.

        Returns:
            (float):
                The current scaling factor.
        """
        return self._scale

    @property
    def smoothing(self) -> float:
        """
        The smoothing constant used in exponentially-weighted moving average
        tracking the gradient norm mean and variance within AdaScale.

        This is exposed API since the value is computed and caller may
        want to obtain this value and log it.

        Returns:
            (float):
                The current smoothing value.
        """
        return self._smoothing

    def set_scale(self, scale: float, update_estimate: bool = True) -> None:
        """
        Set the scaling factor of the current batch size. It is up to the
        application to invoke this function to make sure that AdaScale's
        scaling factor matches the actual batch size used during training.

        Args:
            scale (float):
                New scaling factor to be applied to AdaScale.
            update_estimate (bool):
                Whether to update the scale-depenent estimate of gradient
                variance; this is highly recommended. (default: True)
        """
        assert self._local_grad_sqr is None, "Don't change scale in backward phase"
        assert scale >= 1, "Scale must be at least 1"
        if update_estimate and hasattr(self, "_scale"):
            assert self._scale >= 1, "bug: old scale isn't valid"
            # Rescale grad_var_avg to account for the change in scale
            if self._debias_ewma and "grad_var_avg_biased" in self._state:
                self._state["grad_var_avg_biased"] *= self._scale / scale
            elif "grad_var_avg_total" in self._state:  # _debias_ewma==False
                self._state["grad_var_avg_total"] *= self._scale / scale
            self._state["grad_var_avg"] *= self._scale / scale
        self._scale = scale

    def _grad_sqr_avg(self, pg_idx: Optional[int] = None) -> float:
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
            return self._state["grad_sqr_avg"][pg_idx]
        else:
            return float(np.sum(self._state["grad_sqr_avg"]))

    def _grad_var_avg(self, pg_idx: Optional[int] = None) -> float:
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
            return self._state["grad_var_avg"][pg_idx]
        else:
            return float(np.sum(self._state["grad_var_avg"]))

    def gain(self, pg_idx: Optional[int] = None) -> float:
        """
        Current estimate of the AdaScale gain ratio (r_t in the paper).

        Args:
            pg_idx (int):
                Optional index of a parameter group.
                Default None: returns "averaged" gain for all groups.

        Returns:
            (float):
                Estimate of gain ratio.
        """
        var = self._grad_var_avg(pg_idx)
        sqr = self._grad_sqr_avg(pg_idx)
        gain = (var + sqr) / (var / self.scale + sqr)
        return gain

    def _update_avg(self, name: str, value: np.ndarray, factor: float) -> None:
        if self._debias_ewma:
            # This function computes and stores the moving average of a vector
            # using a smoothing factor.
            biased = self._state.get(name + "_biased", np.zeros(value.shape[0]))
            unbias = self._state.get(name + "_unbias", np.zeros(value.shape[0]))
            biased = factor * biased + (1.0 - factor) * value
            unbias = factor * unbias + (1.0 - factor)
            self._state[name + "_biased"] = biased
            self._state[name + "_unbias"] = unbias
            self._state[name] = biased / unbias
        else:
            # Moving average procedure described in Appendix B.3
            # For iterations t < 1 / (1 - smoothing) define grad_var_avg
            # and grad_sqr_avg as mean of the past samples. After that
            # start using running average.
            #
            # Note: we only keep a single _count for all parameter groups.
            #       Ideally, it should be a vector and in case a PG is added
            #       after some iterations are done. But, then the if condition
            #       below will need to be a np.where. I leave this corner
            #       case to a future exercise.
            count = self._state.get(name + "_count", np.zeros(1))
            count[0] += 1
            self._state[name + "_count"] = count
            if count < 1 / (1 - self._smoothing):
                total = self._state.get(name + "_total", None)
                if total is None:
                    total = value
                else:
                    total += value
                self._state[name + "_total"] = total
                self._state[name] = total / count
            else:
                self._state[name] = factor * self._state[name] + (1.0 - factor) * value

    def _backward_hook(self, pg_idx: int, grad: torch.Tensor) -> None:
        # This method should be invoked once for each parameter during the
        # backward pass, before gradients are synchronized between world_size.

        # Store the local gradient square sums in a vector.
        # This vector is also used for error checking. Whenever it is not None,
        # it means that we are in backward pass.
        if self._local_grad_sqr is None:
            self._local_grad_sqr = torch.zeros(
                len(self._optimizer.param_groups), device=grad.device, requires_grad=False,
            )
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
        # TODO (min): this may not work with activation checkpointing when
        #             multiple backward calls happen in a big backward.
        self._num_backward_calls += 1

        # TODO (min, mike): We need to have a way to check that training loop & DDP
        #                   is doing the right thing where the gradient is reduced
        #                   in this backward pass.
        #                   Longer term, we may compute the gain and then inform
        #                   the training loop when it is a good time to step().
        assert (
            self._num_backward_calls - self._last_final_backward_call
        ) <= self._num_grads_to_accum, (
            f"bug: {self._num_backward_calls} - {self._last_final_backward_call} should <= {self._num_grads_to_accum}"
        )
        if (self._num_backward_calls - self._last_final_backward_call) % self._num_grads_to_accum != 0:
            assert self._local_grad_sqr is not None, "We should still be in backward phase"
            return

        # Since self._local_grad_sqr is FP32, sum shouldn't overflow.
        # This vector has length of # of param_groups, so it is small, but we
        # use async to hide the all_reduce latency, esp when # of nodes is large.
        work = None
        if self._world_size > 1:
            work = dist.all_reduce(self._local_grad_sqr, async_op=True)  # SUM

        # Compute the sums of squares for reduced gradients.
        # Divide by _num_grads_to_accum since the gradients are accumulated.
        total_grad_sqr = np.array(
            [sum(param.grad.pow(2).sum().item() for param in group["params"]) for group in self._optimizer.param_groups]
        )
        # Divide by (_num_grads_to_accum ** 2) to account for gradient
        # accumulation.
        if self._num_grads_to_accum > 1:
            # np array doesn't support /=.
            total_grad_sqr = total_grad_sqr / (self._num_grads_to_accum ** 2)

        # Wait for all_reduce to be done and move it to cpu & np.
        if work:
            work.wait()
        local_grad_sqr = self._local_grad_sqr.cpu().numpy()

        # See appendix B.3 of the paper.
        # Modified to handle cases where scale != world_size
        #
        # local_grad_sqr is \sum_{i=1}^{c N} \norm{g_t_i}^2
        # where N is world size and c is num_grads_to_accum
        # total_grad_sqr is \norm{\bar{g}_t}^2
        S = self._scale
        cN = self._world_size * self._num_grads_to_accum
        grad_var = local_grad_sqr * (S / cN) / (cN - 1) - total_grad_sqr * S / (cN - 1)
        grad_sqr = total_grad_sqr - grad_var / S
        grad_var = np.maximum(grad_var, 1e-6)
        grad_sqr = np.maximum(grad_sqr, 0.0)
        self._update_avg("grad_sqr_avg", grad_sqr, self.smoothing)
        self._update_avg("grad_var_avg", grad_var, self.smoothing)
        self._last_final_backward_call = self._num_backward_calls
        # Indicating backward is done.
        self._local_grad_sqr = None

    def step(self, *args: Any, **kwargs: Any) -> Optional[float]:
        """
        Run one optimizer step using Adascale. Essentially just invokes
        ``optimizer.step(*args, **kwargs)`` with a scaled learning rate.

        .. note::

            It is possible that this function becames a performance
            bottleneck if you have frequent updates. To avoid that,
            making bigger steps and reducing update frequency is generally
            better for performance.

        Args:
            args (Any):
                Positional arguments passed to ``optimizer.step``.
            kwargs (Any):
                Keyword arguments passed to ``optimizer.step``.

        Returns:
            (Tensor):
                The loss tensor if a closure if used to re-evaluate the model.
        """
        assert self._local_grad_sqr is None, "Don't step without finishing backward phase"
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

    def add_param_group(self, pg: Dict) -> None:
        """ Support adding parameter groups

            We need to re-size some of the state and re-register the backward hooks.
        """
        assert self._local_grad_sqr is None, "Can't add parameter group during backward"
        self._optimizer.add_param_group(pg)
        # Update the hooks.
        self.unhook()
        self._hook()
        # Extend the states.
        for name in self._state.keys():
            assert name.startswith("grad_sqr_avg") or name.startswith("grad_var_avg"), name
            if name.endswith("_count"):
                # This is the "_count" variable, should be a 1D int.
                assert self._state[name].shape == (1,), self._state[name].shape
                continue
            # must be a np array, extend it with the right value and check the shape.
            val = 1 if name == "grad_sqr_avg" else 0
            self._state[name] = np.append(self._state[name], val)  # type: ignore
            assert self._state[name].shape == (len(self._optimizer.param_groups),)

    def zero_grad(self) -> None:
        """Proxy function to optimizer, because some training loops need this."""
        assert self._local_grad_sqr is None, "Don't zero_grad in backward"
        return self._optimizer.zero_grad()

    def state_dict(self) -> Dict:
        """ Proxy function to optimizer, checkpointing needs this.

            .. note::

                Do NOT checkpoint in the middle of gradient accumulation since
                associated AdaScale internal states are not saved in the checkpoint.
        """
        assert self._local_grad_sqr is None, "Don't checkpoint in backward"
        return self._optimizer.state_dict()

    def load_state_dict(self, data: Dict) -> None:
        """ Proxy function to optimizer, checkpointing needs this.

            .. note::

                Do NOT checkpoint in the middle of gradient accumulation since
                associated AdaScale internal states are not saved in the checkpoint.
        """
        assert self._local_grad_sqr is None, "Don't load checkpoint in backward"
        return self._optimizer.load_state_dict(data)

    def set_num_gradients_to_accumulate(self, num_gradients_to_accumulate: int, update_smoothing: bool = True,) -> None:
        """Set the number of gradients to accumulate to a new value.

        This is experimental. This could be called while training so that
        we can gradually increasing the steps between updates. Almost always,
        `set_scale` needs to be called to update the scale as well.

        TODO (min): need a way of determine how much to increase the step size?

        TODO (min): have both `set_scale` and `set_num_gradients_to_accumulate`
        is hard to use and easy to make mistake. I think it is better
        to specific a specify a `base_scale`. But more discussion is
        needed here.

        Args:
            num_gradients_to_accumulate (int):
                Number of gradients to accumulate (calls to backward) between
                each optimizer step
            update_smoothing (bool):
                Whether to update smoothing factor or not. Default: True.
        """
        assert self._local_grad_sqr is None, "Don't change num_grad_to_accum in backward"
        assert num_gradients_to_accumulate >= 1, f"Invalid value {num_gradients_to_accumulate}"
        self._num_grads_to_accum = num_gradients_to_accumulate
        if update_smoothing:
            # Set smoothing based on effective world_size rather than scale here,
            # since world_size determines the number of samples being averaged over
            # at every update.
            #
            # When effective world size is large enough, smoothing is probably
            # not needed, so the smoothing factor is 0.
            self._smoothing = max(1 - self._world_size * self._num_grads_to_accum / 1000, 0)

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped optimizer."""
        try:
            return super().__getattr__(name)  # defer to Optimizer logic
        except AttributeError:
            return getattr(self._optimizer, name)  # fallback to wrapped optim


class AdaScaleWrapper(AdaScale):
    """
    A thin wrapper for AdaScale so that the constructor resembles a
    standard optimizer. This allows it to work with other Optimizer
    Wrappers, like `OSS`.

    .. warn::
        OSS(AdaScaleWrapper) (i.e. OSS wrapping AdaScale) resulting in each
        rank's AdaScale operates on different set of parameters. They
        will get different gain values and it is unclear how to adjust
        effective step size in that case. We have not validated effectiveness
        or benefit in this case.

        OTOH, AdaScale(OSS) (i.e. AdaScale wrapping OSS) is recommended
        and is numerically identical to AdaScale without OSS. Since
        AdaScale doesn't incur per-parameter state, the memory benefit
        of OSS is still the same.

    Args:
        params (list of tensors):
            parameters to be optimized
        optim (class subtyping torch.optim.Optimizer):
            a optimizer class to be wrapped.
        additional_optim_args (argument dict):
            keyward arguments to the `optim` class above.

        The rest params are in-sync with the `AdaScale` class above.
    """

    def __init__(
        self,
        params: _params_t,
        world_size: Optional[int] = None,
        scale: Optional[float] = None,
        smoothing: float = None,
        num_gradients_to_accumulate: int = 1,
        debias_ewma: bool = True,
        optim_cls: Type[Optimizer] = SGD,
        **additional_optim_args: Any,
    ):
        optim_obj = optim_cls(params, **additional_optim_args)
        super().__init__(optim_obj, world_size, scale, smoothing, num_gradients_to_accumulate, debias_ewma)
