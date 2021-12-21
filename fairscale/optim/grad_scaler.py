# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import abc, defaultdict
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Union
import warnings

import torch
from torch.cuda import FloatTensor  # type: ignore
from torch.cuda.amp import GradScaler as TorchGradScaler
from torch.cuda.amp.common import amp_definitely_not_available
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.sgd import SGD

from fairscale.utils import torch_version


class _GeneralMultiDeviceReplicator(object):
    """
    Lazily serves copies of a tensor to requested devices.  Copies are cached per-device.
    This class adds the cpu option to the _MultiDeviceReplicator class in PyTorch grad_scaler.py.
    https://pytorch.org/docs/stable/_modules/torch/cuda/amp/grad_scaler.html#GradScaler
    """

    def __init__(self, master_tensor: torch.Tensor) -> None:
        assert master_tensor.is_cuda or master_tensor.device.type == "xla" or master_tensor.device.type == "cpu"
        self.master = master_tensor
        self._per_device_tensors: Dict[torch.device, torch.Tensor] = {}

    def get(self, device: torch.device) -> torch.Tensor:
        retval = self._per_device_tensors.get(device, None)
        if retval is None:
            retval = self.master.to(device=device, non_blocking=True, copy=True)
            self._per_device_tensors[device] = retval
        return retval


# Defines default_factory for GradScaler's _per_optimizer_states defaultdict,
# as well as associated "enum" values.  Prefers defining these at top level because
# - Lambdas can't be pickled, so we don't want to supply a lambda as the factory.
# - Defining READY, UNSCALED, STEPPED and _refresh_per_optimizer_state within GradScaler
#   causes a circular reference, which we'd rather avoid.
class OptState(Enum):

    READY = 0
    UNSCALED = 1
    STEPPED = 2


def _refresh_per_optimizer_state() -> Dict:
    return {"stage": OptState.READY, "found_inf_per_device": {}}


class GradScaler(TorchGradScaler):
    def _unscale_grads_(
        self,
        optimizer: Optimizer,
        inv_scale: torch.Tensor,
        found_inf: torch.Tensor,
        allow_fp16: bool,
    ) -> Dict[torch.device, torch.Tensor]:
        return super()._unscale_grads_(optimizer, inv_scale, found_inf, True)


class ShardedGradScaler(TorchGradScaler):
    """
    A shard aware Grad Scaler which enables loss scaling with/without cpu_offload. This is a
    slight modification of the pytorch grad scaler.
    https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler
    """

    def __init__(
        self,
        init_scale: float = 2.0 ** 16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
        process_group: Any = dist.group.WORLD,
    ):
        super().__init__(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
        if enabled and amp_definitely_not_available():
            warnings.warn("torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.")
            self._enabled = False
        else:
            self._enabled = enabled

        if self._enabled:
            self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)
            self.group = process_group

    def scale(self, outputs: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, abc.Iterable]:
        """
        Multiplies ('scales') a tensor or list of tensors by the scale factor.

        Returns scaled outputs.  If this instance of :class:`GradScaler` is not enabled, outputs are returned
        unmodified.

        Args:
            outputs (Tensor or iterable of Tensors):  Outputs to scale.
        """
        if not self._enabled:
            return outputs

        # Short-circuit for the common case.
        if isinstance(outputs, torch.Tensor):
            assert outputs.is_cuda or outputs.device.type == "xla" or outputs.device.type == "cpu"
            if self._scale is None:
                self._lazy_init_scale_growth_tracker(outputs.device)  # type: ignore
            assert self._scale is not None
            return outputs * self._scale.to(device=outputs.device, non_blocking=True)

        # Invoke the more complex machinery only if we're treating multiple outputs.
        stash: List[_GeneralMultiDeviceReplicator] = []  # holds a reference that can be overwritten by apply_scale

        def apply_scale(val: Union[torch.Tensor, abc.Iterable]) -> Union[torch.Tensor, abc.Iterable]:
            if isinstance(val, torch.Tensor):
                assert val.is_cuda or val.device.type == "xla" or val.device.type == "cpu"
                if len(stash) == 0:
                    if self._scale is None:
                        self._lazy_init_scale_growth_tracker(val.device)  # type: ignore
                    assert self._scale is not None
                    stash.append(_GeneralMultiDeviceReplicator(self._scale))
                return val * stash[0].get(val.device)
            elif isinstance(val, abc.Iterable):
                iterable = map(apply_scale, val)
                if isinstance(val, list) or isinstance(val, tuple):
                    return type(val)(iterable)
                else:
                    return iterable
            else:
                raise ValueError("outputs must be a Tensor or an iterable of Tensors")

        return apply_scale(outputs)

    # This function is required enable cpu based grad scaler. It is inspired from its corresponding CUDA
    # implementation which can be found here
    # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/AmpKernels.cu#L88
    def _foreach_non_finite_check_and_unscale_cpu_(
        self, grads: List, found_inf: torch.Tensor, inv_scale: torch.Tensor
    ) -> None:
        if len(grads) == 0:
            return
        assert inv_scale.numel() == 1, "inv_scale must be a 1-element tensor."
        assert found_inf.numel() == 1, "found_inf must be a 1-element tensor."

        expected_device = grads[0].device
        for tensor in grads:
            try:
                assert tensor.device == expected_device, "grads must be on the same device"
            except AssertionError:
                logging.error("tensor device is %s and expected device is %s" % (tensor.device, expected_device))

            # check for non_overlapping_and_dense doesn't exist in the python world
            # as remarked here https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/AmpKernels.cu#L108
            # we assume tensor is not MTA(multi tensor apply) safe. iterate through each item regardless of dtype
            if torch.isinf(tensor).any().item() is True or torch.isnan(tensor).any().item() is True:  # type: ignore
                found_inf.data = torch.tensor([1.0])
                break
            else:
                tensor.data *= inv_scale.item()

    def _unscale_grads_(  # type: ignore
        self, optimizer: SGD, inv_scale: torch.Tensor, found_inf: torch.Tensor, allow_fp16: bool = True
    ) -> Dict[torch.device, torch.Tensor]:
        per_device_inv_scale = _GeneralMultiDeviceReplicator(inv_scale)
        per_device_found_inf = _GeneralMultiDeviceReplicator(found_inf)

        # To set up _amp_foreach_non_finite_check_and_unscale_, split grads by device and dtype.
        # There could be hundreds of grads, so we'd like to iterate through them just once.
        # However, we don't know their devices or dtypes in advance.

        # https://stackoverflow.com/questions/5029934/defaultdict-of-defaultdict
        # Google says mypy struggles with defaultdicts type annotations.
        per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))  # type: ignore[var-annotated]
        with torch.no_grad():
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    if (not allow_fp16) and param.grad.dtype == torch.float16:
                        raise ValueError("Attempting to unscale FP16 gradients.")
                    if param.grad.is_sparse:
                        # is_coalesced() == False means the sparse grad has values with duplicate indices.
                        # coalesce() deduplicates indices and adds all values that have the same index.
                        # For scaled fp16 values, there's a good chance coalescing will cause overflow,
                        # so we should check the coalesced _values().
                        if param.grad.dtype is torch.float16:
                            param.grad = param.grad.coalesce()
                        to_unscale = param.grad._values()
                    else:
                        to_unscale = param.grad

                    # TODO: is there a way to split by device and dtype without appending in the inner loop?
                    per_device_and_dtype_grads[to_unscale.device][to_unscale.dtype].append(to_unscale)

            for device, per_dtype_grads in per_device_and_dtype_grads.items():
                for grads in per_dtype_grads.values():
                    if grads[0].device.type == "cpu":
                        self._foreach_non_finite_check_and_unscale_cpu_(
                            grads,
                            per_device_found_inf.get(device),
                            per_device_inv_scale.get(device),
                        )
                    else:
                        torch._amp_foreach_non_finite_check_and_unscale_(  # type: ignore
                            grads,
                            per_device_found_inf.get(device),
                            per_device_inv_scale.get(device),
                        )

        return per_device_found_inf._per_device_tensors

    def unscale_(self, optimizer: SGD) -> None:  # type: ignore
        if not self._enabled:
            return

        super()._check_scale_growth_tracker("unscale_")  # type: ignore

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.UNSCALED:
            raise RuntimeError("unscale_() has already been called on this optimizer since the last update().")
        elif optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")

        # FP32 division can be imprecise for certain compile options, so we carry out the reciprocal in FP64.
        assert self._scale is not None
        inv_scale = self._scale.double().reciprocal().float()
        found_inf = torch.full((1,), 0.0, dtype=torch.float32, device=self._scale.device)

        optimizer_state["found_inf_per_device"] = self._unscale_grads_(optimizer, inv_scale, found_inf, True)
        optimizer_state["stage"] = OptState.UNSCALED

        # Synchronize the detected inf across the ranks
        optimizer_state = self._per_optimizer_states[id(optimizer)]
        last_handle = None

        for v in optimizer_state["found_inf_per_device"].values():
            if v.device.type == "cpu":
                v_on_cuda = v.cuda()
                last_handle = dist.all_reduce(v_on_cuda, async_op=True, group=self.group)
                v_on_cuda.cpu()
            else:
                last_handle = dist.all_reduce(v, async_op=True, group=self.group)

        # Make sure that the calls are done before moving out.
        # The calls are executed in sequence, waiting for the last one is enough
        if last_handle is not None:
            last_handle.wait()

    def step(self, optimizer: SGD, *args, **kwargs) -> Optional[float]:  # type: ignore
        """
        :meth:`step` carries out the following two operations:

        1.  Internally invokes ``unscale_(optimizer)`` (unless :meth:`unscale_` was explicitly called for ``optimizer``
            earlier in the iteration).  As part of the :meth:`unscale_`, gradients are checked for infs/NaNs.
        2.  If no inf/NaN gradients are found, invokes ``optimizer.step()`` using the unscaled
            gradients.  Otherwise, ``optimizer.step()`` is skipped to avoid corrupting the params.

        ``*args`` and ``**kwargs`` are forwarded to ``optimizer.step()``.

        Returns the return value of ``optimizer.step(*args, **kwargs)``.

        Args:
            optimizer (torch.optim.Optimizer):  Optimizer that applies the gradients.
            args:  Any arguments.
            kwargs:  Any keyword arguments.

        .. warning::
            Closure use is not currently supported.

        Note: This is an exact copy of the step function in grad_scaler.py. If this copy is deleted then the
        unittest test_cpu_offload_and_cpu_grads fails. This is because the parent class step function calls
        the parent class unscale_ function which does not handle torch.distributed.all_reduce on cpu.
        """
        if not self._enabled:
            return optimizer.step(*args, **kwargs)

        if "closure" in kwargs:
            raise RuntimeError("Closure use is not currently supported if GradScaler is enabled.")

        self._check_scale_growth_tracker("step")  # type: ignore

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("step() has already been called since the last update().")

        retval = None

        if hasattr(optimizer, "_step_supports_amp_scaling") and optimizer._step_supports_amp_scaling:
            # This optimizer has customized scale-handling logic, so we can call optimizer.step() directly.
            # The contract with custom optimizers is that their step() should accept an additional,
            # optional grad_scaler kwarg.  We append self to the kwargs so the custom optimizer has full information:
            # it can query its own state, invoke unscale_ on itself, etc
            retval = optimizer.step(*args, **dict(kwargs, grad_scaler=self))
            optimizer_state["stage"] = OptState.STEPPED
            return retval

        if optimizer_state["stage"] is OptState.READY:
            self.unscale_(optimizer)

        assert len(optimizer_state["found_inf_per_device"]) > 0, "No inf checks were recorded for this optimizer."
        retval = self._maybe_opt_step(optimizer, optimizer_state, *args, **kwargs)  # type: ignore
        optimizer_state["stage"] = OptState.STEPPED
        return retval

    # This function is required enable cpu based grad scaler. It is inspired from its corresponding CUDA
    # implementation which can be found here
    # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/AmpKernels.cu#L219
    def _amp_update_scale_cpu_(self, found_inf):  # type: ignore
        """
        If found_inf is 1.0 (True), then scale is multiplied by backoff_factor and growth_tracker is set to zero.
        Otherwise, scale is multiplied by the growth factor when the growth interval is reached.
        """
        if found_inf.item() == 1.0:
            self._scale *= self._backoff_factor  # type: ignore
            self._growth_tracker = 0
        else:
            successful = self._growth_tracker + 1
            if successful == self._growth_interval:  # type: ignore
                self._scale *= self._growth_factor  # type: ignore
                self._growth_tracker = 0
            else:
                self._growth_tracker = successful

    def update(self, new_scale: Optional[Union[float, FloatTensor]] = None) -> None:
        """
        Updates the scale factor.

        If any optimizer steps were skipped the scale is multiplied by ``backoff_factor``
        to reduce it. If ``growth_interval`` unskipped iterations occurred consecutively,
        the scale is multiplied by ``growth_factor`` to increase it.

        Passing ``new_scale`` sets the new scale value manually. (``new_scale`` is not
        used directly, it's used to fill GradScaler's internal scale tensor. So if
        ``new_scale`` was a tensor, later in-place changes to that tensor will not further
        affect the scale GradScaler uses internally.)

        Args:
            new_scale (float or :class:`torch.cuda.FloatTensor`, optional, default=None):  New scale factor.

        .. warning::
            :meth:`update` should only be called at the end of the iteration, after ``scaler.step(optimizer)`` has
            been invoked for all optimizers used this iteration.
        """

        if not self._enabled:
            return

        _scale, _growth_tracker = self._check_scale_growth_tracker("update")  # type: ignore

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale.fill_(new_scale)
            else:
                reason = "new_scale should be a float or a 1-element torch.cuda.FloatTensor with requires_grad=False."
                assert isinstance(new_scale, torch.cuda.FloatTensor), reason  # type: ignore[attr-defined]
                assert new_scale.numel() == 1, reason
                assert new_scale.requires_grad is False, reason
                self._scale.copy_(new_scale)
        else:
            # Consume shared inf/nan data collected from optimizers to update the scale.
            # If all found_inf tensors are on the same device as self._scale, this operation is asynchronous.
            found_infs = [
                found_inf.to(device=_scale.device, non_blocking=True)
                for state in self._per_optimizer_states.values()
                for found_inf in state["found_inf_per_device"].values()
            ]

            assert len(found_infs) > 0, "No inf checks were recorded prior to update."

            found_inf_combined = found_infs[0]
            if len(found_infs) > 1:
                for i in range(1, len(found_infs)):
                    found_inf_combined += found_infs[i]

            if _scale.device.type == "cpu":
                self._amp_update_scale_cpu_(found_inf_combined)  # type: ignore
            else:
                if torch_version() >= (1, 9, 0):
                    torch._amp_update_scale_(  # type: ignore
                        self._scale,
                        self._growth_tracker,
                        found_inf_combined,
                        self._growth_factor,  # type: ignore
                        self._backoff_factor,  # type: ignore
                        self._growth_interval,  # type: ignore
                    )
                elif torch_version() >= (1, 8, 0) and torch_version() < (1, 9, 0):
                    self._scale = torch._amp_update_scale(  # type: ignore
                        self._growth_tracker,
                        _scale,
                        found_inf_combined,
                        self._growth_factor,  # type: ignore
                        self._backoff_factor,  # type: ignore
                        self._growth_interval,  # type: ignore
                    )

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)
