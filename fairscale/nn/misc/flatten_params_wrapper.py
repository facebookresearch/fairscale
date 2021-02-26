# Copyright (c) Tongzhou Wang
# Licensed under the MIT License.

from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any, Dict, Generator, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn

from fairscale.utils.state_dict import replace_by_prefix_

if TYPE_CHECKING:
    from collections import OrderedDict  # noqa: F401


class FlattenParamsWrapper(nn.Module):
    """
    A wrapper for transparently flattening a Module's parameters.

    Compared to the original implementation [1], this version:
    - removes tracing
    - supports shared parameters
    - handles state_dict/load_state_dict transparently
    - is renamed to FlattenParamsWrapper

    [1] https://github.com/SsnL/PyTorch-Reparam-Module

    Args:
        module (nn.Module): module to wrap
        param_list (Optional[List[nn.Parameter]]): only flatten parameters
            appearing in the given list (default: flatten all parameters)
    """

    def __init__(self, module: nn.Module, param_list: Optional[List[nn.Parameter]] = None):
        super().__init__()
        self._fpw_module = module
        self.is_flattened = False

        if param_list is not None:
            assert len(param_list) > 0, "param_list can't be empty"
        else:
            param_list = list(module.parameters())
        param_set = set(param_list)

        # convert from list of Parameters to set of (Module, name) tuples, which
        # will survive in case the Parameter instances are reset
        self._param_set = set()
        for m in self.modules():
            for n, p in m.named_parameters(recurse=False):
                if p in param_set:
                    self._param_set.add((m, n))

        self._flatten_params()

        # register the views as plain attributes
        self._unflatten_params_as_views()

        # Register hook to be called after state_dict() to remove the
        # "_fpw_module." prefix and before load_state_dict() to add it back.
        self._register_state_dict_hook(_post_state_dict_hook)
        self._register_load_state_dict_pre_hook(_pre_load_state_dict_hook)

        # Flag to indicate whether state_dict() should automatically unflatten
        # params. This defaults to True, but may be set to False if the user
        # explicitly requests a flat state dict via flat_state_dict().
        self._auto_unflatten_state_dict = True

    @property
    def module(self) -> nn.Module:
        return self._fpw_module

    def _flatten_params(self) -> None:
        assert not self.is_flattened
        self.is_flattened = True

        param_infos = []
        shared_param_memo: Dict[nn.Parameter, Tuple[nn.Module, str]] = {}
        shared_param_infos = []
        params = []
        param_numels = []
        param_shapes = []
        for m in self.modules():
            for n, p in m.named_parameters(recurse=False):
                if p is not None and (m, n) in self._param_set:
                    if p in shared_param_memo:
                        shared_m, shared_n = shared_param_memo[p]
                        shared_param_infos.append((m, n, shared_m, shared_n))
                    else:
                        shared_param_memo[p] = (m, n)
                        param_infos.append((m, n))
                        params.append(p.detach())
                        param_numels.append(p.numel())
                        param_shapes.append(p.size())
        del shared_param_memo

        assert len(set(p.dtype for p in params)) <= 1, "expects all parameters in module to have same dtype"

        # store the info for unflatten
        self._param_infos = tuple(param_infos)
        self._shared_param_infos = tuple(shared_param_infos)
        self._param_numels = tuple(param_numels)
        self._param_shapes = tuple(param_shapes)

        # flatten
        flat_param = nn.Parameter(torch.cat([p.reshape(-1) for p in params], 0))
        self.register_parameter("flat_param", flat_param)
        self.param_numel = flat_param.numel()
        del params

        # deregister the names as parameters
        for m, n in self._param_infos:
            delattr(m, n)
        for m, n, _, _ in self._shared_param_infos:
            delattr(m, n)

    def _get_param_views(self) -> Generator:
        return (t.view(s) for (t, s) in zip(self.flat_param.split(self._param_numels), self._param_shapes))

    def _unflatten_params(self) -> None:
        assert self.is_flattened
        self.is_flattened = False

        ps = self._get_param_views()
        for (m, n), p in zip(self._param_infos, ps):
            if hasattr(m, n):
                delattr(m, n)
            m.register_parameter(n, nn.Parameter(p))
        for (m, n, shared_m, shared_n) in self._shared_param_infos:
            if hasattr(m, n):
                delattr(m, n)
            m.register_parameter(n, getattr(shared_m, shared_n))
        del self.flat_param

    def _unflatten_params_as_views(self) -> None:
        assert self.is_flattened
        ps = self._get_param_views()
        for (m, n), p in zip(self._param_infos, ps):
            setattr(m, n, p)  # This will set as plain attr
        for (m, n, shared_m, shared_n) in self._shared_param_infos:
            setattr(m, n, getattr(shared_m, shared_n))

    @contextmanager
    def unflatten_params(self, recurse: bool = True) -> Generator:
        """
        Unflatten params (optionally recursively on all nested instances).
        If the current instance is already unflattened, then it will remain
        unflattened after the context manager exits.
        """
        if recurse:
            with ExitStack() as stack:
                # unflatten any nested FlattenParamsWrapper instances
                for module in self.modules():
                    if isinstance(module, FlattenParamsWrapper):
                        stack.enter_context(module.unflatten_params(recurse=False))
                # yield to the caller, with unflattened params in all nested instances
                yield
            # exiting from the ExitStack will re-flatten params
            return
        else:
            orig_flattened = self.is_flattened
            if self.is_flattened:
                self._unflatten_params()
            yield
            if orig_flattened:
                self._flatten_params()
                self._unflatten_params_as_views()

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.module, name)  # fallback to wrapped module

    def state_dict(self, *args: Any, **kwargs: Any) -> "OrderedDict[str, Tensor]":  # type: ignore
        """Return the wrapped module's state_dict (unflattened)."""
        if self.is_flattened and self._auto_unflatten_state_dict:
            with self.unflatten_params(recurse=False):
                return super().state_dict(*args, **kwargs)
        else:
            return super().state_dict(*args, **kwargs)

    def flat_state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return the flattened state_dict."""
        assert self.is_flattened
        with ExitStack() as stack:
            # tell any nested FlattenParamsWrapper instances not to auto unflatten
            for module in self.modules():  # includes self
                if isinstance(module, FlattenParamsWrapper):
                    stack.enter_context(module._no_auto_unflatten_state_dict())
            state_dict = self.state_dict(*args, **kwargs)
        return state_dict

    @contextmanager
    def _no_auto_unflatten_state_dict(self) -> Generator:
        backup = self._auto_unflatten_state_dict
        self._auto_unflatten_state_dict = False
        yield
        self._auto_unflatten_state_dict = backup

    def load_state_dict(
        self, state_dict: Union[Dict[str, Tensor], "OrderedDict[str, Tensor]"], strict: bool = True
    ) -> NamedTuple:
        """
        Load a state dict. If necessary, ``unflatten_params`` will be called to
        match the input state_dict.
        """
        # unflatten the module automatically if the state_dict is non-flat
        if self.is_flattened and "flat_param" not in state_dict:
            with self.unflatten_params(recurse=True):
                return super().load_state_dict(state_dict, strict)
        else:
            return super().load_state_dict(state_dict, strict)

    def forward(self, *inputs: Any, **kwinputs: Any) -> Any:
        self._unflatten_params_as_views()
        return self.module(*inputs, **kwinputs)


def _post_state_dict_hook(
    module: nn.Module, state_dict: "OrderedDict[str, Tensor]", prefix: str, *args: Any
) -> "OrderedDict[str, Tensor]":
    replace_by_prefix_(state_dict, prefix + "_fpw_module.", prefix)
    return state_dict


def _pre_load_state_dict_hook(
    state_dict: Union[Dict[str, Tensor], "OrderedDict[str, Tensor]"], prefix: str, *args: Any
) -> None:
    replace_by_prefix_(state_dict, prefix, prefix + "_fpw_module.")
    # flat_param actually needs to move one level up though
    flat_param_key = prefix + "_fpw_module.flat_param"
    if flat_param_key in state_dict:
        replace_by_prefix_(state_dict, flat_param_key, prefix + "flat_param")
