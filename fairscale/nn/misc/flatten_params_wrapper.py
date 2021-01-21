# Copyright (c) Tongzhou Wang
# Licensed under the MIT License.

from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn


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
        self.module = module

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

    def _flatten_params(self) -> None:
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
        ps = self._get_param_views()
        for (m, n), p in zip(self._param_infos, ps):
            setattr(m, n, p)  # This will set as plain attr
        for (m, n, shared_m, shared_n) in self._shared_param_infos:
            setattr(m, n, getattr(shared_m, shared_n))

    @contextmanager
    def unflatten_params(self) -> Generator:
        self._unflatten_params()
        yield
        self._flatten_params()
        self._unflatten_params_as_views()

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.module, name)  # fallback to wrapped module

    def state_dict(self, prefix: str = "", keep_vars: bool = False) -> "OrderedDict[str, Tensor]":  # type: ignore
        """Return an unflattened state_dict."""
        with self.unflatten_params():
            return self.module.state_dict(prefix=prefix, keep_vars=keep_vars)

    def flat_state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return the flattened state_dict."""
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        if "flat_param" in state_dict:
            super().load_state_dict(state_dict, strict=True)
        else:
            with self.unflatten_params():
                return self.module.load_state_dict(state_dict, *args, **kwargs)

    def forward(self, *inputs: Any, **kwinputs: Any) -> Any:
        self._unflatten_params_as_views()
        return self.module(*inputs, **kwinputs)
