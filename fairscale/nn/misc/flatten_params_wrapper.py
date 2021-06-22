# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Tongzhou Wang
# Licensed under the MIT License.

from contextlib import contextmanager
from itertools import chain
import typing
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn

from fairscale.utils.state_dict import replace_by_prefix_

if TYPE_CHECKING:
    from collections import OrderedDict  # noqa: F401


class FlatParameter(nn.Parameter):
    """ A parameter that is initialized from a list of parameters and can be
        turned into a list of views as needed.
    """

    def __new__(cls, params: Sequence[nn.Parameter], requires_grad: bool = True) -> "FlatParameter":
        """ Make an object using the parent's __new__ function. """

        # A empty of non-list input doesn't make sense.
        if not isinstance(params, (list, tuple)) or len(params) == 0:
            raise ValueError("An non-empty list or tuple argument is needed")

        # Normally, all items are Parameters. But during pickling, we will have a single
        # Tensor as the input and later in __init__, the correct _param_numels and _param_shapes
        # are set.
        if not all(isinstance(p, (nn.Parameter, Tensor)) for p in params):
            raise ValueError("List items need to be Parameter types")

        # Flattening involves (1) making a tensor flat (i.e. single dimensional) and (2) making a module
        # heirarchy flat (using a single tensor to replace a tree of tensors). Therefore,
        # adding back nesting and heirarchy is counter-productive. If nesting is encountered
        # in the future, the reasonable thing to do is likely for the top level FlatParameter to
        # absorb the nested one and keep the result flat, free from hierarchy.
        if any(isinstance(p, FlatParameter) for p in params):
            raise ValueError("Nesting FlatParameter is not supported")

        data = torch.cat([p.detach().reshape(-1) if isinstance(p, nn.Parameter) else p.reshape(-1) for p in params], 0)
        return super(FlatParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __init__(self, params: Sequence[nn.Parameter], requires_grad: bool = True):
        """ Initialize the _param_numels and _param_shapes lists. """
        self._param_numels = [p.numel() for p in params]
        assert self.numel() <= sum(
            self._param_numels
        ), f"Something wrong with __new__ method, {self.numel()} vs. {sum(self._param_numels)}"
        self._param_shapes = [p.size() for p in params]

    def get_param_views(self, external_data: Optional[Tensor] = None) -> Generator[Tensor, None, None]:
        """ Return a generator of views that map to the original parameters. """
        # Note, self.data could be sharded, so its numel is <= to the sum.
        assert self.data.numel() <= sum(
            self._param_numels
        ), f"Incorrect internal state {self.data.numel()} vs. {sum(self._param_numels)}"
        data = external_data if external_data is not None else self
        if data.numel() != sum(self._param_numels):
            raise ValueError(
                f"Incorrect numel of supplied data: got {data.numel()} but expected {sum(self._param_numels)}"
            )
        return (t.view(s) for (t, s) in zip(data.split(self._param_numels), self._param_shapes))

    def __setstate__(self, state: Tuple[Any, Any]) -> None:
        """ Use by pickle to set the internal states. """
        self._param_numels, self._param_shapes = state
        assert self.numel() <= sum(
            self._param_numels
        ), f"Incorrect pickling {self.numel()} vs. {sum(self._param_numels)}"

    def __reduce_ex__(self, proto: int) -> Tuple[Any, Any, Any]:
        """ Support pickling between ranks. """
        return (
            FlatParameter,  # Callable
            ([self.data], self.requires_grad),  # Args to the callable above
            (self._param_numels, self._param_shapes),  # Args to __setstate__
        )


class FlattenParamsWrapper(nn.Module):
    """
    A wrapper for transparently flattening a Module's parameters.

    Compared to the original implementation [1], this version:
    - removes tracing
    - supports shared parameters
    - handles state_dict/load_state_dict transparently
    - is renamed to FlattenParamsWrapper
    - refactored to use the FlatParameter class

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

        # Since the parameters will be deleted, let's record the number original parameters
        # managed by this class.
        self.num_params_managed = len(param_set)

        # convert from list of Parameters to set of (Module, name) tuples, which
        # will survive in case the Parameter instances are reset
        self._param_set = set()
        for m in self.modules():
            for n, p in m.named_parameters(recurse=False):
                if p in param_set:
                    self._param_set.add((m, n))

        # TODO (Min): double check we handle the special case of module without any params.
        self._flatten_params()

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

    def _init_flatten_params(self) -> List[nn.Parameter]:
        """ Build metadata for need-to-be-flatten parameters and returns a list
            contains the need-to-be-flatten parameters.
        """
        param_infos = []
        param_full_infos = []
        shared_param_memo: Dict[nn.Parameter, Tuple[nn.Module, str]] = {}
        shared_param_infos = []
        params = []
        for module_name, m in self.named_modules():
            for n, p in m.named_parameters(recurse=False):
                if p is not None and (m, n) in self._param_set:
                    if p in shared_param_memo:
                        shared_m, shared_n = shared_param_memo[p]
                        shared_param_infos.append((m, n, shared_m, shared_n))
                    else:
                        shared_param_memo[p] = (m, n)
                        param_infos.append((m, n))
                        param_full_infos.append((module_name, n))
                        params.append(p)
        del shared_param_memo

        assert len(set(p.dtype for p in params)) <= 1, "expects all parameters in module to have same dtype"

        # store the info for unflatten
        self._param_infos = tuple(param_infos)
        self._param_full_infos = tuple(param_full_infos)
        self._shared_param_infos = tuple(shared_param_infos)

        return params

    def _flatten_params(self, flat_param: Optional[nn.Parameter] = None) -> None:
        assert not self.is_flattened
        self.is_flattened = True

        if not hasattr(self, "_param_infos"):
            assert flat_param is None
            params = self._init_flatten_params()

            assert (
                len(set(p.requires_grad for p in params)) == 1
            ), "expects all parameters in module to have same requires_grad"

            flat_param = FlatParameter(params, params[0].requires_grad)
            self.param_numel = flat_param.numel()

        # flatten
        assert flat_param is not None
        self.register_parameter("flat_param", flat_param)

        # deregister the names as parameters
        for m, n in self._param_infos:
            delattr(m, n)
        for m, n, _, _ in self._shared_param_infos:
            delattr(m, n)

        # register the views as plain attributes
        self._unflatten_params_as_views()

    def _unflatten_params(self, external_data: Optional[Tensor] = None) -> None:
        """ Undo flattening and create separate parameters from the already flattened
            self.flat_param or a user supplied external data.
        """
        assert self.is_flattened or external_data is not None
        self.is_flattened = False

        ps = self.get_param_views([external_data])
        for (m, n), p in zip(self._param_infos, ps):
            if hasattr(m, n):
                delattr(m, n)
            m.register_parameter(n, nn.Parameter(p))
        for (m, n, shared_m, shared_n) in self._shared_param_infos:
            if hasattr(m, n):
                delattr(m, n)
            m.register_parameter(n, getattr(shared_m, shared_n))
        if hasattr(self, "flat_param"):
            del self.flat_param

    def _unflatten_params_as_views(self) -> None:
        """ Unlike ``_unflatten_params``, this function unflatten into views and keep
            self.flat_param unchanged.
        """
        assert self.is_flattened
        ps = self.get_param_views()
        for (m, n), p in zip(self._param_infos, ps):
            setattr(m, n, p)  # This will set as plain attr
        for (m, n, shared_m, shared_n) in self._shared_param_infos:
            setattr(m, n, getattr(shared_m, shared_n))

    @contextmanager
    def unflatten_params(self, flat_param: Optional[Tensor] = None) -> Generator:
        """
        Unflatten params. If the current instance is already unflattened, then
        it will remain unflattened after the context manager exits.

        Args:
            flat_param (Tensor, Optional): flat param to use for unflattening.
                If provided, the current instance must be in a flattened state
                at the start of the context manager. The provided Tensor must be
                appropriately sized and will only be used within the context
                manager. After the context manager exits, we will revert to
                using ``self.flat_param`` (default: None).
        """
        assert (
            flat_param is None or self.is_flattened
        ), "Unflattening with external flat_param requires current instance to be flattened"

        orig_flattened = self.is_flattened
        if orig_flattened:
            orig_flat_param = self.flat_param
            self._unflatten_params(flat_param)

        # Put yield in a try...finally in case the caller catches the exception and handles
        # it. In that case, we need to properly handle the undoing of state.
        try:
            yield
        finally:
            if orig_flattened:
                self._flatten_params(orig_flat_param)

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.module, name)  # fallback to wrapped module

    @typing.overload
    def state_dict(
        self, destination: Mapping[str, Tensor], prefix: str = ..., keep_vars: bool = ...
    ) -> Mapping[str, Tensor]:
        ...

    @typing.overload
    def state_dict(self, prefix: str = ..., keep_vars: bool = ...) -> "OrderedDict[str, Tensor]":
        ...

    # Since we have overloads above, we can use Any here.
    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        """Return the wrapped module's state_dict."""
        if self.is_flattened and self._auto_unflatten_state_dict:
            # Returns the original version.
            with self.unflatten_params():
                return super().state_dict(*args, **kwargs)
        else:
            # Returns flattened version.
            return super().state_dict(*args, **kwargs)

    def flat_state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return the flattened state_dict."""
        assert self.is_flattened
        with self._no_auto_unflatten_state_dict():
            return self.state_dict(*args, **kwargs)

    @contextmanager
    def _no_auto_unflatten_state_dict(self) -> Generator:
        backup = self._auto_unflatten_state_dict
        self._auto_unflatten_state_dict = False
        # Put yield in a try...finally in case the caller catches the exception and handles
        # it. In that case, we need to properly handle the undoing of state.
        try:
            yield
        finally:
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
            # This object is flatten but state_dict is not. So we unflatten and load.
            with self.unflatten_params():
                return super().load_state_dict(state_dict, strict)
        else:
            # Otherwise, load as it.
            return super().load_state_dict(state_dict, strict)

    def forward(self, *inputs: Any, **kwinputs: Any) -> Any:
        self._unflatten_params_as_views()
        return self.module(*inputs, **kwinputs)

    def get_param_views(
        self, external_data_list: Optional[List[Optional[Tensor]]] = None
    ) -> Generator[Tensor, None, None]:
        """ Used to get a generator over all views from a list of external data list. """
        params = [self.flat_param]  # For now, there is only a single flat param.
        if external_data_list is None:
            external_data_list = [None] * len(params)

        gens = []
        for p, data in zip(params, external_data_list):
            gens.append(p.get_param_views(data))

        return chain(*gens)  # type: ignore


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
