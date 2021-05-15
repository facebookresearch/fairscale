# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from ... import Tensor, device, dtype
from .. import Parameter
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, Generic, NamedTuple
from collections import OrderedDict
from ...utils.hooks import RemovableHandle

_grad_t = Union[Tuple[Tensor, ...], Tensor]
# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.
T = TypeVar('T')
# We parameter modules by the return type of its `forward` (and therefore `__call__`) method. This allows
# type inference to infer that the return value of calling a module in the canonical way (via `__call__)` is the
# same as the custom `forward` function of the submodule. Submodules tha wish to opt in this functionality be 
# defined as eg class ReturnsTwoTensors(Module[Tuple[Tensor, Tensor]]): ...
T_co = TypeVar('T_co', covariant=True)


class Module(Generic[T_co]):
    def __init__(self) -> None: ...

    def forward(self, *input: Any, **kwargs: Any) -> T_co: ...  # type: ignore

    def __call__(self, *input: Any, **kwargs: Any) -> T_co: ...  # type: ignore

    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None: ...

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None: ...

    def add_module(self, name: str, module: 'Module') -> None: ...

    def apply(self: T, fn: Callable[['Module'], None]) -> T: ...

    def cuda(self: T, device: Optional[Union[int, str, device]] = ...) -> T: ...

    def cpu(self: T) -> T: ...

    def type(self: T, dst_type: Union[dtype, str]) -> T: ...

    def float(self: T) -> T: ...

    def double(self: T) -> T: ...

    def half(self: T) -> T: ...

    @overload
    def to(self: T, device: Optional[Union[int, device]] = ..., dtype: Optional[Union[dtype, str]] = ...,
           non_blocking: bool = ...) -> T: ...

    @overload
    def to(self: T, dtype: Union[dtype, str], non_blocking: bool = ...) -> T: ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T: ...

    def register_backward_hook(self, hook: Callable[
        ['Module', _grad_t, _grad_t], Union[None, Tensor]]) -> RemovableHandle: ...

    # The hook takes a module as a first argument and variadic arguments after that, but there is no way to express that
    def register_forward_pre_hook(self, hook: Callable[..., None]) -> RemovableHandle: ...

    def register_forward_hook(self, hook: Callable[..., None]) -> RemovableHandle: ...

    def __getattr__(self, name: str) -> Union[Tensor, 'Module']: ...

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None: ...

    def __setstate__(self, state: Dict[str, Any]) -> None: ...

    # The user can pass an optional arbitrary mappable object to `state_dict`, in which case `state_dict` returns
    # back that same object. But if they pass nothing, an `OrederedDict` is created and returned.
    @overload
    def state_dict(self, destination: Mapping[str, Tensor], prefix: str = ..., keep_vars: bool = ...) -> Mapping[str, Tensor]: ...

    @overload
    def state_dict(self, prefix: str = ..., keep_vars: bool = ...) -> OrderedDict[str, Tensor]: ...

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], OrderedDict[str, Tensor]], strict: bool = ...) -> NamedTuple: ...

    def parameters(self, recurse: bool = ...) -> Iterator[Parameter]: ...

    def named_parameters(self, prefix: str = ..., recurse: bool = ...) -> Iterator[Tuple[str, Parameter]]: ...

    def buffers(self, recurse: bool = ...) -> Iterator[Tensor]: ...

    def named_buffers(self, prefix: str = ..., recurse: bool = ...) -> Iterator[Tuple[str, Tensor]]: ...

    def children(self) -> Iterator['Module']: ...

    def named_children(self) -> Iterator[Tuple[str, 'Module']]: ...

    def modules(self) -> Iterator['Module']: ...

    def named_modules(self, memo: Optional[Set['Module']] = ..., prefix: str = ...) -> Iterator[
        Tuple[str, 'Module']]: ...

    def train(self: T, mode: bool = ...) -> T: ...

    def eval(self: T) -> T: ...

    def zero_grad(self) -> None: ...

    def share_memory(self: T) -> T: ...

    def extra_repr(self) -> str: ...

    # This is added by checkpoint_wrapper
    _checkpoint_fwd_counter: int

    # This is added torchgpipe
    training: bool

    # Added by auto_wrap.py.
    wrapper_config: dict
