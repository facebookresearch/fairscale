# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from typing import Any, Callable, Dict, Generator, Tuple, Union

import torch.nn as nn

from fairscale.nn.data_parallel.fully_sharded_data_parallel import FullyShardedDataParallel


@contextlib.contextmanager
def enable_wrap(**kwargs: Any) -> Generator[None, None, None]:
    """
    Context manager to wrap modules in FullyShardedDataParallel.

    Useful for when you'd like to apply the same parameters to all child modules
    that you wrap. A particularly important use case is wrapping large layers so
    that they get sharded (in-place) during initialization, to avoid running out of
    system memory. Large layers can indicate that they should be sharded via
    the ``wrap`` annotation and this context manager can provide the
    exact configuration for these nested instances.

    Usage::

        with enable_wrap(**params):
            # Wraps layer in FSDP by default if within context
            self.l1 = wrap(torch.nn.Linear(5, 5))
            # Wraps children modules by default based on min_num_params
            self.l2 = auto_wrap(TransformerBlock(), min_num_params=1e8)

    Args:
        **kwargs: Configuration settings that will be passed to all ``wrap``
            instances inside the context
    """
    with ConfigAutoWrap(**kwargs):
        yield


def wrap(module: nn.Module, cls: Callable = FullyShardedDataParallel, **wrap_overrides: Any) -> nn.Module:
    """
    Annotate that a module should be wrapped.
    Annotated modules will only be wrapped if inside of an ``enable_wrap``
    context manager. An important use case is annotating large layers that
    should be sharded (in-place) during initialization, to avoid running out
    of system memory.

    Usage::

        with enable_wrap(**params):
            # Wraps layer in FSDP by default if within context
            self.l1 = wrap(torch.nn.Linear(5, 5))

    Args:
        module (nn.Module): module to wrap (if in ``enable_wrap`` context)
        cls (Callable): Class wrapper to wrap the model with if in context (Default FullyShardedDataParallel)
        **wrap_overrides: Configuration overrides that will take
            priority over the values provided by the ``enable_wrap`` context
    """
    if ConfigAutoWrap.in_autowrap_context:
        wrap_overrides = {**ConfigAutoWrap.kwargs, **wrap_overrides}
        return cls(module, **wrap_overrides)
    return module


def auto_wrap(
    module: nn.Module, min_num_params: float = 1e8, cls: Callable = FullyShardedDataParallel, **kwargs: Any
) -> nn.Module:
    """
    Annotate a module should be wrapped, and recursively wrap children modules if above min_num_params.
    This is useful when wrapping large complex layer, and automatically wrapping large layers that
    should be re-sharded during runtime when using ``FullyShardedDataParallel``.

    .. warning:: It is not recommended to use ``auto_wrap`` with
        ``FullyShardedDataParallel`` on modules that have shared parameters, as
        the parameter sharing may be broken if they shared parameters are not
        wrapped under the same FSDP wrapper.

    Usage::

        with enable_wrap(**params):
            # Wraps children modules by default based on min_num_params
            self.l1 = auto_wrap(TransformerBlock(), min_num_params=1e8)

    Args:
        min_num_params (int, Optional): min number of parameters for a child
            Module to be wrapped
    """
    if ConfigAutoWrap.in_autowrap_context:
        wrapped_module, remainder = ConfigAutoWrap.recursive_wrap(
            module, cls=cls, min_num_params=min_num_params, **kwargs
        )
        return wrapped_module
    return module


class ConfigAutoWrap:
    """
    Helper class to wrap modules based on default config args via a context manager.
    See ``FullyShardedDataParallel.enable_wrap`` for more information.
    """

    in_autowrap_context = False
    kwargs: Dict[str, Any] = {}

    def __init__(self, **kwargs: Dict[str, Any]):
        self.kwargs = kwargs

    @staticmethod
    def enable_autowrap_context(kwargs: Any) -> None:
        ConfigAutoWrap.in_autowrap_context = True
        ConfigAutoWrap.kwargs = kwargs

    @staticmethod
    def disable_autowrap_context() -> None:
        ConfigAutoWrap.in_autowrap_context = False
        ConfigAutoWrap.kwargs = {}

    def __enter__(self) -> None:
        self.enable_autowrap_context(self.kwargs)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disable_autowrap_context()

    @staticmethod
    def recursive_wrap(
        x: nn.Module, min_num_params: Union[int, float], cls: Callable, **kwargs: Any
    ) -> Tuple[nn.Module, int]:
        """
        Automatically wrap child modules of *x* that meet the given criteria with ``auto_wrap``.

        Args:
            min_num_params (int): min number of parameters for a child Module to be wrapped
        """
        num_params = sum([p.numel() for p in x.parameters()])

        if len(list(x.named_children())) == 0:
            # If the module has no children, no need to recurse, wrap it if needed
            if num_params >= min_num_params:
                return wrap(x, cls=cls, **kwargs), num_params
            return x, 0

        if num_params >= min_num_params:
            total_wrapped_params = 0
            # Iterate through the children, recursively wrap if necessary
            for name, module in x.named_children():
                wrapped_module, num_wrapped_params = ConfigAutoWrap.recursive_wrap(
                    x=module, cls=cls, min_num_params=min_num_params, **kwargs
                )
                setattr(x, name, wrapped_module)
                # Keep track of how many parameters have been wrapped
                total_wrapped_params += num_wrapped_params
            # decide if we need to wrap the current module,
            # since the left over parameters exceed the number of params to wrap
            remainder = num_params - total_wrapped_params
            if remainder >= min_num_params:
                return wrap(x, cls=cls, **kwargs), num_params
            else:
                return x, total_wrapped_params
        return x, 0
