# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import torch.nn as nn

from fairscale.nn.data_parallel.fully_sharded_data_parallel import FullyShardedDataParallel
from fairscale.nn.misc import checkpoint_wrapper

# Modules that don't wrap.
FSDP_MODULE_EXCLUDE_WRAP = [nn.ModuleList, nn.ModuleDict]
# Modules that we don't recurse down to their children.
FSDP_MODULE_BLOCKLIST = [nn.MultiheadAttention]


@contextlib.contextmanager
def enable_wrap(module_blocklist: Optional[List] = None, **wrapper_kwargs: Any) -> Generator[None, None, None]:
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
        module_blocklist: List of additional Module Classes to not wrap when
            using :func:`auto_wrap`. This is useful to exclude unsupported
            modules when wrapping recursively.
        **wrapper_kwargs: Configuration settings that will be passed to all ``wrap``
            instances inside the context
    """
    with ConfigAutoWrap(module_blocklist, **wrapper_kwargs):
        yield


def wrap(
    module: nn.Module,
    cls: Callable = FullyShardedDataParallel,
    activation_checkpoint: bool = False,
    **wrap_overrides: Any
) -> nn.Module:
    """
    Annotate that a module should be wrapped. Annotated modules will only be
    wrapped if inside of an :func:`enable_wrap` context manager. An important
    use case is annotating large layers that should be sharded (in-place) during
    initialization, to avoid running out of system memory.

    Usage::

        with enable_wrap(**params):
            # Wraps layer in FSDP by default if within context
            self.l1 = wrap(torch.nn.Linear(5, 5))

    Args:
        module (nn.Module): module to wrap (if in :func:`enable_wrap` context)
        cls (Callable): class wrapper to wrap the model with if in context
            (default: :class:`FullyShardedDataParallel`)
        activation_checkpoint (bool): use activation checkpointing wrapper
            (default: False)
        **wrap_overrides: configuration overrides that will take priority over
            the values provided by the :func:`enable_wrap` context
    """
    if ConfigAutoWrap.in_autowrap_context:
        wrap_overrides = {**ConfigAutoWrap.kwargs, **wrap_overrides}
        if activation_checkpoint:
            module = checkpoint_wrapper(module)
        return cls(module, **wrap_overrides)
    return module


def auto_wrap(
    module: nn.Module,
    min_num_params: int = int(1e8),
    cls: Callable = FullyShardedDataParallel,
    activation_checkpoint: bool = False,
    **kwargs: Any
) -> nn.Module:
    """
    Annotate that a module should be wrapped with *cls* and recursively wrap
    children modules that meet the given criteria. This is useful for wrapping
    large complex layers.

    .. warning:: It is not recommended to use :func:`auto_wrap` with
        :class:`FullyShardedDataParallel` on modules that have shared
        parameters, as the parameter sharing may be broken (i.e. end up not
        shared) if the shared parameters are not (auto-)wrapped under the same
        FSDP wrapper instance.

    Usage::

        with enable_wrap(**params):
            # Wraps children modules by default based on min_num_params
            self.l1 = auto_wrap(TransformerBlock(), min_num_params=1e8)

    Args:
        module (nn.Module): module to wrap (if in :func:`enable_wrap` context)
        cls (Callable): class wrapper to wrap the model with if in context
            (default: :class:`FullyShardedDataParallel`)
        min_num_params (int, Optional): min number of parameters for a child
            Module to be wrapped
        activation_checkpoint (bool): use activation checkpointing wrapper
            (default: False)
    """
    if ConfigAutoWrap.in_autowrap_context:
        wrapped_module, remainder = ConfigAutoWrap.recursive_wrap(
            module, cls=cls, activation_checkpoint=activation_checkpoint, min_num_params=min_num_params, **kwargs
        )
        return wrapped_module
    return module


class ConfigAutoWrap:
    """
    Helper class to wrap modules based on default config args via a context manager.
    See :func:`enable_wrap` for more information.
    """

    module_blocklist: List = []
    in_autowrap_context: bool = False
    kwargs: Dict[str, Any] = {}

    def __init__(self, module_blocklist: Optional[List] = None, **kwargs: Dict[str, Any]):
        if module_blocklist:
            self.module_blocklist += module_blocklist
        self.kwargs = kwargs

    @staticmethod
    def enable_autowrap_context(kwargs: Any) -> None:
        ConfigAutoWrap.in_autowrap_context = True
        ConfigAutoWrap.kwargs = kwargs
        ConfigAutoWrap.module_blocklist += FSDP_MODULE_BLOCKLIST

    @staticmethod
    def disable_autowrap_context() -> None:
        ConfigAutoWrap.in_autowrap_context = False
        ConfigAutoWrap.kwargs = {}
        ConfigAutoWrap.module_blocklist = []

    def __enter__(self) -> None:
        self.enable_autowrap_context(self.kwargs)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disable_autowrap_context()

    @staticmethod
    def recursive_wrap(module: nn.Module, min_num_params: int, **kwargs: Any) -> Tuple[nn.Module, int]:
        """
        Automatically wrap child modules of *module* that meet the given
        criteria with :func:`auto_wrap`.

        Args:
            module (nn.Module): module to recursively wrap
            min_num_params (int): min number of parameters for a child Module to
                be wrapped
        """
        if isinstance(module, tuple(ConfigAutoWrap.module_blocklist)):
            # If the module has been blocklisted from wrapping, we return
            return module, 0

        num_params = sum([p.numel() for p in module.parameters()])

        if len(list(module.named_children())) == 0:
            # If the module has no children, no need to recurse, wrap it if needed
            if num_params >= min_num_params and not isinstance(module, tuple(FSDP_MODULE_EXCLUDE_WRAP)):
                return wrap(module, **kwargs), num_params
            return module, 0

        if num_params >= min_num_params:
            total_wrapped_params = 0
            # Iterate through the children, recursively wrap if necessary
            for name, child in module.named_children():
                wrapped_child, num_wrapped_params = ConfigAutoWrap.recursive_wrap(
                    module=child, min_num_params=min_num_params, **kwargs
                )
                setattr(module, name, wrapped_child)
                # Keep track of how many parameters have been wrapped
                total_wrapped_params += num_wrapped_params
            # decide if we need to wrap the current module,
            # since the left over parameters exceed the number of params to wrap
            remainder = num_params - total_wrapped_params
            if remainder >= min_num_params and not isinstance(module, tuple(FSDP_MODULE_EXCLUDE_WRAP)):
                return wrap(module, **kwargs), num_params
            else:
                return module, total_wrapped_params
        return module, 0
