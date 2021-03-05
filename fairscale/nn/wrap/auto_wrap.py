# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from typing import Any, Callable, Dict, Generator, Optional, Set, Tuple, Type, cast

import torch.nn as nn


def default_should_wrap_policy(
    recurse: bool,
    module: nn.Module,
    unwrapped_params: int,
    # These are customizable for this default policy function.
    min_num_params: int = int(1e8),
    force_leaf_modules: Optional[Set[Type[nn.Module]]] = None,
    exclude_wrap_modules: Optional[Set[Type[nn.Module]]] = None,
) -> bool:
    """Default policy function for :func:`auto_wrap`.

       Return if a module should be wrapped during :func:`auto_wrap`.

       The first three parameters are used by :func:`auto_wrap`. If
       you write a custom version of this policy function, your version
       needs to at least accept the first three parameters and free
       to do whatever you want in the function.

    Args:
       recurse (bool):
           Indicate if this is called to make a decision on whether we
           should recurse down a subgraph of the module structure.
           If False, it means this function is called to make a decision
           on whether we should wrap the said module.
       module (nn.Module):
           The module to be considered in this decision.
       unwrapped_params (int):
           The number of parameters yet to be wrapped in this module.

       min_num_params (int):
           Customizable policy input. It controls the size threshold
           on how big should a module be to be considered wrapped.
       force_leaf_modules (Set[Type[nn.Module]]): set of module types to
           keep as leaves, i.e., their children will never be wrapped.
       exclude_wrap_modules (Set[Type[nn.Module]]):
           Customizable set of module types to be excluded in wrapping.
    """
    force_leaf_modules = (
        default_should_wrap_policy.FORCE_LEAF_MODULES  # type: ignore
        if force_leaf_modules is None
        else force_leaf_modules
    )
    exclude_wrap_modules = (
        default_should_wrap_policy.EXCLUDE_WRAP_MODULES  # type: ignore
        if exclude_wrap_modules is None
        else exclude_wrap_modules
    )

    is_large = unwrapped_params >= min_num_params
    if recurse:
        # We should recurse if the module is big enough but not force_leaf_modulesed.
        return is_large and not isinstance(module, tuple(force_leaf_modules))
    else:
        # If we are not recursing, we should wrap but not the exclude list.
        return is_large and not isinstance(module, tuple(exclude_wrap_modules))


# Set those defaults to the default_should_wrap_policy function. Make them easy to be imported.
default_should_wrap_policy.EXCLUDE_WRAP_MODULES = {nn.ModuleList, nn.ModuleDict}  # type: ignore
default_should_wrap_policy.FORCE_LEAF_MODULES = {nn.MultiheadAttention}  # type: ignore


@contextlib.contextmanager
def enable_wrap(should_wrap: Optional[Callable] = None, **wrapper_kwargs: Any) -> Generator[None, None, None]:
    """
    Context manager to wrap modules using a wrapper.

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
            # Wraps children modules based on a different min_num_params
            my_should_wrap = functools.partial(should_wrap, min_num_params=1e7)
            self.l2 = auto_wrap(TransformerBlock(), shuold_wrap=my_should_wrap)

    Args:
        should_wrap (Callable, Optional):
            Custom function to control how to do :func:`auto_wrap`. This is
            useful to exclude unsupported modules or wrap based on sizes when
            wrapping recursively.
            (default: :func:`default_should_wrap_policy`)
        **wrapper_kwargs:
            Configuration settings that will be passed to all ``wrap``
            instances inside the context
    """
    with ConfigAutoWrap(default_should_wrap_policy, **wrapper_kwargs):
        yield


def wrap(module: nn.Module, **wrap_overrides: Any) -> nn.Module:
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
        **wrap_overrides: configuration overrides that will take priority over
            the values provided by the :func:`enable_wrap` context
    """
    if ConfigAutoWrap.in_autowrap_context:
        wrap_overrides = {**ConfigAutoWrap.kwargs, **wrap_overrides}
        assert ConfigAutoWrap.wrapper_cls is not None
        return ConfigAutoWrap.wrapper_cls(module, **wrap_overrides)
    return module


def auto_wrap(module: nn.Module, should_wrap: Optional[Callable] = None, **kwargs: Any) -> nn.Module:
    """
    Annotate that a module should be wrapped with the *wrapper_cls* from the
    :func:`enable_wrap` context (if the context exists) and recursively wrap
    children modules that meet the criteria given by :func:`should_wrap`. This
    is useful for wrapping large complex layers.

    .. warning:: It is not recommended to use :func:`auto_wrap` with
        :class:`FullyShardedDataParallel` on modules that have shared
        parameters, as the parameter sharing may be broken (i.e. end up not
        shared) if the shared parameters are not (auto-)wrapped under the same
        FSDP wrapper instance.

    Usage::

        with enable_wrap(**params):
            # Wraps children modules.
            self.l1 = auto_wrap(TransformerBlock())

    Args:
        module (nn.Module):
            module to wrap (if in :func:`enable_wrap` context)
        should_wrap (Callable):
            a function to determine should Module to be wrapped.
            (default: wrap if > 100M parameters)
    """
    if ConfigAutoWrap.in_autowrap_context:
        wrapped_module, remainder = ConfigAutoWrap.recursive_wrap(module, should_wrap=should_wrap, **kwargs)
        return wrapped_module
    return module


class ConfigAutoWrap:
    """
    Helper class to wrap modules based on default config args via a context manager.
    See :func:`enable_wrap` for more information.
    """

    in_autowrap_context: bool = False  # Context flag
    wrapper_cls: Optional[Callable] = None  # The wrapper class
    kwargs: Dict[str, Any] = {}  # Wrapper's args
    should_wrap: Optional[Callable] = None  # Used only in auto_wrap

    def __init__(self, should_wrap: Optional[Callable] = None, **kwargs: Dict[str, Any]):
        self.should_wrap = should_wrap
        self.kwargs = kwargs

    @staticmethod
    def enable_autowrap_context(should_wrap: Optional[Callable], kwargs: Any) -> None:
        if ConfigAutoWrap.in_autowrap_context:
            raise NotImplementedError(
                "You are already within an autowrap context and we currently do not supported nested autowrap."
            )
        ConfigAutoWrap.in_autowrap_context = True
        # Get and save the wrapper cls for the context.
        assert "wrapper_cls" in kwargs.keys()
        ConfigAutoWrap.wrapper_cls = cast(Callable, kwargs["wrapper_cls"])
        del kwargs["wrapper_cls"]
        # Save the rest.
        ConfigAutoWrap.should_wrap = default_should_wrap_policy if should_wrap is None else should_wrap
        ConfigAutoWrap.kwargs = kwargs

    @staticmethod
    def disable_autowrap_context() -> None:
        ConfigAutoWrap.in_autowrap_context = False
        ConfigAutoWrap.wrapper_cls = None
        ConfigAutoWrap.kwargs = {}
        ConfigAutoWrap.should_wrap = None

    def __enter__(self) -> None:
        self.enable_autowrap_context(self.should_wrap, self.kwargs)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disable_autowrap_context()

    @staticmethod
    def recursive_wrap(module: nn.Module, should_wrap: Optional[Callable], **kwargs: Any) -> Tuple[nn.Module, int]:
        """
        Automatically wrap child modules of *module* that meet the given
        criteria with :func:`auto_wrap`.

        Args:
            module (nn.Module):
                module to recursively wrap
            should_wrap (Callable, Optional):
                optionally, override the :func:`should_wrap` from the context.

        Returns:
            (nn.Module, int):
                Wrapped module and the number parameters wrapped recursively.
        """
        if should_wrap is None:
            should_wrap = ConfigAutoWrap.should_wrap

        # Make sure no child is not already wrapped.
        for _, child in module.named_modules():
            assert not isinstance(child, cast(type, ConfigAutoWrap.wrapper_cls))

        # We count all params, assuming none of them is already wrapped.
        num_params = sum([p.numel() for p in module.parameters()])

        assert should_wrap is not None
        if should_wrap(True, module, num_params):
            total_wrapped_params = 0
            # Iterate through the children, recursively wrap if necessary
            for name, child in module.named_children():
                wrapped_child, num_wrapped_params = ConfigAutoWrap.recursive_wrap(
                    module=child, should_wrap=should_wrap, **kwargs
                )
                setattr(module, name, wrapped_child)
                # Keep track of how many parameters have been wrapped
                total_wrapped_params += num_wrapped_params
            # decide if we need to wrap the current module,
            # since the left over parameters exceed the number of params to wrap
            remainder = num_params - total_wrapped_params
            if should_wrap(False, module, remainder):
                # Leaf node or final wrapping of the remainder both happen here.
                return wrap(module, **kwargs), num_params
            else:
                return module, total_wrapped_params
        return module, 0
