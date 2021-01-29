# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch

"""Useful functions to deal with tensor types with other python container types."""


def apply_to_tensors(fn: Callable, container: Union[torch.Tensor, Dict, List, Tuple, Set]) -> Any:
    """Recursively apply to all tensor in 4 kinds of container types."""

    def _apply(x: Union[torch.Tensor, Dict, List, Tuple, Set]) -> Any:
        if torch.is_tensor(x):
            return fn(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(container)


def pack_kwargs(*args: Any, **kwargs: Any) -> Tuple[List[str], List[Any]]:
    """
    Usage::

        kwarg_keys, flat_args = pack_kwargs(1, 2, a=3, b=4)
        args, kwargs = unpack_kwargs(kwarg_keys, flat_args)
        assert args == [1, 2]
        assert kwargs == {"a": 3, "b": 4}
    """
    kwarg_keys = []
    flat_args = list(args)
    for k, v in kwargs.items():
        kwarg_keys.append(k)
        flat_args.append(v)
    return kwarg_keys, flat_args


def unpack_kwargs(kwarg_keys: List[str], flat_args: List[Any]) -> Tuple[List[Any], Dict[str, Any]]:
    """See pack_kwargs."""
    if len(kwarg_keys) == 0:
        return flat_args, {}
    args = flat_args[: -len(kwarg_keys)]
    kwargs = {k: v for k, v in zip(kwarg_keys, flat_args[-len(kwarg_keys) :])}
    return args, kwargs


def split_non_tensors(
    mixed: Union[torch.Tensor, Tuple[Any]]
) -> Tuple[Tuple[torch.Tensor, ...], Optional[Dict[str, List[Any]]]]:
    """
    Usage::

        x = torch.Tensor([1])
        y = torch.Tensor([2])
        tensors, packed_non_tensors = split_non_tensors((x, y, None, 3))
        recon = unpack_non_tensors(tensors, packed_non_tensors)
        assert recon == (x, y, None, 3)
    """
    if isinstance(mixed, torch.Tensor):
        return (mixed,), None
    tensors: List[torch.Tensor] = []
    packed_non_tensors: Dict[str, List[Any]] = {"is_tensor": [], "objects": []}
    for o in mixed:
        if isinstance(o, torch.Tensor):
            packed_non_tensors["is_tensor"].append(True)
            tensors.append(o)
        else:
            packed_non_tensors["is_tensor"].append(False)
            packed_non_tensors["objects"].append(o)
    return tuple(tensors), packed_non_tensors


def unpack_non_tensors(tensors: Tuple[torch.Tensor], packed_non_tensors: Dict[str, List[Any]]) -> Tuple[Any, ...]:
    """See split_non_tensors."""
    if packed_non_tensors is None:
        return tensors
    assert isinstance(packed_non_tensors, dict)
    mixed = []
    is_tensor_list = packed_non_tensors["is_tensor"]
    objects = packed_non_tensors["objects"]
    assert len(tensors) + len(objects) == len(is_tensor_list)
    obj_i = tnsr_i = 0
    for is_tensor in is_tensor_list:
        if is_tensor:
            mixed.append(tensors[tnsr_i])
            tnsr_i += 1
        else:
            mixed.append(objects[obj_i])
            obj_i += 1
    return tuple(mixed)
