# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Useful functions for manipulating state_dicts."""

from typing import TYPE_CHECKING, Dict, List, Tuple, Type, Union

from torch import Tensor, nn

if TYPE_CHECKING:
    from collections import OrderedDict  # noqa: F401


def find_module_instances(module: nn.Module, search_class: Type[nn.Module]) -> List[Tuple[str, nn.Module]]:
    """
    Find all occurrences of a given search_class among the given Modules's
    children and return the corresponding paths in the same format as
    state_dicts.

    Usage::

        net = nn.Sequential(
            nn.Linear(1, 1),
            nn.ModuleDict({"ln": nn.LayerNorm(1), "linear": nn.Linear(1, 1)}),
            nn.LayerNorm(1)
        )

        >>> find_module_instances(net, nn.LayerNorm)
        [('1.ln.', LayerNorm((1,), eps=1e-05, elementwise_affine=True)), ('2.', LayerNorm((1,), eps=1e-05, elementwise_affine=True))]
        >>> find_module_instances(net, nn.Dropout)
        []
        >>> find_module_instances(net, nn.Sequential)
        [('', Sequential(
          (0): Linear(in_features=1, out_features=1, bias=True)
          (1): ModuleDict(
            (ln): LayerNorm((1,), eps=1e-05, elementwise_affine=True)
            (linear): Linear(in_features=1, out_features=1, bias=True)
          )
          (2): LayerNorm((1,), eps=1e-05, elementwise_affine=True)
        ))]
    """
    paths = []

    def add_paths_(module: nn.Module, prefix: str = "") -> None:
        if isinstance(module, search_class):
            paths.append((prefix, module))
        for name, child in module.named_children():
            add_paths_(child, prefix + name + ".")

    add_paths_(module)
    return paths


def replace_by_prefix_(
    state_dict: Union[Dict[str, Tensor], "OrderedDict[str, Tensor]"], old_prefix: str, new_prefix: str
) -> None:
    """
    Replace all keys that match a given old_prefix with a new_prefix (in-place).

    Usage::

        state_dict = {"layer.xyz": torch.tensor(1)}
        replace_by_prefix_(state_dict, "layer.", "module.layer.")
        assert state_dict == {"module.layer.xyz": torch.tensor(1)}
    """
    if old_prefix == new_prefix:
        raise ValueError("old_prefix and new_prefix must be distinct")
    for key in list(state_dict.keys()):
        if not key.startswith(old_prefix):
            continue
        new_key = new_prefix + key[len(old_prefix) :]
        state_dict[new_key] = state_dict[key]
        del state_dict[key]
