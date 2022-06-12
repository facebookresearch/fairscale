# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test utility classes from state_dict.py. """

import torch
from torch import nn

from fairscale.internal.state_dict import find_module_instances, replace_by_prefix_


def test_find_module_instances():
    net = nn.Sequential(
        nn.Linear(1, 1), nn.ModuleDict({"ln": nn.LayerNorm(1), "linear": nn.Linear(1, 1)}), nn.LayerNorm(1)
    )
    assert find_module_instances(net, nn.LayerNorm) == [("1.ln.", net[1]["ln"]), ("2.", net[2])]
    assert find_module_instances(net, nn.Linear) == [("0.", net[0]), ("1.linear.", net[1]["linear"])]
    assert find_module_instances(net, nn.Dropout) == []
    assert find_module_instances(net, nn.Sequential) == [("", net)]


def test_replace_by_prefix():
    state_dict = {"layer.a": torch.tensor(1), "abc.layer.def": torch.tensor(2), "layer.b": torch.tensor(3)}
    replace_by_prefix_(state_dict, "layer.", "module.layer.")
    assert state_dict == {
        "module.layer.a": torch.tensor(1),
        "abc.layer.def": torch.tensor(2),
        "module.layer.b": torch.tensor(3),
    }
