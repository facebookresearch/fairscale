# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

""" Shared functions related to testing GPU memory sizes. """

import gc
from typing import Tuple

import torch


def find_tensor_by_shape(target_shape: Tuple, only_param: bool = True) -> bool:
    """ Find a tensor from the heap

    Args:
        target_shape (tuple):
            Tensor shape to locate.
        only_param (bool):
            Only match Parameter type (e.g. for weights).

    Returns:
        (bool):
            Return True if found.
    """
    for obj in gc.get_objects():
        try:
            # Only need to check parameter type objects if asked.
            if only_param and "torch.nn.parameter.Parameter" not in str(type(obj)):
                continue
            if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                if obj.shape == target_shape:
                    return True
        except Exception as e:
            pass
    return False
