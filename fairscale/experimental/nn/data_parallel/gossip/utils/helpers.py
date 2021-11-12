# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Collection of commonly used utility functions
"""

import collections
import logging
import sys
from typing import Any, Dict, List, MutableMapping, Set, Tuple

import torch
import torch.distributed as dist


def flatten_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually
    Args:
        tensors (Iterable[Tensor]): dense tensors to flatten
    Returns:
        A 1D buffer containing input tensors
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat: torch.Tensor, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Args:
        flat (Tensor): flattened dense tensors to unflatten
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return outputs


def group_by_dtype(tensors: List[torch.Tensor]) -> Dict[torch.dtype, List[torch.Tensor]]:
    """
    Returns a dict mapping from the tensor dtype to a list containing all
    tensors of that dtype.
    Arg:
        tensors (Iterable[Tensor]): list of tensors
    """
    tensors_by_dtype = collections.defaultdict(list)
    for tensor in tensors:
        tensors_by_dtype[tensor.dtype].append(tensor)
    return tensors_by_dtype


def communicate(tensors: List[torch.Tensor], communication_op: Any, logger: logging.Logger = None) -> None:
    """
    Communicate a list of tensors
    Args:
        tensors (Iterable[Tensor]): list of tensors
        communication_op: a method or partial object which takes a tensor as
            input and communicates it. It can be a partial object around
            something like torch.distributed.all_reduce
    """
    tensors_by_dtype = group_by_dtype(tensors)
    for tensors_with_same_dtype in tensors_by_dtype.values():
        flat_tensor = flatten_tensors(tensors_with_same_dtype)
        if logger is not None:
            logger.debug("Flatten completed")
        communication_op(tensor=flat_tensor)
        if logger is not None:
            logger.debug("Commmunication completed")
        with torch.no_grad():
            for f, t in zip(
                unflatten_tensors(flat_tensor, tensors_with_same_dtype),
                tensors_with_same_dtype,
            ):
                t.copy_(f)
        if logger is not None:
            logger.debug("Unflatten completed")


HANDLER_AND_LEVEL_SET: Set[logging.Logger] = set()

# TODO: deprecate this function
def make_logger(rank: int, verbose: bool = True) -> logging.Logger:
    """
    Return a logger for writing to stdout
    Args:
        rank (int): rank of node making logger
        verbose (bool): whether to set log-level to INFO; o.w. WARNING
    Returns:
        Python logger
    """
    logger = logging.getLogger(__name__)
    if logger not in HANDLER_AND_LEVEL_SET:
        # if not getattr(logger, "handler_and_level_set", None):
        console = logging.StreamHandler(stream=sys.stdout)
        format_str = "{}".format(rank)
        format_str += ": %(levelname)s -- %(threadName)s -- %(message)s"
        console.setFormatter(logging.Formatter(format_str))
        logger.addHandler(console)  # prints to console
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        HANDLER_AND_LEVEL_SET.add(logger)
        # logger.handler_and_level_set = True
    return logger


def create_process_group(ranks: List[int]) -> torch.distributed.ProcessGroup:
    """
    Creates and intializes a new process group. Assumes init_process_group
    has already been called
    Arguments:
        ranks (list<int>): ranks corresponding to the processes which should
            belong the created process group
    Returns:
        New process group
    """
    new_group = dist.new_group(ranks=ranks)
    init_tensor_fp32, init_tensor_fp16 = torch.zeros(1), torch.zeros(1).half()

    for init_tensor in [init_tensor_fp32, init_tensor_fp16]:
        if torch.cuda.is_available():
            init_tensor = init_tensor.cuda()
        if dist.get_rank() in ranks:
            dist.all_reduce(init_tensor, group=new_group)
        torch.cuda.synchronize()
    return new_group


class MultiProcessAdapter(logging.LoggerAdapter):
    """
    Creates an adapter to make logging for multiple processes cleaner
    """

    def process(self, msg: str, kwargs: Any) -> Tuple[str, MutableMapping[str, Any]]:
        # use process_num from kwargs or the default given on instantiation
        process_num = kwargs.pop("process_num", self.extra["process_num"])
        return f"process: {process_num} {msg}", kwargs
