# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import ProcessGroup


class Bucket:
    def __init__(self, data: Tensor, group: ProcessGroup):
        self.data = data
        self.group = group
        self.offset = 0
        self.callbacks: List[Callable] = []
        self.output_shard = torch.zeros_like(data[0])

    def flush(self) -> None:
        if self.offset == 0:
            assert len(self.callbacks) == 0
            return
        # reduce-scatter bucket
        dist.reduce_scatter(
            self.output_shard[: self.offset], list(self.data[:, : self.offset].unbind(0)), group=self.group
        )
        # execute post-reduction callbacks
        for callback_fn in self.callbacks:
            callback_fn()
        # reuse input bucket but allocate a fresh output shard
        self.data[:, : self.offset].zero_()
        self.offset = 0
        self.callbacks.clear()
        self.output_shard = torch.zeros_like(self.data[0])


class ReduceScatterBucketer:
    """
    Helper for bucketing multiple reduce-scatter operations on small tensors
    into larger reduce-scatter ops to improve communication efficiency.

    Usage::

        bucketer = ReduceScatterBucketer()
        bucketer.reduce_scatter_async(
            small_tensors, callback_fn=lambda result: print("small")
        )
        bucketer.reduce_scatter_async(
            big_tensors, callback_fn=lambda result: print("big")
        )
        bucketer.reduce_scatter_async(
            more_small_tensors, callback_fn=lambda result: print("small2")
        )
        bucketer.flush()  # callbacks only guaranteed to be called after flush()
        # Example output (note that it is out of order, due to bucketing):
        # big
        # small
        # small2

    Args:
        bucket_cap_mb (int, Optional): bucket size for communicating. Buckets
            are sub-divided based on world_size. Values <= 0 disable bucketing.
    """

    def __init__(self, bucket_cap_mb: int = 25):
        self.bucket_cap_mb = bucket_cap_mb
        self.buckets: Dict[Tuple[torch.dtype, torch.device, ProcessGroup], Bucket] = {}

    @torch.no_grad()
    def reduce_scatter_async(
        self, input_list: List[Tensor], group: ProcessGroup, callback_fn: Optional[Callable] = None,
    ) -> None:
        """
        Reduce-scatter a list of tensors asynchronously, so smaller reductions
        can be bucketed together. The given callback (``callback_fn``) will be
        called with the reduced result at some later time. Call ``flush()`` to
        force all queued ops and callbacks to be executed.

        Note that large inputs will be reduced immediately, and this function
        may also flush the relevant bucket to make room for ``input_list``.

        Args:
            input_list (List[Tensor]): list of tensors to reduce-scatter. List
                should contain ``group.size()`` tensors and each tensor should
                have identical shape, dtype and device.
            group (ProcessGroup): process group for reduction
            callback_fn (Callable, Optional): callback function to call after
                the reduction executes. Function will be called with a single
                argument corresponding to the reduced result.
        """
        world_size = group.size()

        assert (
            len(input_list) == world_size
        ), f"reduce_scatter received {len(input_list)} inputs, expected group.size() ({world_size})"

        first_input = input_list[0]
        first_input_size = first_input.numel()

        bucket_shard_size = self._get_shard_size(first_input.element_size(), world_size)
        if first_input_size > bucket_shard_size:
            # input is too big to fit in the bucket, reduce-scatter directly
            output = torch.zeros_like(input_list[0])
            dist.reduce_scatter(output, input_list, group=group)
            if callback_fn is not None:
                callback_fn(output)
            return

        bucket = self._get_bucket(first_input, group)
        if first_input_size > bucket.data.size(1) - bucket.offset:
            # not enough space remaining in bucket, flush it now
            bucket.flush()

        # copy data from input_list into bucket
        stacked_input = torch.stack(input_list).view(world_size, first_input_size)
        offset = bucket.offset
        bucket.data[:, offset : offset + first_input_size].copy_(stacked_input)
        bucket.offset += first_input_size

        # callback will be given the reduced result
        if callback_fn is not None:
            result_view = bucket.output_shard[offset : offset + first_input_size].view_as(first_input)
            bucket.callbacks.append(functools.partial(callback_fn, result_view))

    @torch.no_grad()
    def flush(self) -> None:
        """Reduce-scatter any partial buckets."""
        for bucket in self.buckets.values():
            bucket.flush()

    @functools.lru_cache()
    def _get_shard_size(self, element_size: int, num_shards: int) -> int:
        if self.bucket_cap_mb <= 0:  # Values <= 0 disable bucketing.
            return 0
        MB = 1024 * 1024
        bucket_size = self.bucket_cap_mb * MB / element_size
        return int(bucket_size // num_shards)

    def _get_bucket(self, tensor: Tensor, group: ProcessGroup) -> Bucket:
        key = (tensor.dtype, tensor.device, group)
        if key not in self.buckets:
            # buckets are divided into world_size pieces, bucket.data shaped (world_size, shard_size)
            world_size = group.size()
            shard_size = self._get_shard_size(tensor.element_size(), world_size)
            data = tensor.new_zeros((world_size, shard_size))
            self.buckets[key] = Bucket(data, group)
        return self.buckets[key]
