# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Useful functions for parallel training."""

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


def compute_shard_size(numel: int, world_size: int) -> int:
    """Compute shard size like the behavior of torch.chunk()."""
    assert numel > 0 and world_size > 0, "invalid inputs"
    if numel % world_size == 0:
        # easy case, including world_size == 1.
        shard_size = numel // world_size
    else:
        if world_size == 2:
            # two shards, shard size is the size of the bigger one.
            shard_size = numel // world_size + 1
        else:
            # find the equal chunks until reminder is smaller than shard_size
            for div in range(world_size - 1, 1, -1):
                shard_size, rem = divmod(numel, div)
                if shard_size >= rem:
                    break
            # corner case: bunch of 1 elements and rest are 0s.
            if shard_size == 0:
                shard_size = 1
    assert shard_size > 0, f"bug: {shard_size}"
    return shard_size


def validate_process_group(device: torch.device, process_group: ProcessGroup) -> None:
    """Do a quick test in case user called FSDP without calling torch.cuda.set_device()
       correctly. This can easily happen in cpu_offload case where the model resides on
       the CPU.
    """
    if not hasattr(process_group, "allgather"):
        # Likely a dummy pg for unit test, skip checking.
        return

    world_size = process_group.size()
    if "cuda" in str(device):
        input_tensor = torch.ones(1).to(device)
        output = list(torch.zeros(world_size).to(device).chunk(world_size))
        dist.all_gather(output, input_tensor, group=process_group)
        assert torch.cat(output).sum() == float(world_size), (
            f"found {torch.cat(output).sum()} devices in process group but "
            f"world_size={world_size}. Check torch.cuda.set_device is called properly"
        )
