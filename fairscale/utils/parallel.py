# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

"""Useful functions for parallel training."""


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
