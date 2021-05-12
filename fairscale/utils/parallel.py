# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Useful functions for parallel training."""

from typing import List, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
import torch.nn.functional as F


def chunk_and_pad(tensor: torch.Tensor, num_chunks: int) -> List[torch.Tensor]:
    """Chunk a given Tensor into num_chunks parts and add any necessary padding."""
    chunks = list(torch.flatten(tensor).chunk(num_chunks))
    # torch.chunk may return fewer than num_chunks chunks, pad accordingly.
    num_pad_for_partial_chunk = chunks[0].numel() - chunks[-1].numel()
    if num_pad_for_partial_chunk > 0:
        chunks[-1] = F.pad(chunks[-1], [0, num_pad_for_partial_chunk])
    if len(chunks) < num_chunks:
        chunks.extend([torch.zeros_like(chunks[0]) for _ in range(num_chunks - len(chunks))])
    return chunks


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


def enable_pytorch_sync_bn(module: torch.nn.Module) -> None:
    """Call _specify_ddp_gpu_num for all pytorch SyncBN layers so that it
       is happily running even without DDP. E.g. this is used by FSDP.
    """
    for layer in module.modules():
        if isinstance(layer, torch.nn.modules.SyncBatchNorm) and hasattr(layer, "_specify_ddp_gpu_num"):
            # Number "1" below meant to be the number of GPUs for each DDP worker.
            # (i.e. "device_ids" in DDP. As far as I see, the value is not actually
            # used, but this call needs to be made to avoid an exception.
            # This function is removed from pytorch since 1.9.
            layer._specify_ddp_gpu_num(1)  # type: ignore


def get_process_group_cached(ranks: Optional[List[int]] = None) -> ProcessGroup:
    """
    Singleton PyTorch distributed group cache. Inspired by the code from fairseq.

    For FSDP, it is important to use the same group between outer and inner FSDP instances,
    otherwise, inner FSDP instances will not share the gradient reduction bucket buffer with
    the root instance. This will result in increased GPU memory utilization.

    Each separate process group also uses separate NCCL library instances, which will have
    a significant effect on GPU memory use if too many process groups are created and used.
    Setting NCCL_BUFFSIZE=102400 env variable is a useful technique to check if the NCCL
    memory is causing GPU OOM. Note, the NCCL buffers are not allocated
    through the PyTorch caching allocator, therefore, you may see GPU OOM even when
    torch.cuda.reserved_memory() is still way below the total amount of GPU memory.

    Extra process groups can also reduce training speed (observed on VISSL models).

    Args:
        ranks (Optional[List[int]]):
            Ranks requested in the target group. None for all ranks.
            Default: None

    Returns:
        (ProcessGroup):
            Return the requested process group. Throws RuntimeError if torch.distributed module is not yet initialized.
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not yet initialized but process group is requested.")

    if not hasattr(get_process_group_cached, "_global_group_cache"):
        get_process_group_cached._global_group_cache = {}  # type: ignore
    cache = get_process_group_cached._global_group_cache  # type: ignore

    if ranks is None:
        ranks = list(range(dist.get_world_size()))

    ranks_set = frozenset(ranks)  # take care of ordering and duplicates in the ranks list.
    if ranks_set not in cache:
        cache[ranks_set] = dist.new_group(list(ranks_set))

    return cache[ranks_set]
