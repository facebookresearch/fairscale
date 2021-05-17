# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from fairscale.nn import FullyShardedDataParallel
from fairscale.utils.testing import in_temporary_directory, skip_if_single_gpu, temp_files_ctx


class SimpleNestedModel(nn.Module):
    def __init__(self, embedding_size: int, with_fsdp: bool, process_group):
        super().__init__()
        self.fc1: nn.Module = nn.Linear(embedding_size, 2 * embedding_size)
        self.fc2: nn.Module = nn.Linear(2 * embedding_size, 2 * embedding_size)
        self.fc3: nn.Module = nn.Linear(2 * embedding_size, embedding_size + 1)
        self.fc4: nn.Module = nn.Linear(embedding_size + 1, embedding_size)
        if with_fsdp:
            self.fc1 = FullyShardedDataParallel(self.fc1, process_group=process_group)
            self.fc3 = FullyShardedDataParallel(self.fc3, process_group=process_group, flatten_parameters=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


def _create_model(embedding_size: int, with_fsdp: bool, process_group):
    model = SimpleNestedModel(with_fsdp=with_fsdp, process_group=process_group, embedding_size=embedding_size).cuda()
    if with_fsdp:
        return FullyShardedDataParallel(model, process_group=process_group)
    else:
        return model


def _load_sharded_checkpoint(rank: int):
    return torch.load(f"checkpoint_{rank}.torch")  # type: ignore


def _worker(gpu_id: int, sync_file: str, world_size: int, embedding_size: int, flatten_parameters: bool):
    torch.manual_seed(0)
    torch.cuda.set_device(gpu_id)
    torch.distributed.init_process_group(
        backend="nccl", init_method=f"file://{sync_file}", world_size=world_size, rank=gpu_id,
    )
    process_group = torch.distributed.new_group()

    # Create a dummy model with dummy inputs and targets
    input = torch.randn(size=(16, embedding_size)).cuda()
    target = torch.zeros(size=(16, embedding_size)).cuda()
    model = _create_model(with_fsdp=True, process_group=process_group, embedding_size=embedding_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    # Train the model for a few epochs
    for epoch in range(2):
        out = model(input)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save a bunch of checkpoint, one by shard
    cp_data = {
        "weights": {k: v.cpu() for k, v in model.local_state_dict().items()},
        "meta": model.local_metadata_dict(),
    }
    torch.save(cp_data, f"checkpoint_{gpu_id}.torch")

    # Wait for all files to be written on the disk
    dist.barrier()  # type: ignore

    # Reconstruct a full checkpoint from the sharded checkpoints
    all_checkpoints = [_load_sharded_checkpoint(rank) for rank in range(world_size)]
    consolidated_checkpoint = FullyShardedDataParallel.consolidate_shard_weights(
        shard_weights=[c["weights"] for c in all_checkpoints], shard_metadata=[c["meta"] for c in all_checkpoints],
    )

    # Check that the reconstructed parameters are correct and of the right shape
    full_model = _create_model(with_fsdp=False, process_group=process_group, embedding_size=embedding_size)
    full_model_state_dict = full_model.state_dict()
    assert set(full_model_state_dict.keys()) == set(consolidated_checkpoint.keys())
    for k in full_model_state_dict.keys():
        assert consolidated_checkpoint[k].shape == full_model_state_dict[k].shape

    # Verify that the checkpoint can be loaded by a FSDP model
    loaded_model = _create_model(with_fsdp=True, process_group=process_group, embedding_size=embedding_size)
    loaded_model.load_state_dict(consolidated_checkpoint)
    for m in loaded_model.modules():
        if isinstance(m, FullyShardedDataParallel):
            m._reset_lazy_init()

    # Verify that the model saved and the model loaded give the same results
    with torch.no_grad():
        before_checkpoint_loss = criterion(model(input), target).item()
        after_checkpoint_loss = criterion(loaded_model(input), target).item()
        assert before_checkpoint_loss == after_checkpoint_loss


@skip_if_single_gpu
@pytest.mark.parametrize("embedding_size", [128, 129])
@pytest.mark.parametrize("flatten_parameters", [True, False])
def test_consolidation(embedding_size: int, flatten_parameters: bool):
    import torch.multiprocessing as mp

    world_size = 2
    with in_temporary_directory():
        with temp_files_ctx(num=1) as temp_files:
            mp.spawn(_worker, (temp_files[0], world_size, embedding_size, flatten_parameters), nprocs=world_size)
