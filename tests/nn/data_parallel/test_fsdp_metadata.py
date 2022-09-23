# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import functools
import os
import tempfile

from parameterized import parameterized
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.optim import Adam

from fairscale.fair_dev.testing.testing import in_temporary_directory, skip_if_single_gpu, temp_files_ctx
from fairscale.nn import FullyShardedDataParallel
from tests.nn.data_parallel.test_fsdp import DistributedTest, MixtureOfExperts, rename_test, spawn_and_init

USE_TEMPFILE = True  # False for debugging


class ConvolutionalModel(nn.Module):
    def __init__(self, embedding_size: int, with_fsdp: bool, process_group):
        super().__init__()
        self.conv1 = self._conv_block(3, embedding_size)
        self.conv2: nn.Module = self._conv_block(embedding_size, embedding_size // 2)
        self.conv3: nn.Module = self._conv_block(embedding_size // 2, embedding_size)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.relu = nn.ReLU()
        self.fc1: nn.Module = nn.Linear(embedding_size, 2 * embedding_size)
        self.fc2: nn.Module = nn.Linear(2 * embedding_size, 2 * embedding_size)
        self.fc3: nn.Module = nn.Linear(2 * embedding_size, embedding_size + 1)
        self.fc4: nn.Module = nn.Linear(embedding_size + 1, embedding_size)
        if with_fsdp:
            self.conv2 = FullyShardedDataParallel(self.conv2, process_group=process_group)
            self.conv3 = FullyShardedDataParallel(self.conv3, process_group=process_group, flatten_parameters=False)
            self.fc1 = FullyShardedDataParallel(self.fc1, process_group=process_group)
            self.fc3 = FullyShardedDataParallel(self.fc3, process_group=process_group, flatten_parameters=False)

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


def _create_model(embedding_size: int, with_fsdp: bool, process_group, flatten_parameters: bool = True):
    model = ConvolutionalModel(with_fsdp=with_fsdp, process_group=process_group, embedding_size=embedding_size).cuda()
    if with_fsdp:
        return FullyShardedDataParallel(model, process_group=process_group, flatten_parameters=flatten_parameters)
    else:
        return model


def _load_sharded_checkpoint(rank: int):
    return torch.load(f"checkpoint_{rank}.torch")  # type: ignore


def _worker(gpu_id: int, sync_file: str, world_size: int, embedding_size: int, flatten_parameters: bool):
    torch.manual_seed(0)
    torch.cuda.set_device(gpu_id)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"file://{sync_file}",
        world_size=world_size,
        rank=gpu_id,
    )
    process_group = torch.distributed.new_group()

    # Create a dummy model with dummy inputs and targets
    batch_size = 4
    input = torch.randn(size=(batch_size, 3, 32, 32)).cuda()
    target = torch.zeros(size=(batch_size, embedding_size)).cuda()
    model = _create_model(
        with_fsdp=True,
        process_group=process_group,
        embedding_size=embedding_size,
        flatten_parameters=flatten_parameters,
    )
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
        shard_weights=[c["weights"] for c in all_checkpoints],
        shard_metadata=[c["meta"] for c in all_checkpoints],
    )

    # Check that the reconstructed parameters are correct and of the right shape
    full_model = _create_model(with_fsdp=False, process_group=process_group, embedding_size=embedding_size)
    full_model_state_dict = full_model.state_dict()
    assert set(full_model_state_dict.keys()) == set(consolidated_checkpoint.keys())
    for k in full_model_state_dict.keys():
        assert consolidated_checkpoint[k].shape == full_model_state_dict[k].shape

    # Verify that the checkpoint can be loaded by a FSDP model
    loaded_model = _create_model(
        with_fsdp=True,
        process_group=process_group,
        embedding_size=embedding_size,
        flatten_parameters=flatten_parameters,
    )
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

    world_size = 2
    with in_temporary_directory():
        with temp_files_ctx(num=1) as temp_files:
            mp.spawn(_worker, (temp_files[0], world_size, embedding_size, flatten_parameters), nprocs=world_size)


@skip_if_single_gpu
class TestConsolidatedWeights(DistributedTest):
    @parameterized.expand(
        [[True], [False]],
        name_func=rename_test,
    )
    def test_consolidate_weights(self, transformer):
        config = {"mixed_precision": True, "flatten_parameters": True, "compute_dtype": torch.float32}
        world_size = min(torch.cuda.device_count(), 4)
        if USE_TEMPFILE:
            with tempfile.TemporaryDirectory() as d:
                paths = [os.path.join(d, f"checkpoint_{rank}.pt") for rank in range(world_size)]
                test_fn = functools.partial(
                    self._test_consolidate_weights, config, transformer=transformer, paths=paths
                )
                spawn_and_init(test_fn, world_sizes=[world_size])
        else:
            paths = [f"checkpoint_{rank}.pt" for rank in range(world_size)]
            test_fn = functools.partial(self._test_consolidate_weights, config, transformer=transformer, paths=paths)
            spawn_and_init(test_fn, world_sizes=[world_size])

    @classmethod
    def _test_consolidate_weights(self, config, rank, group, paths=None, transformer=False):
        """FSDP.gather_full_optim_state_dict() should return something very similar to optimizer.state_dict()"""
        # Establish reference behavior.

        if transformer:
            fsdp = self.get_wrapped_model(group, config=config).cuda()
        else:
            fsdp = FullyShardedDataParallel(MixtureOfExperts(group, wrapper_config=config)).cuda()

        optim = Adam(
            fsdp.parameters(),
            lr=0.01,
        )
        optim.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            x = fsdp.module.get_input(torch.device("cuda"))
            output = fsdp(*x)
            loss = fsdp.module.get_loss(x, output).to("cuda")
            fsdp.module.run_backward(loss)
            optim.step()

        # each worker saves a checkpoint with local_state_dict
        cp_data = {
            "weights": {k: v.cpu() for k, v in fsdp.local_state_dict().items()},
            "meta": fsdp.local_metadata_dict(),
        }
        torch.save(cp_data, paths[fsdp.rank])
        full_model_state_dict = fsdp.state_dict()
        torch.distributed.barrier()
        if fsdp.rank > 0:
            return
        all_checkpoints = [torch.load(p) for p in paths]
        consolidated_checkpoint = FullyShardedDataParallel.consolidate_shard_weights(
            shard_weights=[c["weights"] for c in all_checkpoints],
            shard_metadata=[c["meta"] for c in all_checkpoints],
        )
        full_model_extra = set(full_model_state_dict).difference(set(consolidated_checkpoint))
        consolidated_extra = set(consolidated_checkpoint).difference(set(full_model_state_dict))
        msg = f"full model extra keys: {full_model_extra}, consolidated extra {consolidated_extra}"
        for k in full_model_state_dict.keys():
            assert consolidated_checkpoint[k].shape == full_model_state_dict[k].shape
        assert set(full_model_state_dict.keys()) == set(consolidated_checkpoint.keys()), msg


def test_consolidate_missing_params():
    """This tests that fairseq experts, which are saved independently from the rest of the model, can be consolidated."""
    desired_path = "decoder.layers.1.moe_layer.experts.0"
    shard_metadata = {
        "param_metadata": [
            {
                "fsdp_path": "",
                "params": {
                    "flat_param_0": {"names": ["missing"], "shapes": [(12, 4)], "numels": [12 * 4], "padding": 0}
                },
                "no_broadcast_optim_state": False,
                "shared_param_info": [],
            },
            {
                "fsdp_path": desired_path,
                "params": {
                    "flat_param_0": {
                        "names": ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"],
                        "shapes": [(4, 4), (4,), (4, 4), (4,)],
                        "numels": [16, 4, 16, 4],
                        "padding": 0,
                    }
                },
                "no_broadcast_optim_state": True,
                "shared_param_info": [],
            },
        ],
        "buffer_names": ["missing.buffer"],
    }
    shard_weights = {"decoder.layers.1.moe_layer.experts.0.flat_param_0": torch.randn(40, dtype=torch.float16)}
    consolidated_weights = FullyShardedDataParallel.consolidate_shard_weights(
        [shard_weights], [shard_metadata], strict=False
    )
    assert len(consolidated_weights) == 4
    for k in consolidated_weights:
        assert k.startswith(desired_path), f"{k} doesnt start with {desired_path}"
