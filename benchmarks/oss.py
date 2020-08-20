# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import os
import time
from typing import Any, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader

from fairscale.optim.oss import OSS
from torchvision.datasets import FakeData
from torchvision.models import resnet50
from torchvision.transforms import ToTensor

BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO  # type: ignore


def dist_init(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend=BACKEND, rank=rank, world_size=world_size)


def train(rank: int, world_size: int, num_epochs: int = 10, batch_size: int = 32):
    # DDP
    dist_init(rank, world_size)

    # Standard RN50
    model = resnet50(pretrained=False, progress=True).to(rank)
    print("Benchmarking model: ", model)

    # Data setup, dummy data
    def collate(inputs: List[Any]):
        return {
            "inputs": torch.stack([i[0] for i in inputs]).to(rank),
            "label": torch.stack([i[1] for i in inputs]).to(rank),
        }

    def _print(msg):
        if dist.get_rank() == 0:
            print(msg)

    num_images = 200
    dataloader = DataLoader(
        dataset=FakeData(transform=ToTensor(), size=num_images), batch_size=batch_size, collate_fn=collate
    )
    loss_fn = nn.CrossEntropyLoss()

    # Shard the optimizer
    optimizer = OSS(params=model.parameters(), optim=torch.optim.SGD, lr=1e-4)

    # Dummy training loop
    model.train()
    for epoch in range(num_epochs):
        _print(f"\n[{dist.get_rank()}] : Epoch {epoch}")
        epoch_start = time.monotonic()

        for batch in dataloader:

            def closure():
                model.zero_grad()
                outputs = model(batch["inputs"])
                loss = loss_fn(outputs, batch["label"])
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= world_size
                loss.backward()
                _print(f"[{dist.get_rank()}] : loss {loss.item():.2f}")
                return loss

            optimizer.step(closure)

        epoch_end = time.monotonic()
        img_per_sec = num_images / (epoch_end - epoch_start)
        _print(f"[{dist.get_rank()}] : processed {img_per_sec:.2f} img per sec")


if __name__ == "__main__":
    # TODO: really use DDP, not multiprocessing
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
