# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import os
import time
from typing import Any, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.models import resnet101
from torchvision.transforms import ToTensor

from fairscale.optim.oss import OSS

BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO  # type: ignore


def dist_init(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend=BACKEND, rank=rank, world_size=world_size)


def train(
    rank: int, world_size: int, num_epochs: int = 10, batch_size: int = 32, data_size: int = 200, use_oss: bool = True
):
    # DDP
    dist_init(rank, world_size)

    # Standard RN101
    model = resnet101(pretrained=False, progress=True).to(rank)

    # Data setup, dummy data
    def collate(inputs: List[Any]):
        return {
            "inputs": torch.stack([i[0] for i in inputs]).to(rank),
            "label": torch.stack([i[1] for i in inputs]).to(rank),
        }

    def _print(msg):
        if dist.get_rank() == 0:
            print(msg)

    dataloader = DataLoader(
        dataset=FakeData(transform=ToTensor(), size=data_size), batch_size=batch_size, collate_fn=collate
    )
    loss_fn = nn.CrossEntropyLoss()

    # Shard the optimizer
    optimizer = (
        OSS(params=model.parameters(), optim=torch.optim.SGD, lr=1e-4, momentum=0.9)
        if use_oss
        else torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    )

    # Dummy training loop
    torch.cuda.synchronize(rank)
    training_start = time.monotonic()
    model.train()
    for epoch in range(num_epochs):
        epoch_start = time.monotonic()

        for batch in dataloader:

            def closure():
                model.zero_grad()
                outputs = model(batch["inputs"])
                loss = loss_fn(outputs, batch["label"])
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= world_size
                loss.backward()
                return loss

            optimizer.step(closure)

        epoch_end = time.monotonic()
        img_per_sec = data_size / (epoch_end - epoch_start)
        _print(f"[{dist.get_rank()}] : Epoch {epoch} - processed {img_per_sec:.2f} img per sec")

    torch.cuda.synchronize(rank)
    training_stop = time.monotonic()
    img_per_sec = data_size / (training_stop - training_start) * num_epochs
    max_memory = torch.cuda.max_memory_allocated(rank) / 2 ** 20

    _print(f"[{dist.get_rank()}] : Training done. {img_per_sec:.2f} img per sec overall")
    print(f"[{dist.get_rank()}] : Peak memory {max_memory:.1f}MiB")


if __name__ == "__main__":
    WORLD_SIZE = 2
    EPOCHS = 10
    BATCH_SIZE = 32
    DATA_SIZE = 512

    print("\nBenchmark vanilla SGD")
    mp.spawn(train, args=(WORLD_SIZE, EPOCHS, BATCH_SIZE, DATA_SIZE, False), nprocs=WORLD_SIZE, join=True)

    print("\nBenchmark OSS")
    mp.spawn(train, args=(WORLD_SIZE, EPOCHS, BATCH_SIZE, DATA_SIZE, True), nprocs=WORLD_SIZE, join=True)
