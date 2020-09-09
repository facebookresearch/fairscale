# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import argparse
import math
import os
import time
from typing import Any, List, Union, cast

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
OPTIM = torch.optim.RMSprop


def dist_init(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend=BACKEND, rank=rank, world_size=world_size)


def train(
    rank: int,
    world_size: int,
    num_epochs: int = 10,
    batch_size: int = 32,
    data_size: int = 200,
    use_oss: bool = True,
    check_regression: bool = True,
    reference_speed: float = -1.0,
    reference_memory: float = -1.0,
):

    # DDP
    dist_init(rank, world_size)

    # Standard RN101
    model = resnet101(pretrained=False, progress=True).to(rank)

    # Data setup, dummy data
    def collate(inputs: List[Any]):
        return {
            "inputs": torch.stack([i[0] for i in inputs]).to(torch.device(rank)),
            "label": torch.stack([i[1] for i in inputs]).to(torch.device(rank)),
        }

    dataloader = DataLoader(
        dataset=FakeData(transform=ToTensor(), size=data_size), batch_size=batch_size, collate_fn=collate
    )
    loss_fn = nn.CrossEntropyLoss()

    # Reset the memory use counter
    torch.cuda.reset_peak_memory_stats(rank)

    # Shard the optimizer
    optimizer: Union[OSS, OPTIM] = OSS(
        params=model.parameters(), optim=OPTIM, lr=1e-4, momentum=0.9
    ) if use_oss else OPTIM(model.parameters(), lr=1e-4, momentum=0.9)

    # Dummy training loop
    torch.cuda.synchronize(rank)
    training_start = time.monotonic()
    model.train()

    measurements = []

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

        if use_oss:
            # Check the checkpointing in the case of the OSS optimizer
            # Memory usage could spill over from there
            optimizer = cast(OSS, optimizer)
            # optimizer.consolidate_state_dict()
            if dist.get_rank() == 0:
                # _ = optimizer.state_dict()
                print("... State dict collected")

        measurements.append(data_size / (epoch_end - epoch_start))
        if dist.get_rank() == 0:
            print(f"Epoch {epoch} - processed {measurements[-1]:.2f} img per sec")

    torch.cuda.synchronize(rank)
    training_stop = time.monotonic()
    img_per_sec = data_size / (training_stop - training_start) * num_epochs
    max_memory = torch.cuda.max_memory_allocated(rank) / 2 ** 20

    print(f"[{dist.get_rank()}] : Training done. {img_per_sec:.2f} img per sec overall")
    print(f"[{dist.get_rank()}] : Peak memory {max_memory:.1f}MiB")

    # Compute the mean and average img per second
    mean = sum(measurements) / len(measurements)
    diff = map(lambda x: pow(x - mean, 2.0), measurements)
    std = math.sqrt(sum(diff) / (len(measurements) - 1))
    print(f"[{dist.get_rank()}] : Mean speed: {mean:.2f} +/- {std:.2f}")

    if use_oss and check_regression and dist.get_rank() == 0:
        assert (mean + 3.0 * std) > reference_speed, "Speed regression detected"
        assert max_memory < 1.05 * reference_memory, "Memory use regression detected"
        print("[Regression Test] VALID")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Benchmark the optimizer state sharding, on a typical computer vision workload"
    )
    parser.add_argument("--world_size", action="store", default=2, type=int)
    parser.add_argument("--epochs", action="store", default=10, type=int)
    parser.add_argument("--batch_size", action="store", default=32, type=int)
    parser.add_argument("--data_size", action="store", default=512, type=int)
    parser.add_argument("--check_regression", action="store_true", default=False)
<<<<<<< HEAD
    parser.add_argument("--reference_speed", action="store", default=33, type=float)
=======
    parser.add_argument("--reference_speed", action="store", default=32.32, type=float)
>>>>>>> upstream/master
    parser.add_argument("--reference_memory", action="store", default=4475, type=float)

    args = parser.parse_args()
    print(f"Benchmark arguments: {args}")

    print("\nBenchmark vanilla optimizer")
    mp.spawn(
        train,
        args=(args.world_size, args.epochs, args.batch_size, args.data_size, False, False),
        nprocs=args.world_size,
        join=True,
    )

    print("\nBenchmark OSS")
    mp.spawn(
        train,
        args=(
            args.world_size,
            args.epochs,
            args.batch_size,
            args.data_size,
            True,
            args.check_regression,
            args.reference_speed,
            args.reference_memory,
        ),
        nprocs=args.world_size,
        join=True,
    )
