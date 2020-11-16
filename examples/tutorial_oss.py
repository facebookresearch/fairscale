import time
from typing import Optional, Union, cast

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from fairscale.optim.oss import OSS

WORLD_SIZE = 2
EPOCHS = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def dist_init(rank, world_size):
    backend = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO  # type: ignore
    print(f"Using backend: {backend}")
    dist.init_process_group(backend=backend, init_method="tcp://localhost:29501", rank=rank, world_size=world_size)


def getModel():
    return nn.Sequential(torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 5))


def getData():
    target = torch.randint(0, 2, size=(20, 1)).squeeze()
    data = torch.randn(20, 10)
    return [(data, target)]


def getLossFun():
    return F.nll_loss


def train(rank: int, world_size: int, epochs: int, use_oss: bool):

    # DDP
    dist_init(rank, world_size)

    # Problem statement
    model = getModel().to(rank)
    dataloader = getData()
    loss_fn = getLossFun()

    optimizer: Optional[Union[OSS, torch.optim.SGD]] = None

    if not use_oss:
        optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-4)
    else:
        base_optimizer = torch.optim.SGD
        base_optimizer_arguments = {"lr": 1e-4}  # any optimizer specific arguments, LR, momentum, etc...
        optimizer = OSS(params=model.parameters(), optim=base_optimizer, default=base_optimizer_arguments)

    training_start = time.monotonic()
    # Any relevant training loop, nothing specific to OSS. For example:
    model.train()
    for e in range(epochs):
        for (data, target) in dataloader:
            data, target = data.to(rank), target.to(rank)

            # Train
            model.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, target)
            loss.backward()

            # if you want to clip the gradients / get the current max:
            max_norm = 1000.0
            norm_type = 1
            if not use_oss:
                _total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=norm_type)  # type: ignore
            else:
                optimizer = cast(OSS, optimizer)
                _total_norm = optimizer.clip_grad_norm(max_norm, norm_type=norm_type)

            optimizer.step()

            print(f"Loss: {loss.item()}")

    training_end = time.monotonic()
    max_memory = torch.cuda.max_memory_allocated(rank)

    print(f"[{dist.get_rank()}] : Training done. {training_end-training_start:.2f} sec")
    print(f"[{dist.get_rank()}] : Peak memory {max_memory:.1f}MiB")


if __name__ == "__main__":

    training_start1 = time.monotonic()
    mp.spawn(train, args=(WORLD_SIZE, EPOCHS, False), nprocs=WORLD_SIZE, join=True)
    training_end1 = time.monotonic()

    training_start2 = time.monotonic()
    mp.spawn(train, args=(WORLD_SIZE, EPOCHS, True), nprocs=WORLD_SIZE, join=True)
    training_end2 = time.monotonic()

    print("Total Time without:", training_end1 - training_start1)
    print("Total Time with:", training_end2 - training_start2)
