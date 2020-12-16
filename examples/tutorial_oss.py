import time
from typing import Optional, Union, cast

from helpers import dist_init, getData, getLossFun, getModel
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fairscale.optim.oss import OSS

WORLD_SIZE = 2
EPOCHS = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(rank: int, world_size: int, epochs: int, use_oss: bool):

    # DDP
    dist_init(rank, world_size)
    device = torch.device("cpu") if DEVICE == "cpu" else rank  # type:ignore

    # Problem statement
    model = getModel().to(device)
    dataloader = getData(n_batches=1)
    loss_fn = getLossFun()

    optimizer: Optional[Union[OSS, torch.optim.SGD]] = None

    if not use_oss:
        optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-4)
    else:
        base_optimizer = torch.optim.SGD
        base_optimizer_arguments = {"lr": 1e-4}  # any optimizer specific arguments, LR, momentum, etc...
        optimizer = OSS(
            params=model.parameters(), optim=base_optimizer, broadcast_buffer_size=2 ** 17, **base_optimizer_arguments
        )

    training_start = time.monotonic()
    # Any relevant training loop, nothing specific to OSS. For example:
    model.train()

    for _ in range(epochs):
        for (data, target) in dataloader:
            data, target = data.to(device), target.to(device)

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
    print(f"[{dist.get_rank()}] : Training done. {training_end-training_start:.2f} sec")

    if DEVICE == "cuda":
        max_memory = torch.cuda.max_memory_allocated(rank)
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
