import torch
from fairscale.optim.oss import OSS

import torch.distributed as dist
from torchvision import datasets, transforms
import torch.nn as nn
import torch.multiprocessing as mp

import time

WORLD_SIZE = 2
EPOCHS = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def dist_init(rank, world_size):
    backend = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO  # type: ignore
    print(f"Using backend: {backend}")
    dist.init_process_group(backend=backend, init_method="tcp://localhost:29501", rank=rank, world_size=world_size)

def myAwesomeModel():
    return nn.Sequential(
    nn.Conv2d(1, 32, 3, 1),
    nn.ReLU(),
    nn.Conv2d(32, 64, 3, 1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout2d(0.25),
    nn.Flatten(1),
    nn.Linear(9216, 128),
    nn.ReLU(),
    nn.Dropout2d(0.5),
    nn.Linear(128, 10),
    nn.LogSoftmax(dim=1))


def mySuperFastDataloader():
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    return torch.utils.data.DataLoader(dataset)

def myVeryRelevantLoss():
    return nn.CrossEntropyLoss()

def train(
    rank: int,
    world_size: int,
    epochs: int,
    use_oss: bool):
    
    
    # DDP
    dist_init(rank, world_size)

    # Problem statement
    model = myAwesomeModel().to(rank)
    dataloader = mySuperFastDataloader()
    loss_fn = myVeryRelevantLoss()

    base_optimizer_arguments = { "lr": 1e-4} # any optimizer specific arguments, LR, momentum, etc...
    if ~use_oss:
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            **base_optimizer_arguments)
    else:
        base_optimizer = torch.optim.SGD
        optimizer = OSS(
            params=model.parameters(),
            optim=base_optimizer,
            **base_optimizer_arguments)

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
            loss /= world_size
            loss.backward()
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            optimizer.step()
            # print(f"Loss: {loss.item()}")
    
    training_end = time.monotonic()
    max_memory = torch.cuda.max_memory_allocated(rank)

    print(f"[{dist.get_rank()}] : Training done. {training_end-training_start:.2f} sec")
    print(f"[{dist.get_rank()}] : Peak memory {max_memory:.1f}MiB")

if __name__ == '__main__':
    
    training_start1 = time.monotonic()
    mp.spawn(
        train,
        args=(WORLD_SIZE, EPOCHS, False),
        nprocs=WORLD_SIZE,
        join=True
    )
    training_end1 = time.monotonic()
    
    training_start2 = time.monotonic()
    mp.spawn(
        train,
        args=(WORLD_SIZE, EPOCHS, True),
        nprocs=WORLD_SIZE,
        join=True
    )
    training_end2 = time.monotonic()
    
    print("Total Time without:",training_end1-training_start1)
    print("Total Time with:",training_end2-training_start2)

