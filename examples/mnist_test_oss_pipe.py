# adapted from https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function

import argparse
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from fairscale.nn.data_parallel import ShardedDataParallel

WORLD_SIZE = 2
OPTIM = torch.optim.RMSprop
BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO


def dist_init(rank, world_size, backend):
    print(f"Using backend: {backend}")
    dist.init_process_group(backend=backend, init_method="tcp://localhost:29501", rank=rank, world_size=world_size)

net = nn.Sequential(
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
    nn.LogSoftmax(dim=1),
)


def train(rank, args, model, device, train_loader, num_epochs):
    ##############
    # SETUP
    dist_init(rank, WORLD_SIZE, BACKEND)
    ddp = ShardedDataParallel(
        module=model, optimizer=torch.optim.Adadelta, optimizer_params={"lr": 1e-4}, world_size=WORLD_SIZE, broadcast_buffers=True)

    ddp.train()
    optimizer = ddp.optimizer
    # Reset the memory use counter
    torch.cuda.reset_peak_memory_stats(rank)

    # Training loop
    torch.cuda.synchronize(rank)
    training_start = time.monotonic()

    loss_fn = nn.CrossEntropyLoss()
    ##############

    model.train()
    measurements = []
    for epoch in range(num_epochs):
        epoch_start = time.monotonic()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            def closure():
                model.zero_grad()
                outputs = model(data)
                loss = loss_fn(output.to(device), target.to(device))
                loss /= WORLD_SIZE
                loss.backward()

                # if dist.get_rank() == 0:
                #     print(f"Loss: {loss.item()}")

                ddp.reduce()  # Send the gradients to the appropriate shards
                return loss

            optimizer.step(closure)

        epoch_end = time.monotonic()

    torch.cuda.synchronize(rank)
    training_stop = time.monotonic()
    print("Total Time:", training_stop - training_start)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=14, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)

    model = net
    model = fairscale.nn.Pipe(model, balance=[6, 6], devices=[0, 1], chunks=2)
    device = model.devices[0]

    mp.spawn(
        train, args=(args, model, device, train_loader, args.epochs), nprocs=WORLD_SIZE, join=True,
    )


if __name__ == "__main__":
    main()
