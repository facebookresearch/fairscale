import time

from helpers import dist_init, get_data, get_loss_fun, get_model
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fairscale.nn.data_parallel import SlowMoDistributedDataParallel

WORLD_SIZE = 2
EPOCHS = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(rank: int, world_size: int, epochs: int, use_slowmo: bool):

    # DDP
    dist_init(rank, world_size)
    device = torch.device("cpu") if DEVICE == "cpu" else rank  # type:ignore

    # Problem statement
    model = get_model().to(device)
    dataloader = get_data(n_batches=1)
    loss_fn = get_loss_fun()

    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-4)

    if not use_slowmo:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    else:
        model = SlowMoDistributedDataParallel(model, nprocs_per_node=1)

    training_start = time.monotonic()
    # Any relevant training loop, nothing specific to SlowMo. For example:
    model.train()

    for _ in range(epochs):
        for (data, target) in dataloader:
            data, target = data.to(device), target.to(device)

            # Train
            model.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
            if use_slowmo:
                model.perform_slowmo(optimizer)

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
