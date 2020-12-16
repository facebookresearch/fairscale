import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim

import fairscale
from fairscale.nn.model_parallel import initialize_model_parallel
from helpers import dist_init, getModel, getData, getLossFun


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "10638"
    dist_init(rank, world_size)
    os.environ["MASTER_PORT"] = "10639"
    dist.rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
    initialize_model_parallel(1, world_size)

    model = getModel()
    data, target = getData()[0]
    loss_fn = getLossFun()

    device = torch.device("cuda", rank) if DEVICE == "cuda" else torch.device("cpu")

    model = fairscale.nn.Pipe(
        model,
        balance=[2, 1],
        style=fairscale.nn.Pipe.MultiProcess,
        worker_map={0: "worker0", 1: "worker1"},  # Needed to convert ranks to RPC worker names
        input_device=device,
    ).to(device)

    # define optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # zero the parameter gradients
    optimizer.zero_grad()

    # outputs and target need to be on the same device
    # forward step
    outputs = model(data.to(device))
    # compute loss
    if rank == 1:
        loss = loss_fn(outputs.to(device), target.to(device))

        # backward + optimize
        loss.backward()
        optimizer.step()
    else:
        model.back_helper(outputs)

    print(f"Finished Training Step on {rank}")

    del model


if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
