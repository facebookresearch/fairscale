# run with:
# mpirun -np 2 --host localhost:2 -x PYTHONPATH=$PWD python # examples/tutorial_pipe_rpc.py

import os

from helpers import dist_init, get_data, get_loss_fun, get_model
import torch
import torch.optim as optim
import torch_pg

import fairscale
from fairscale.nn.model_parallel import initialize_model_parallel


def register_optimizer(ctx, model):
    # Set the optimizer as an attribute on the model so we can access it later
    model.optimizer = optim.SGD(model.parameters(), **ctx)
    # zero the parameter gradients
    model.optimizer.zero_grad()


def run_optimizer(ctx, model):
    model.optimizer.step()


def run(rank, world_size):
    torch_pg.init_mpi()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "10638"
    dist_init(rank, world_size)  # FIXME (supports gloo)
    os.environ["MASTER_PORT"] = "10639"
    torch.distributed.rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
    initialize_model_parallel(1, world_size, pipeline_backend="mpi")

    if rank == 1:
        # For RPC, all ranks other than 0 just need to call rpc.shutdown()
        torch.distributed.rpc.shutdown()
        return

    model = get_model()
    data, target = get_data()[0]
    loss_fn = get_loss_fun()

    device = torch.device("cuda", rank)

    model = fairscale.nn.PipeRPCWrapper(
        model,
        balance=[2, 1],
        worker_map={0: "worker0", 1: "worker1"},  # Needed to convert ranks to RPC worker names
        input_device=device,
    ).to(device)

    # We can't directly access the model on each worker, so we need to call
    # foreach_worker with a callback to setup the optimizer
    model.foreach_worker(register_optimizer, {"lr": 0.001}, include_self=True)

    outputs = model(data.to(device))
    loss = loss_fn(outputs.to(device), target.to(device))
    loss.backward()

    # Same as earlier, use foreach_worker to step the optimizer on each rank
    model.foreach_worker(run_optimizer, include_self=True)

    print(f"Finished Training Step on {rank}")

    torch.distributed.rpc.shutdown()

    del model


if __name__ == "__main__":
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    run(rank, world_size)
