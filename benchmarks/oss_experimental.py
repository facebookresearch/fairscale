# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import argparse
import math
import time
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from benchmarks.oss import dist_init, get_problem
from fairscale.nn.data_parallel import ShardedDataParallelExperimental

OPTIM = torch.optim.RMSprop


class ReshapeModule(torch.nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


def train(
    rank: int, world_size: int, num_epochs: int = 10, batch_size: int = 32, data_size: int = 200, backend: str = "gloo"
):
    # DDP
    dist_init(rank=rank, world_size=world_size, backend=backend)

    # Setup
    torch.cuda.set_device(rank)
    torch.cuda.manual_seed(0)
    torch.manual_seed(0)  # also sets the cuda seed
    np.random.seed(0)

    if backend == "nccl":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    resnet, dataloader, loss_fn = get_problem(rank, data_size, batch_size)

    # FIXME: Sequentialize the model
    model_seq = torch.nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4,
        resnet.avgpool,
        ReshapeModule(),
        resnet.fc,
    )

    # Shard the optimizer
    optimizer: Optional[torch.optim.Optimizer] = None

    ddp = ShardedDataParallelExperimental(
        module=model_seq, optimizer=OPTIM, optimizer_params={"lr": 1e-4, "momentum": 0.9}, world_size=world_size,
    )
    ddp.train()
    optimizer = ddp.optimizer
    model = ddp

    # Reset the memory use counter
    torch.cuda.reset_peak_memory_stats(rank)

    # Dummy training loop
    torch.cuda.synchronize(rank)
    training_start = time.monotonic()
    model.train()

    measurements = []
    final_loss: Optional[float] = -1.0

    for epoch in range(num_epochs):
        epoch_start = time.monotonic()

        for batch in dataloader:

            def closure():
                model.zero_grad()
                outputs = model(batch["inputs"])
                loss = loss_fn(outputs, batch["label"])
                loss /= world_size
                loss.backward()
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                return loss

            final_loss = optimizer.step(closure)

        epoch_end = time.monotonic()

        measurements.append(data_size / (epoch_end - epoch_start))
        if dist.get_rank() == 0:
            print(f"Epoch {epoch} - processed {measurements[-1]:.2f} img per sec. Loss {final_loss:.3f}")

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

    dist.destroy_process_group()  # type: ignore


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Benchmark the optimizer state sharding, on a typical computer vision workload"
    )
    parser.add_argument("--world_size", action="store", default=2, type=int)
    parser.add_argument("--epochs", action="store", default=10, type=int)
    parser.add_argument("--batch_size", action="store", default=32, type=int)
    parser.add_argument("--data_size", action="store", default=512, type=int)
    parser.add_argument("--gloo", action="store_true", default=False)

    args = parser.parse_args()
    print(f"Benchmark arguments: {args}")

    backend = "nccl" if not args.gloo or not torch.cuda.is_available() else "gloo"

    mp.spawn(
        train,
        args=(args.world_size, args.epochs, args.batch_size, args.data_size, backend,),
        nprocs=args.world_size,
        join=True,
    )
