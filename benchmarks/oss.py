# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import argparse
from enum import Enum
import importlib
import math
import time
from typing import Any, List, Optional, cast

import numpy as np
import torch
import torch.autograd.profiler as profiler
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from fairscale.nn.data_parallel import ShardedDataParallel
from fairscale.optim.oss import OSS

OPTIM = torch.optim.RMSprop


def dist_init(rank, world_size, backend):
    print(f"Using backend: {backend}")
    dist.init_process_group(backend=backend, init_method="tcp://localhost:29501", rank=rank, world_size=world_size)


def get_problem(rank, data_size, batch_size, device, model_name: str):
    # Select the desired model on the fly
    print(f"Using {model_name} for benchmarking")
    model = getattr(importlib.import_module("torchvision.models"), model_name)(pretrained=False).to(device)

    # Data setup, dummy data
    def collate(inputs: List[Any]):
        return {
            "inputs": torch.stack([i[0] for i in inputs]).to(device),
            "label": torch.stack([i[1] for i in inputs]).to(device),
        }

    dataloader = DataLoader(
        dataset=FakeData(transform=ToTensor(), size=data_size, random_offset=rank),
        batch_size=batch_size,
        collate_fn=collate,
    )
    loss_fn = nn.CrossEntropyLoss()
    return model, dataloader, loss_fn


class OptimType(str, Enum):
    vanilla = "pytorch"
    oss = "oss"
    oss_sdp = "oss_sdp"
    everyone = "everyone"


def train(
    rank: int,
    args: argparse.Namespace,
    backend: str = "gloo",
    optim_type: OptimType = OptimType.vanilla,
    check_regression: bool = True,
):
    # DDP
    dist_init(rank=rank, world_size=args.world_size, backend=backend)

    # Setup
    torch.cuda.set_device(rank)
    torch.cuda.manual_seed(0)
    torch.manual_seed(0)  # also sets the cuda seed
    np.random.seed(0)

    if backend == "nccl":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cpu") if args.cpu else torch.device(rank)
    model, dataloader, loss_fn = get_problem(rank, args.data_size, args.batch_size, device, args.torchvision_model)

    # Shard the optimizer
    optimizer: Optional[torch.optim.Optimizer] = None

    if optim_type == OptimType.oss_sdp:
        ddp = ShardedDataParallel(
            module=model,
            optimizer=OPTIM,
            optimizer_params={"lr": 1e-4, "momentum": 0.9},
            world_size=args.world_size,
            broadcast_buffers=True,
        )
        ddp.train()
        optimizer = ddp.optimizer
        model = ddp
    else:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)  # type: ignore
        optimizer = (
            OSS(params=model.parameters(), optim=OPTIM, lr=1e-4, momentum=0.9)
            if optim_type == OptimType.oss
            else OPTIM(model.parameters(), lr=1e-4, momentum=0.9)
        )

    # Reset the memory use counter
    if not args.cpu:
        torch.cuda.reset_peak_memory_stats(rank)
        torch.cuda.synchronize(rank)

    # Dummy training loop
    training_start = time.monotonic()
    model.train()

    measurements = []
    final_loss: Optional[float] = -1.0
    need_profiling = args.profile

    for epoch in range(args.epochs):
        epoch_start = time.monotonic()

        for batch in dataloader:

            def closure():
                model.zero_grad()
                outputs = model(batch["inputs"])
                loss = loss_fn(outputs, batch["label"])
                loss.backward()

                if optim_type == OptimType.oss_sdp:
                    ddp.reduce()  # Send the gradients to the appropriate shards

                return loss

            if need_profiling and not args.cpu:
                print("Profiling the run")
                with profiler.profile(use_cuda=True) as prof:  # type: ignore
                    with profiler.record_function("batch"):
                        final_loss = optimizer.step(closure)
                        print("profiling done, final loss ", cast(float, final_loss))

                if rank == 0:
                    prof.export_chrome_trace(f"{optim_type}_trace.json")

                need_profiling = False  # only profile once

            else:
                final_loss = optimizer.step(closure)

        epoch_end = time.monotonic()

        if optim_type == OptimType.oss:
            # Check the checkpointing in the case of the OSS optimizer
            # Memory usage could spill over from there
            optimizer = cast(OSS, optimizer)
            optimizer.consolidate_state_dict()
            if dist.get_rank() == 0:
                _ = optimizer.state_dict()
                print("... State dict collected")

        measurements.append(args.data_size / (epoch_end - epoch_start))
        if dist.get_rank() == 0:
            print(f"Epoch {epoch} - processed {measurements[-1]:.2f} img per sec. Loss {final_loss:.3f}")

    if not args.cpu:
        torch.cuda.synchronize(rank)
    training_stop = time.monotonic()
    img_per_sec = args.data_size / (training_stop - training_start) * args.epochs
    max_memory = torch.cuda.max_memory_allocated(rank) / 2 ** 20

    print(f"[{dist.get_rank()}] : Training done. {img_per_sec:.2f} img per sec overall")
    print(f"[{dist.get_rank()}] : Peak memory {max_memory:.1f}MiB")

    # Compute the mean and average img per second
    mean = sum(measurements) / len(measurements)
    diff = map(lambda x: pow(x - mean, 2.0), measurements)
    std = math.sqrt(sum(diff) / (len(measurements) - 1))
    print(f"[{dist.get_rank()}] : Mean speed: {mean:.2f} +/- {std:.2f}")

    if check_regression and dist.get_rank() == 0:
        assert (mean + 3.0 * std) > args.reference_speed, "Speed regression detected"
        assert max_memory < 1.05 * args.reference_memory, "Memory use regression detected"
        assert abs(cast(float, final_loss) - args.reference_loss) < 1e-3, "Loss regression detected"

        print("[Regression Test] VALID")

    dist.destroy_process_group()  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the optimizer state sharding, on a typical computer vision workload"
    )
    parser.add_argument("--world_size", action="store", default=2, type=int)
    parser.add_argument("--epochs", action="store", default=10, type=int)
    parser.add_argument("--batch_size", action="store", default=32, type=int)
    parser.add_argument("--data_size", action="store", default=512, type=int)
    parser.add_argument("--check_regression", action="store_true", default=False)
    parser.add_argument("--reference_speed", action="store", default=29.7, type=float)
    parser.add_argument("--reference_memory", action="store", default=4475, type=float)
    parser.add_argument("--reference_loss", action="store", default=0.866, type=float)
    parser.add_argument(
        "--optim_type", type=OptimType, choices=[o.value for o in OptimType], default=OptimType.everyone
    )
    parser.add_argument("--gloo", action="store_true", default=False)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--torchvision_model", type=str, help="Any torchvision model name (str)", default="resnet101")

    args = parser.parse_args()
    print(f"Benchmark arguments: {args}")

    backend = "nccl" if (not args.gloo or not torch.cuda.is_available()) and not args.cpu else "gloo"

    if args.optim_type == OptimType.vanilla or args.optim_type == OptimType.everyone:
        print("\nBenchmark vanilla optimizer")
        mp.spawn(
            train,
            args=(args, backend, OptimType.vanilla, False,),  # no regression check
            nprocs=args.world_size,
            join=True,
        )

    if args.optim_type == OptimType.oss or args.optim_type == OptimType.everyone:
        print("\nBenchmark OSS with DDP")
        mp.spawn(
            train, args=(args, backend, OptimType.oss, args.check_regression), nprocs=args.world_size, join=True,
        )

    if args.optim_type == OptimType.oss_sdp or args.optim_type == OptimType.everyone:
        print("\nBenchmark OSS with SDP")
        mp.spawn(
            train,
            args=(args, backend, OptimType.oss_sdp, False,),  # FIXME: @lefaudeux - SDP should give the same results
            nprocs=args.world_size,
            join=True,
        )
