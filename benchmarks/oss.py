# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import argparse
from enum import Enum
import importlib
import logging
import shutil
import tempfile
import time
from typing import Any, List, Optional, cast

from golden_configs import oss_mnist
import numpy as np
import torch
import torch.autograd.profiler as profiler
from torch.cuda.amp import GradScaler as TorchGradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import BatchSampler, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor

from fairscale.nn.data_parallel import OffloadDataParallelExperimental as OffloadDDPExperimental
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim import OSS
from fairscale.optim.grad_scaler import ShardedGradScaler

OPTIM = torch.optim.RMSprop
TEMPDIR = tempfile.gettempdir()


class ReshapeModule(torch.nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class ReshapeTokens(torch.nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        return torch.cat((cls_token, x), dim=1)


class Norm(torch.nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.norm = vit.norm

    def forward(self, x):
        x = self.norm(x)
        return x[:, 0]


class PosEmbed(torch.nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.pos_embed = vit.pos_embed
        self.pos_drop = vit.pos_drop

    def forward(self, x):
        return self.pos_drop(x + self.pos_embed)


def dist_init(rank, world_size, backend):
    logging.info(f"Using backend: {backend}")
    dist.init_process_group(backend=backend, init_method="tcp://localhost:29501", rank=rank, world_size=world_size)


def get_problem(rank, world_size, batch_size, device, model_name: str, unroll_model: bool = False):
    # Select the desired model on the fly
    logging.info(f"Using {model_name} for benchmarking")
    try:
        model = getattr(importlib.import_module("torchvision.models"), model_name)(pretrained=False)
    except AttributeError:
        model = getattr(importlib.import_module("timm.models"), model_name)(pretrained=False)

    # Tentatively unroll the model
    if unroll_model:
        if "resnet" in model_name:
            model = torch.nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                *model.layer1,
                *model.layer2,
                *model.layer3,
                *model.layer4,
                model.avgpool,
                ReshapeModule(),
                model.fc,
            )
        elif "vit" in model_name:
            model = torch.nn.Sequential(ReshapeTokens(model), PosEmbed(model), *model.blocks, Norm(model), model.head)
        else:
            raise RuntimeError("This model cannot be unrolled")

    # Data setup, duplicate the grey channels to get pseudo color
    def collate(inputs: List[Any]):
        return {
            "inputs": torch.stack([i[0] for i in inputs]).repeat(1, 3, 1, 1).to(device),
            "label": torch.tensor([i[1] for i in inputs]).to(device),
        }

    # Transforms
    transforms = []
    if model_name.startswith("vit"):
        # ViT models are fixed size. Add a ad-hoc transform to resize the pictures accordingly
        pic_size = int(model_name.split("_")[-1])
        transforms.append(Resize(pic_size))

    transforms.append(ToTensor())

    dataset = MNIST(transform=Compose(transforms), download=False, root=TEMPDIR)
    sampler: Sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)
    dataloader = DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=collate)

    loss_fn = nn.CrossEntropyLoss()
    return model, dataloader, loss_fn


class OptimType(str, Enum):
    vanilla = "pytorch"
    oss_ddp = "oss_ddp"
    oss_sharded_ddp = "oss_sharded_ddp"
    oss_offload_ddp = "oss_offload_ddp"
    everyone = "everyone"


def validate_benchmark(measurements, args, check_regression):
    """Validate the measurments against the golden benchmark config."""

    golden_data = oss_mnist.get_golden_real_stats()

    max_memory = -1.0
    rank = dist.get_rank()
    if not args.cpu:
        # TODO(anj-s): Check if we need to synchronize before we caculate total training time.
        torch.cuda.synchronize(rank)
        max_memory = torch.cuda.max_memory_allocated(rank) / 2 ** 20
        logging.info(f"[{rank}] : Peak memory {max_memory:.1f}MiB")

    measurements.sort()
    median = measurements[len(measurements) // 2]
    # Compute the median and median of absolute differences img per second.
    abs_diff = list(map(lambda x: abs(x - median), measurements))
    abs_diff.sort()
    mad = abs_diff[len(measurements) // 2] if args.epochs > 2 else -1

    # TODO(anj-s): Add a debug flag to perform the above calculation only when required.
    logging.info(f"[{rank}] : Median speed: {median:.2f} +/- {mad:.2f}")

    if check_regression and rank == 0:
        assert (median + 3.0 * mad) > golden_data["reference_speed"], "Speed regression detected"
        assert max_memory < 1.05 * golden_data["reference_memory"], "Memory use regression detected"
        assert abs(cast(float, final_loss) - golden_data["reference_loss"]) < 1e-3, "Loss regression detected"

        logging.info("[Regression Test] VALID")


def train(
    rank: int,
    args: argparse.Namespace,
    backend: str = "gloo",
    optim_type: OptimType = OptimType.vanilla,
    check_regression: bool = True,
):
    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)

    # DDP
    dist_init(rank=rank, world_size=args.world_size, backend=backend)

    # Setup
    if not args.cpu:
        torch.cuda.set_device(rank)
        torch.cuda.manual_seed(0)
    torch.manual_seed(0)  # also sets the cuda seed
    np.random.seed(0)
    torch.cuda.device(rank)

    if backend == "nccl":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cpu") if args.cpu else torch.device(rank)
    model, dataloader, loss_fn = get_problem(
        rank,
        args.world_size,
        args.batch_size,
        device,
        args.model,
        unroll_model=optim_type == OptimType.oss_offload_ddp,
    )

    # Shard the optimizer, test different methods
    optimizer: Optional[torch.optim.Optimizer] = None
    model = cast(nn.Module, model)
    scaler = (TorchGradScaler() if args.optim_type == OptimType.vanilla else ShardedGradScaler()) if args.amp else None

    if optim_type == OptimType.oss_sharded_ddp:
        model = model.to(device)
        optimizer = OSS(params=model.parameters(), optim=OPTIM, lr=1e-4, momentum=0.9)
        model = ShardedDDP(model, optimizer)
    elif optim_type == OptimType.oss_offload_ddp:
        ddp_exp = OffloadDDPExperimental(
            model_cpu=model,
            optimizer=OPTIM,
            optimizer_params={"lr": 1e-4, "momentum": 0.9},
            world_size=args.world_size,
            device=torch.device(torch.cuda.current_device()),
            offload_device=torch.device("cpu"),
        )
        optimizer = ddp_exp.optimizer
        model = ddp_exp
    else:
        model = model.to(device)
        device_ids = None if args.cpu else [rank]
        model = DDP(model, device_ids=device_ids, find_unused_parameters=False)  # type: ignore
        optimizer = (
            OSS(params=model.parameters(), optim=OPTIM, lr=1e-4, momentum=0.9)
            if optim_type == OptimType.oss_ddp
            else OPTIM(model.parameters(), lr=1e-4, momentum=0.9)
        )

    optimizer = cast(torch.optim.Optimizer, optimizer)

    # Reset the memory use counter
    if not args.cpu:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(rank)
        torch.cuda.synchronize(rank)

    # Standard training loop
    training_start = time.monotonic()
    model.train()

    measurements = []
    final_loss: Optional[float] = -1.0
    optimizer = cast(Optimizer, optimizer)
    need_profiling = args.profile

    for epoch in range(args.epochs):
        n_items = 0
        epoch_runtime = 0.0

        for batch in dataloader:
            if not args.cpu:
                torch.cuda.synchronize(rank)
            batch_start = time.monotonic()

            def closure(data=batch, grad_scaler=None):
                model.zero_grad()
                if args.debug and rank == 0 and next(model.parameters()).grad is not None:
                    logging.debug(
                        "\nbefore:  param {} -- grad {}".format(
                            next(model.parameters()).norm().item(), next(model.parameters()).grad.norm().item()
                        )
                    )
                if grad_scaler is not None:
                    # Automatically computes the FW pass in half precision
                    with torch.cuda.amp.autocast():
                        outputs = model(data["inputs"])
                        loss = loss_fn(outputs, data["label"])

                        # Accumulates scaled gradients.
                        grad_scaler.scale(loss).backward()
                else:
                    outputs = model(data["inputs"])
                    loss = loss_fn(outputs, data["label"])
                    loss.backward()

                if args.debug and rank == 0 and next(model.parameters()).grad is not None:
                    logging.debug(
                        "after BW: param {} -- grad {}".format(
                            next(model.parameters()).norm().item(), next(model.parameters()).grad.norm().item()
                        )
                    )
                return loss

            def run_closure(closure, scaler, optimizer):
                if scaler is not None:
                    final_loss = closure(grad_scaler=scaler)  # AMP scaler.step does not support closures
                    scaler.step(optimizer)
                    scaler.update()
                    return final_loss
                else:
                    return optimizer.step(closure)

            if need_profiling and not args.cpu:
                logging.info("Profiling the run")
                with profiler.profile(use_cuda=True, record_shapes=True, profile_memory=True) as prof:  # type: ignore
                    with profiler.record_function("batch"):
                        final_loss = run_closure(closure, scaler, optimizer)

                prof.export_chrome_trace(f"{optim_type}_trace_rank_{rank}.json")
                need_profiling = False  # only profile once

            else:
                final_loss = run_closure(closure, scaler, optimizer)

            if args.debug and rank == 0:
                logging.debug("buffer: {}".format(next(model.buffers()).norm().item()))
                logging.debug(
                    "after update: param {} -- grad {}".format(
                        next(model.parameters()).norm().item(), next(model.parameters()).grad.norm().item()
                    )
                )

            n_items += args.batch_size

            if not args.cpu:
                # make sure that the cuda kernels are finished before taking a timestamp
                torch.cuda.synchronize(rank)

            batch_end = time.monotonic()
            epoch_runtime += batch_end - batch_start

        if optim_type == OptimType.oss_ddp or optim_type == OptimType.oss_sharded_ddp:
            # Check the checkpointing in the case of the OSS optimizer
            # Memory usage could spill over from there
            optimizer = cast(OSS, optimizer)
            optimizer.consolidate_state_dict()
            if dist.get_rank() == 0:
                _ = optimizer.state_dict()
                logging.info("... State dict collected")

        measurements.append(n_items / epoch_runtime)
        if dist.get_rank() == 0:
            logging.info(f"Epoch {epoch} - processed {measurements[-1]:.2f} img per sec. Loss {final_loss:.3f}")

    training_stop = time.monotonic()
    img_per_sec = n_items / (training_stop - training_start) * args.epochs
    logging.info(f"[{dist.get_rank()}] : Training done. {img_per_sec:.2f} img per sec inc. checkpoint")

    validate_benchmark(measurements, args, check_regression)

    dist.destroy_process_group()  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the optimizer state sharding, on a typical computer vision workload"
    )
    parser.add_argument("--world_size", action="store", default=2, type=int)
    parser.add_argument("--epochs", action="store", default=10, type=int)
    parser.add_argument("--batch_size", action="store", default=256, type=int)
    parser.add_argument("--check_regression", action="store_true", default=False)
    parser.add_argument("--reference_speed", action="store", default=1430, type=float)
    parser.add_argument("--reference_memory", action="store", default=1220, type=float)
    parser.add_argument("--reference_loss", action="store", default=0.006, type=float)
    parser.add_argument(
        "--optim_type", type=OptimType, choices=[o.value for o in OptimType], default=OptimType.everyone
    )
    parser.add_argument("--gloo", action="store_true", default=False)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--model", type=str, help="Any torchvision or timm model name (str)", default="resnet101")
    parser.add_argument("--debug", action="store_true", default=False, help="Display additional debug information")
    parser.add_argument("--amp", action="store_true", default=False, help="Activate torch AMP")
    parser.add_argument("--fake_data", action="store_true", default=False, help="Use fake data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)
    logging.info("Benchmark arguments: %s" % args)

    BACKEND = "nccl" if (not args.gloo or not torch.cuda.is_available()) and not args.cpu else "gloo"

    # Download dataset once for all processes
    dataset, tentatives = None, 0
    while dataset is None and tentatives < 5:
        try:
            dataset = MNIST(transform=None, download=True, root=TEMPDIR)
        except (RuntimeError, EOFError) as e:
            if isinstance(e, RuntimeError):
                # Corrupted data, erase and restart
                shutil.rmtree(TEMPDIR + "/MNIST")

            logging.warning("Failed loading dataset: %s " % e)
            tentatives += 1

    if dataset is None:
        logging.error("Could not download MNIST dataset")
        exit(-1)
    else:
        logging.info("Dataset downloaded")

    # Benchmark the different configurations, via multiple processes
    if args.optim_type == OptimType.vanilla or args.optim_type == OptimType.everyone:
        logging.info("\n*** Benchmark vanilla optimizer")
        mp.spawn(
            train,  # type: ignore
            args=(args, BACKEND, OptimType.vanilla, False),  # no regression check
            nprocs=args.world_size,
            join=True,
        )

    if args.optim_type == OptimType.oss_ddp or args.optim_type == OptimType.everyone:
        logging.info("\n*** Benchmark OSS with DDP")
        mp.spawn(
            train, args=(args, BACKEND, OptimType.oss_ddp, args.check_regression), nprocs=args.world_size, join=True,  # type: ignore
        )

    if args.optim_type == OptimType.oss_sharded_ddp or args.optim_type == OptimType.everyone:
        logging.info("\n*** Benchmark OSS with ShardedDDP")
        mp.spawn(
            train,  # type: ignore
            args=(
                args,
                BACKEND,
                OptimType.oss_sharded_ddp,
                False,
            ),  # FIXME: @lefaudeux - SDP should give the same results
            nprocs=args.world_size,
            join=True,
        )

    if args.optim_type == OptimType.oss_offload_ddp or args.optim_type == OptimType.everyone:
        print("\nBenchmark OSS experimental")
        mp.spawn(
            train, args=(args, BACKEND, OptimType.oss_offload_ddp, False,), nprocs=args.world_size, join=True,  # type: ignore
        )
