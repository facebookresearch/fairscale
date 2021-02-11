# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Based on https://github.com/pytorch/tutorials/blob/master/beginner_source/transformer_tutorial.py
# Apply CPU offload and problem sharding to a big transformer model

import argparse
import contextlib
import logging
import time

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

OPTIM = torch.optim.SGD
LR = 1e-3

from fairscale.nn.misc.offload import OffloadModel


def _get_fp16_context(use_fp16=False):
    if use_fp16:
        return torch.cuda.amp.autocast()
    else:
        return contextlib.nullcontext()


def _get_profiler_context(use_profiler=False):
    if use_profiler:
        return torch.autograd.profiler.profile(use_cuda=True, profile_memory=True)
    else:
        return contextlib.nullcontext()


def _get_profiler_record_context(record_name, use_profiler=False):
    if use_profiler:
        return torch.autograd.profiler.record_function(record_name)
    else:
        return contextlib.nullcontext()


def train(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    torch.manual_seed(5)

    # Setup the problem
    model = torch.nn.Sequential(
        torch.nn.Linear(args.inputs * args.inputs, args.hidden),
        *([torch.nn.Linear(args.hidden, args.hidden) for _ in range(args.layers)]),
        torch.nn.Linear(args.hidden, args.outputs),
    ).cpu()

    # Optim loop
    criterion = nn.CrossEntropyLoss()
    if args.offload:
        logging.info("Using sharded offloading for training")
        model = OffloadModel(
            model_cpu=model, device=device, offload_device=torch.device("cpu"), n_slices=args.slices,
            checkpoint_activation=args.checkpoint_activation
        )  # type: ignore

    else:
        logging.info("Using Pytorch for training")
        model = model.to(torch.device("cuda"))

    optimizer = OPTIM(model.parameters(), lr=LR)

    transform = ToTensor()
    dataloader = DataLoader(
        FakeData(image_size=(1, args.inputs, args.inputs), num_classes=args.outputs, transform=transform),
        batch_size=args.batch_size,
    )

    def train_epoch(args):
        model.train()
        iter_count = 2
        for batch_inputs, batch_outputs in dataloader:
            batch_inputs, batch_outputs = batch_inputs.to("cuda"), batch_outputs.to("cuda")
            start = time.time_ns()
            with _get_profiler_context() as prof:
                optimizer.zero_grad()
                inputs = batch_inputs.reshape(-1, args.inputs * args.inputs)
                with _get_profiler_record_context("model_training"):
                    with _get_fp16_context(use_fp16=args.use_fp16):
                        output = model(inputs)
                        loss = criterion(output, target=batch_outputs)
                        loss.backward()
                    optimizer.step()
            logging.info(
                "Memory stats are {:.2f}GB".format(torch.cuda.memory_stats(0)["allocated_bytes.all.peak"] / 2 ** 30)
            )
            logging.info(
                "Loss {:.2f} - throughput {:.2f}fps".format(
                    loss.item(), args.batch_size / (time.time_ns() - start) * 10 ** 9
                )
            )
            break
        if args.use_profiler:
            prof.export_chrome_trace("/tmp/offload_prof")
    train_epoch(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the CPU offload + sharding with a Transformer training")
    parser.add_argument("--epochs", action="store", default=1, type=int)
    parser.add_argument("--batch_size", action="store", default=16, type=int)
    parser.add_argument("--inputs", action="store", help="The dimension of the inputs", default=100, type=int)
    parser.add_argument("--hidden", action="store", help="The dimension of the hidden state", default=1000, type=int)
    parser.add_argument("--layers", action="store", help="he number of hidden layers", default=100, type=int)
    parser.add_argument("--outputs", action="store", help="The number of predicted classes", default=5, type=int)

    parser.add_argument("--offload", action="store_true", default=False)
    parser.add_argument("--slices", action="store", default=3, type=int)
    parser.add_argument("--use_fp16", action="store_true", default=False)
    parser.add_argument("--checkpoint_activation", action="store_true", default=False)
    parser.add_argument("--use_profiler", action="store_true", default=False)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Benchmark arguments: %s" % args)
    logging.info("Starting training")
    train(args)
