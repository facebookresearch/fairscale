# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Based on https://github.com/pytorch/tutorials/blob/master/beginner_source/transformer_tutorial.py
# Apply CPU offload and problem sharding to a big transformer model

import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

OPTIM = torch.optim.SGD
LR = 1e-3

from fairscale.nn.misc.offload import OffloadWrapperExperimental


def train(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda")

    # Setup the problem
    model = torch.nn.Sequential(
        torch.nn.Linear(args.inputs * args.inputs, args.hidden),
        *([torch.nn.Linear(args.hidden, args.hidden)] * args.layers),
        torch.nn.Linear(args.hidden, args.outputs)
    ).cpu()

    # Optim loop
    criterion = nn.CrossEntropyLoss()
    if args.offload:
        logging.info("Using sharded offloading for training")
        offload_model = OffloadWrapperExperimental(
            model_cpu=model,
            optimizer=OPTIM,
            optimizer_params={"lr": LR},
            device=device,
            offload_device=torch.device("cpu"),
            n_slices=args.slices,
        )

        optimizer = offload_model.optimizer
        model = offload_model  # type: ignore

    else:
        logging.info("Using Pytorch for training")
        model = model.to(torch.device("cuda"))
        optimizer = OPTIM(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    transform = ToTensor()
    dataloader = DataLoader(
        FakeData(image_size=(1, args.inputs, args.inputs), num_classes=args.outputs, transform=transform), batch_size=32
    )

    def train_epoch():
        model.train()
        for batch_inputs, batch_outputs in dataloader:
            optimizer.zero_grad()
            inputs = batch_inputs.reshape(-1, args.inputs * args.inputs)
            output = model(inputs)
            loss = criterion(output, target=batch_outputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    train_epoch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the CPU offload + sharding with a Transformer training")
    parser.add_argument("--epochs", action="store", default=10, type=int)
    parser.add_argument("--batch_size", action="store", default=20, type=int)
    parser.add_argument("--inputs", action="store", help="The dimension of the inputs", default=100, type=int)
    parser.add_argument("--hidden", action="store", help="The dimension of the hidden state", default=10000, type=int)
    parser.add_argument("--layers", action="store", help="he number of hidden layers", default=200, type=int)
    parser.add_argument("--outputs", action="store", help="The number of predicted classes", default=5, type=int)

    parser.add_argument("--offload", action="store_true", default=False)
    parser.add_argument("--slices", action="store", default=3, type=int)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Benchmark arguments: %s" % args)
    logging.info("Starting training")
    train(args)
