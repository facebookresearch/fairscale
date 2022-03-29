# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
from functools import reduce
import logging
import math
import operator
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from benchmarks.datasets.wikitext2_data import get_real_dataloaders as get_real_wikitext2_dataloaders
from benchmarks.datasets.wikitext2_data import get_synthetic_dataloaders as get_synthetic_wikitext2_dataloaders
from benchmarks.golden_configs.lm_wikitext2 import Offload_Sequential as offload_seq
from benchmarks.golden_configs.lm_wikitext2 import Offload_Transformer as lm_wikitext2
from benchmarks.models import transformer_lm
from fairscale.experimental.nn.offload import OffloadModel


def init_random_seed(seed: int):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_model_and_optimizer(args, device, benchmark_config, model_specs):
    """Return instantiated model and optimizer function."""

    if args.model_name == "lm":
        model = get_lm_model(args, device, model_specs)
        lr = benchmark_config["lr"]

        def make_adam(params):
            return Adam(params, lr=lr)

        optimizer = make_adam
    elif args.model_name == "seq":
        model = get_seq_model(args, device, model_specs)
        optimizer = torch.optim.SGD

    model = OffloadModel(
        model=model,
        device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        num_slices=benchmark_config["slices"],
        checkpoint_activation=benchmark_config["checkpoint_activation"],
        num_microbatches=benchmark_config["num_microbatches"],
    )

    return model, optimizer


def get_seq_model(args, device, model_specs):
    model = torch.nn.Sequential(
        torch.nn.Linear(model_specs["inputs"] * model_specs["inputs"], model_specs["hidden"]),
        *([torch.nn.Linear(model_specs["hidden"], model_specs["hidden"]) for _ in range(model_specs["layers"])]),
        torch.nn.Linear(model_specs["hidden"], model_specs["outputs"]),
    )
    return model.cpu()


def get_lm_model(args, device, config):
    """Get language model(based on GPT-2) used for sequence prediction."""

    ninp = config["ninp"]
    nhead = config["nhead"]
    initrange = config["initrange"]
    dropout = config["dropout"]
    vocab_size = config["vocab_size"]
    nhid = config["nhid"]
    ndecoder = config["num_decoder_layers"]

    return transformer_lm.TransformerLM(vocab_size, ninp, nhead, nhid, dropout, initrange, ndecoder).to(device)


def log_number_of_parameters(model):

    num_params = reduce(operator.add, (reduce(operator.mul, x.size()) for x in model.parameters()))
    logging.info(f"training model, #params = {num_params}")


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


def train_seq(model_config, benchmark_config, model_specs, args):
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    torch.manual_seed(5)

    model = model_config["model"]
    criterion = benchmark_config["criterion"]
    optimizer = model_config["optimizer"](model.parameters(), lr=benchmark_config["lr"])
    dataloader, _, _ = model_config["data"]

    def train_epoch(args, num_iters):
        model.train()
        for batch_inputs, batch_outputs in dataloader:
            batch_inputs, batch_outputs = batch_inputs.to("cuda"), batch_outputs.to("cuda")
            start = time.time_ns()
            with _get_profiler_context(args.use_profiler) as prof:
                optimizer.zero_grad()
                inputs = batch_inputs.reshape(-1, model_specs["inputs"] * model_specs["inputs"])
                with _get_profiler_record_context("model_training", args.use_profiler):
                    with _get_fp16_context(use_fp16=args.use_fp16):
                        output = model(inputs)
                        loss = criterion(output, target=batch_outputs)
                        loss.backward()
                    optimizer.step()
            logging.info(
                "Memory stats are {:.2f}GB".format(torch.cuda.memory_stats(0)["allocated_bytes.all.peak"] / 2**30)
            )
            logging.info(
                "Loss {:.2f} - throughput {:.2f}fps".format(
                    loss.item(), benchmark_config["batch_size"] / (time.time_ns() - start) * 10**9
                )
            )
            num_iters -= 1
            if num_iters == 0:
                break
        if args.use_profiler:
            prof.export_chrome_trace("/tmp/offload_prof")

    train_epoch(args, num_iters=5)


def train(model_config, model, benchmark_config, model_specs, args):
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    lm_dataloader, _, _ = model_config["data"]
    criterion = benchmark_config["criterion"]
    vocab_size = model_specs["vocab_size"]
    optimizer = model_config["optimizer"]

    model.train()
    log_number_of_parameters(model)

    total_loss = 0.0
    word_counter = 0

    optimizer = optimizer(model.parameters())

    total_tokens = 0
    total_tokens_per_log_interval = 0
    bptt = 2
    start_time = time.time()
    epoch_start_time = 0.0

    def get_batch(source):
        seq_len = len(source) - 1
        data = source[0:seq_len]
        target = source[1 : 1 + seq_len]
        return data, target

    for i, batch in enumerate(lm_dataloader):
        # TODO(anj): Make this a flag for both "lm" and "seq" models.
        if i == 5:
            break

        if i == 1:
            epoch_start_time = time.time()

        source, target = get_batch(batch)
        source, target = source.cuda(), target.cuda()

        if i > 0:
            total_tokens += source.numel()

        with _get_profiler_context(args.use_profiler) as prof:
            optimizer.zero_grad()
            with _get_profiler_record_context("FW pass", args.use_profiler):
                output = model(source)
            with _get_profiler_record_context("Loss", args.use_profiler):
                loss = criterion(output.view(-1, vocab_size), target.view(-1))
            with _get_profiler_record_context("BW pass", args.use_profiler):
                loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), model_specs["clip_value"])
            with _get_profiler_record_context("Opt step", args.use_profiler):
                optimizer.step()

        total_loss += loss.item()
        log_interval = 1
        total_tokens_per_log_interval += source.numel()
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(
                "| batch {:5d} | wps {:5.2f} | loss {:5.2f} | ppl {:8.2f}".format(
                    i, total_tokens_per_log_interval / elapsed, cur_loss, math.exp(cur_loss)
                )
            )
            total_tokens_per_log_interval = 0
            total_loss = 0
            start_time = time.time()
        if args.use_profiler:
            prof.export_chrome_trace("/tmp/offload_prof")

    if epoch_start_time != 0:
        wps = total_tokens / (time.time() - epoch_start_time)
    else:
        raise RuntimeError(
            "Unable to benchmark on a single batch. Increase the size " " of the dataset and rerun the benchmark."
        )
    return wps, loss.item()


def verify_peak_memory(golden_config, std_dev):

    current_device_usage = torch.cuda.memory_stats(0)["allocated_bytes.all.peak"]
    golden_ref = golden_config["peak_mem_usage"]
    if not current_device_usage < golden_ref * std_dev:
        raise RuntimeError(
            "Peak memory usage for cuda device {:d} is {:d} which"
            "is less than golden reference value of {:d}".format(0, current_device_usage, golden_ref)
        )


def verify_lm_throughput(wps, golden_config, args):
    """Verify that words per second for a given benchmark run matches the golden data."""

    if not wps > (golden_config["avg_wps"] - (3 * golden_config["std_dev_wps"])):
        raise RuntimeError(
            "Throughput(wps):{:.2f} is below the golden threshold of an "
            "average value of {:.2f} and standard dev of {:.2f}.".format(
                wps, golden_config["avg_wps"], golden_config["std_dev_wps"]
            )
        )


def benchmark_language_model(model_config, model, benchmark_config, model_specs, args):
    epoch = benchmark_config["epochs"]
    start_time = time.time()
    print("-" * 110)
    print("| start of epoch {:1d}".format(epoch))
    print("-" * 110)
    wps, loss = train(model_config, model, benchmark_config, model_specs, args)
    elapsed_time = time.time() - start_time
    print("-" * 110)
    print("| end of epoch {:1d} | time: {:5.2f}s | train loss {:5.2f} ".format(epoch, elapsed_time, loss))
    print("-" * 110)

    if args.model_name == "seq":
        raise RuntimeError(
            f"Golden data verification is only supported for the Transformer(lm) model and not {args.model_name}"
        )
    print("Throughput(wps) is {:.2f}.".format(wps))
    print("Peak allocated bytes on cuda:0: {:1d}".format(torch.cuda.memory_stats(0)["allocated_bytes.all.peak"]))
    if not args.dry_run:
        golden_config = get_golden_config(args.model_name, args)
        verify_lm_throughput(wps, golden_config, args)
        verify_peak_memory(golden_config, 1.1)


def get_synthetic_dataloaders(args, device, benchmark_config, model_specs):
    """Returns dataloader for synthetic data."""

    if args.model_name == "lm":
        return get_synthetic_wikitext2_dataloaders(args, benchmark_config, model_specs)
    elif args.model_name == "seq":
        transform = ToTensor()
        dataloader = DataLoader(
            FakeData(
                image_size=(1, model_specs["inputs"], model_specs["inputs"]),
                num_classes=model_specs["outputs"],
                transform=transform,
            ),
            batch_size=benchmark_config["batch_size"],
        )
        return dataloader, dataloader, dataloader
    else:
        raise RuntimeError(f"Unrecognized args.model_name {args.model_name}")


def get_real_dataloaders(args, device, benchmark_config, model_specs):
    """Returns dataloaders for real data."""

    if args.model_name == "lm":
        data = get_real_wikitext2_dataloaders(args, benchmark_config, model_specs)
        ntokens, train_dataloader, valid_dataloader, test_dataloader = data
        model_specs["vocab_size"] = ntokens
        return train_dataloader, valid_dataloader, test_dataloader
    else:
        raise RuntimeError(f"Unrecognized args.model_mame {args.model_name}")


def create_model_config(args, benchmark_config=None, model_specs=None):
    """Return a dict with the given model, dataset and optimizer."""

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    if args.model_name == "lm":
        if args.use_synthetic_data:
            dataloader_fn = get_synthetic_dataloaders
        else:
            dataloader_fn = get_real_dataloaders

        data = dataloader_fn(args, device, benchmark_config, model_specs)
        model, optimizer = get_model_and_optimizer(args, device, benchmark_config, model_specs)
        return {
            "model": model,
            "optimizer": optimizer,
            "data": data,
        }
    elif args.model_name == "seq":

        data = get_synthetic_dataloaders(
            args, device, offload_seq.get_benchmark_config(), offload_seq.get_model_config()
        )
        model, optimizer = get_model_and_optimizer(args, device, benchmark_config, model_specs)
        return {
            "model": model,
            "optimizer": optimizer,
            "data": data,
        }
    else:
        raise RuntimeError(f"Unrecognized args.model_mame {args.model_name}")


def create_benchmark_config(args):
    """Return a dict with configurations required for benchmarking `model_name` model."""

    if args.model_name == "lm":
        return lm_wikitext2.get_benchmark_config(checkpoint_activation=args.checkpoint_activation)
    elif args.model_name == "seq":
        return offload_seq.get_benchmark_config()
    else:
        raise RuntimeError(f"Unrecognized args.model_name {args.model_name}")


def get_golden_config(model_name, args):
    """Return a dict with the golden data for throughput and memory usage."""

    if model_name == "lm":
        return lm_wikitext2.get_golden_real_stats()
    else:
        raise RuntimeError(f"Unrecognized args.model_mame {args.model_name}")


def get_model_specs(model_name):
    """Return a dict with configurations required for configuring `model_name` model."""

    if model_name == "lm":
        return lm_wikitext2.get_model_config()
    elif model_name == "seq":
        return offload_seq.get_model_config()
    else:
        raise RuntimeError("Unrecognized args.model_mame " % args.model_name)


def run_benchmark(args):
    """Benchmark a given model using a single process and single devices."""

    # We need at least 1 GPU to benchmark the offload model API.
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    assert num_devices > 0
    init_random_seed(0)

    if args.model_name == "lm":
        benchmark_config = create_benchmark_config(args)
        model_specs = get_model_specs(args.model_name)
        model_config = create_model_config(args, benchmark_config=benchmark_config, model_specs=model_specs)
        model = model_config["model"]

        benchmark_language_model(model_config, model, benchmark_config, model_specs, args)

    elif args.model_name == "seq":
        benchmark_config = create_benchmark_config(args)
        model_specs = get_model_specs(args.model_name)
        model_config = create_model_config(args, benchmark_config=benchmark_config, model_specs=model_specs)
        model = model_config["model"]
        train_seq(model_config, benchmark_config, model_specs, args)
    else:
        raise RuntimeError(f"Unable to recognize model name {args.model_name}")


parser = argparse.ArgumentParser(description="benchmark")
parser.add_argument(
    "--dry_run", default=False, action="store_true", help="Run a sample training run without regression testing."
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=True,
    help="Print debugging statements which is more verbose than the default.",
)
parser.add_argument(
    "--model_name",
    default="lm",
    type=str,
    help="Language Model(LM) used to benchmark nn.pipe.",
)
parser.add_argument(
    "--use_synthetic_data", default=True, action="store_true", help="Uses synthetic data for running benchmarks."
)
parser.add_argument("--use_fp16", action="store_true", default=False)
parser.add_argument("--checkpoint_activation", action="store_true", default=False)
parser.add_argument("--use_profiler", action="store_true", default=False)


if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)
    logging.info("Benchmark arguments: %s" % args)

    run_benchmark(args)
