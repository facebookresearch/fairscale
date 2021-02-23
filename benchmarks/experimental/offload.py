# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse
from functools import reduce
import logging
import math
import operator
import time

from benchmarks.datasets.wikitext2_data import get_real_dataloaders as get_real_wikitext2_dataloaders
from benchmarks.datasets.wikitext2_data import get_synthetic_dataloaders as get_synthetic_wikitext2_dataloaders
from benchmarks.golden_configs import lm_wikitext2
from benchmarks.models import transformer_lm
import numpy as np
import torch
from torch.optim import Adam

from fairscale.experimental.nn.offload import OffloadModel
from fairscale.nn.pipe import LazyModule


def init_random_seed(seed: int):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_model_and_optimizer(args, device, config):
    """Return instantiated model and optimizer function."""

    if args.model_name == "lm":
        model = get_lm_model(args, device, config)

    model = OffloadModel(
        model_cpu=model,
        device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        n_slices=3,
        checkpoint_activation=True,
        num_microbatches=4,
    )

    lr = config["lr"]

    def make_adam(params):
        return Adam(params, lr=lr)

    optimizer = make_adam
    return model, optimizer


def get_lm_model(args, device, config):
    """Get language model(based on GPT-2) used for sequence prediction."""

    ninp = config["ninp"]
    nhead = config["nhead"]
    initrange = config["initrange"]
    dropout = config["dropout"]
    vocab_size = config["vocab_size"]
    nhid = config["nhid"]
    ndecoder = config["num_decoder_layers"]

    if args.lazy_construction:
        layers = [
            LazyModule(lambda: transformer_lm.EmbeddingLayer(vocab_size, ninp, initrange)),
            LazyModule(lambda: transformer_lm.PositionalEncodingLayer(ninp, dropout)),
        ]
        for _ in range(ndecoder):
            layers.append(LazyModule(lambda: transformer_lm.TransformerDecoderLayer(ninp, nhead, nhid, dropout)))

        layers.append(LazyModule(lambda: transformer_lm.LinearLayer(ninp, vocab_size, initrange)))
        model = layers
    else:
        model = transformer_lm.TransformerLM(vocab_size, ninp, nhead, nhid, dropout, initrange, ndecoder).to(device)

    return model


def log_number_of_parameters(model):

    num_params = reduce(operator.add, (reduce(operator.mul, x.size()) for x in model.parameters()))
    if hasattr(model, "group"):
        total = torch.Tensor([num_params])
        if torch.cuda.is_available():
            total = total.cuda()
        torch.distributed.all_reduce(total, group=model.group)
        logging.info(
            f"training model, #params = {num_params}, group: {model.group.rank()}, grank:"
            f" {torch.distributed.get_rank()}, sizes {model.group.size()}"
        )
        torch.distributed.barrier()
        if model.group.rank() == 0:
            logging.info(f"total #prams = {total.item()}")
    else:
        logging.info(f"training model, #params = {num_params}")


def train(model_config, model, benchmark_config, args):
    lm_dataloader, _, _ = model_config["data"]
    criterion = benchmark_config["criterion"]
    vocab_size = benchmark_config["vocab_size"]
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
        if i == 1:
            epoch_start_time = time.time()

        source, target = get_batch(batch)

        if i > 0:
            total_tokens += source.numel()

        optimizer.zero_grad()
        output = model(source)

        target = target.to("cuda")
        output = output.to(target.device)
        loss = criterion(output.view(-1, vocab_size), target.view(-1))
        loss.backward()
        # del target
        # del output

        # torch.nn.utils.clip_grad_value_(model.parameters(), benchmark_config["clip_value"])
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
    if epoch_start_time != 0:
        wps = total_tokens / (time.time() - epoch_start_time)
    else:
        raise RuntimeError(
            "Unable to benchmark on a single batch. Increase the size " " of the dataset and rerun the benchmark."
        )
    return wps, loss.item()


def verify_peak_memory(rank, golden_config, std_dev):
    print("Peak allocated bytes on cuda:0: {:1d}".format(torch.cuda.memory_stats(rank)["allocated_bytes.all.peak"]))
    current_device_usage = torch.cuda.memory_stats(rank)["allocated_bytes.all.peak"]
    golden_ref = golden_config["peak_mem_usage"][rank]
    if not current_device_usage < golden_ref * std_dev:
        raise RuntimeError(
            "Peak memory usage for cuda device {:d} is {:d} which"
            "is less than golden reference value of {:d}".format(rank, current_device_usage, golden_ref)
        )


def verify_lm_run(wps, golden_config, args):
    """Verify that words per second for a given benchmark run matches the golden data."""

    # Verify wps only on the last rank in multiprocess pipe
    if not args.multiprocess or dist.get_rank() == dist.get_world_size() - 1:
        # Assert that words per second is within 3 standard deviations of the average
        # of five golden runs
        print("Throughput(wps) is {:.2f}.".format(wps))
        if not wps > (golden_config["avg_wps"] - (3 * golden_config["std_dev_wps"])):
            raise RuntimeError(
                "Throughput(wps):{:.2f} is below the golden threshold of an "
                "average value of {:.2f} and standard dev of {:.2f}.".format(
                    wps, golden_config["avg_wps"], golden_config["std_dev_wps"]
                )
            )

    if args.multiprocess:
        verify_peak_memory(dist.get_rank(), golden_config, 1.5)
    else:
        for i in range(4):
            verify_peak_memory(i, golden_config, 1.1)


def benchmark_language_model(model_config, model, benchmark_config, args):
    golden_config = get_golden_config(args.model_name, args)
    epoch = benchmark_config["epochs"]
    start_time = time.time()
    if dist.get_rank() == dist.get_world_size() - 1:
        print("-" * 110)
        print("| start of epoch {:1d}".format(epoch))
        print("-" * 110)
    wps, loss = train(model_config, model, benchmark_config, args)
    elapsed_time = time.time() - start_time
    if dist.get_rank() == dist.get_world_size() - 1:
        print("-" * 110)
        print("| end of epoch {:1d} | time: {:5.2f}s | train loss {:5.2f} ".format(epoch, elapsed_time, loss))
        print("-" * 110)
        print("Throughput(wps) is {:.2f}.".format(wps))
    print(
        "Peak allocated bytes on cuda:{}: {:1d}".format(
            dist.get_rank(), torch.cuda.memory_stats(dist.get_rank())["allocated_bytes.all.peak"]
        )
    )

    if args.model_name == "lm":
        verify_lm_run(wps, golden_config, args)
    else:
        raise RuntimeError("Unrecognized args.model_name " % args.model_name)


def get_synthetic_dataloaders(args, benchmark_config):
    """Returns dataloader for synthetic data."""

    if args.model_name == "lm":
        return get_synthetic_wikitext2_dataloaders(args, benchmark_config)
    else:
        raise RuntimeError("Unrecognized args.model_mame " % args.model_name)


def get_real_dataloaders(args, device, benchmark_config):
    """Returns dataloaders for real data."""

    if args.model_name == "lm":
        data = get_real_wikitext2_dataloaders(args, benchmark_config)
        ntokens, train_dataloader, valid_dataloader, test_dataloader = data
        benchmark_config["vocab_size"] = ntokens
        return train_dataloader, valid_dataloader, test_dataloader
    else:
        raise RuntimeError("Unrecognized args.model_mame " % args.model_name)


def create_model_config(args, benchmark_config=None):
    """Return a dict with the given model, dataset and optimizer."""

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    if args.use_synthetic_data:
        dataloader_fn = get_synthetic_dataloaders
    else:
        dataloader_fn = get_real_dataloaders

    data = dataloader_fn(args, device, benchmark_config)
    model, optimizer = get_model_and_optimizer(args, device, benchmark_config)
    return {
        "model": model,
        "optimizer": optimizer,
        "data": data,
    }


def create_benchmark_config(model_name):
    """Return a dict with configurations required for benchmarking `model_name` model."""

    if model_name == "lm":
        return lm_wikitext2.get_benchmark_config()
    else:
        raise RuntimeError("Unrecognized args.model_mame " % args.model_name)


def get_golden_config(model_name, args):
    """Return a dict with the golden data for throughput and memory usage."""

    if model_name == "lm":
        return lm_wikitext2.get_golden_real_stats(False)
    else:
        raise RuntimeError("Unrecognized args.model_mame " % args.model_name)


def benchmark_single_process(args):
    """Benchmark a given model using a single process and single devices."""

    # We need at least 1 GPU to benchmark the offload model API.
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    assert num_devices > 0
    init_random_seed(0)

    benchmark_config = create_benchmark_config(args.model_name)
    model_config = create_model_config(args, benchmark_config=benchmark_config)
    model = model_config["model"]

    if args.dry_run:
        train(model_config, model, benchmark_config, args)
    else:
        benchmark_language_model(model_config, model, benchmark_config, args)


parser = argparse.ArgumentParser(description="benchmark")
parser.add_argument(
    "--lazy-construction", action="store_true", default=False, help="Number of decoder layers in the model"
)
parser.add_argument("--dry_run", action="store_true", help="Run a sample training run without regression testing.")
parser.add_argument(
    "--debug", action="store_true", help="Print debugging statements which is more verbose than the default."
)
parser.add_argument(
    "--model_name",
    default="lm",
    help="Language Model(LM) used to benchmark nn.pipe.",
)
parser.add_argument("--use_synthetic_data", action="store_true", help="Uses synthetic data for running benchmarks.")
parser.add_argument("--batch-size", type=int, default=2, help="size of a batch")


if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)
    logging.info("Benchmark arguments: %s" % args)

    benchmark_single_process(args)