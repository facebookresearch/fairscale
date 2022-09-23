# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import gc
import logging
import math
import time

import torch
import torch.distributed as dist
from torch.distributed import rpc
from torch.nn.parallel import DistributedDataParallel as DDP
import utils

from benchmarks.golden_configs.lm_wikitext2 import Pipe as lm_wikitext2
from fairscale.fair_dev.testing.testing import dist_init
from fairscale.nn import Pipe
from fairscale.nn.model_parallel import initialize_model_parallel

MPI_PORT = 29500
RPC_PORT = 29501


def get_tensors_by_size_bucket():

    size_buckets = defaultdict(int)
    for obj in gc.get_objects():
        if not isinstance(obj, torch.Tensor):
            continue
        if obj.device.type == "cuda":
            size_buckets[(*obj.size(),) + (obj.element_size(),)] += 1

    return size_buckets


def get_device(model, index):
    if isinstance(model, DDP):
        model = model.module

    if not torch.cuda.is_available():
        return torch.device("cpu")
    if hasattr(model, "devices"):
        return model.devices[index]
    else:
        return torch.cuda.current_device()


def get_fake_dataloader(lm_dataloader_len, args):
    fake_input = {"input": torch.zeros(args.batch_size)}

    class FakeDataset:
        def __getitem__(self, index):
            return fake_input

        def __len__(self):
            return lm_dataloader_len

    return FakeDataset()


def train(model_config, model, benchmark_config, model_specs, args):
    lm_dataloader, _, _ = utils.get_data_loader(model_config["dataset_info"], args, benchmark_config, model_specs)
    criterion = benchmark_config["criterion"]
    vocab_size = model_specs["vocab_size"]
    optimizer = model_config["optimizer"]

    model.train()
    utils.log_number_of_parameters(model)

    total_loss = 0.0
    word_counter = 0

    optimizer = optimizer(model.parameters())

    pipe_group = model.group if hasattr(model, "group") else None

    # TODO(anj-s): Avoid sending fake data to all replicas except the first and last one.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if pipe_group and pipe_group.rank() != 0 and pipe_group.rank() != (pipe_group.size() - 1):
        lm_dataloader, _, _ = get_synthetic_dataloaders(args, benchmark_config, model_specs)

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
        if args.max_batch and i > args.max_batch:
            break

        if i > 0:
            total_tokens += source.numel()

        optimizer.zero_grad()
        try:
            if pipe_group is None or pipe_group.rank() == 0:
                tmp = source.to(get_device(model, 0))
                output = model(tmp)
            else:
                output = model(source)
        except Exception as e:
            raise RuntimeError(f"training failed on {torch.distributed.get_rank()}") from e

        if pipe_group is None or pipe_group.rank() == pipe_group.size() - 1:
            target = target.to(get_device(model, -1))
            output = output.to(target.device)
            loss = criterion(output.view(-1, vocab_size), target.view(-1))
            loss.backward()
            del target
        else:
            model.back_helper(output)

        del output

        torch.nn.utils.clip_grad_value_(model.parameters(), model_specs["clip_value"])
        optimizer.step()

        if pipe_group is None or pipe_group.rank() == pipe_group.size() - 1:
            total_loss += loss.item()
            log_interval = 1
            total_tokens_per_log_interval += source.numel()
            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                if dist.get_rank() == dist.get_world_size() - 1:
                    logging.debug(
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
    if dist.get_rank() == dist.get_world_size() - 1:
        return wps, loss.item()
    else:
        return 0.0, 0.0


# TODO(anj-s): Add an option for users to be able to benchmark evaluate.
def evaluate(eval_model, data_source, criterion, ntokens):
    eval_model.eval()
    total_loss = 0.0
    # TODO(anj-s): Move this to the benchmark config if we want to benchmark evaluation.
    bptt = 35

    def get_batch(source, i, bptt):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i : i + seq_len]
        target = source[i + 1 : i + 1 + seq_len].view(-1)
        return data, target

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            output = eval_model(data)
            output = output.to(targets.device)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def get_number_of_words(data):
    return data.size()[0] * data.size()[1]


def verify_peak_memory(rank, golden_config, std_dev):
    logging.debug(
        "Peak allocated bytes on cuda:0: {:1d}".format(torch.cuda.memory_stats(rank)["allocated_bytes.all.peak"])
    )
    current_device_usage = torch.cuda.memory_stats(rank)["allocated_bytes.all.peak"]
    golden_ref = golden_config["peak_mem_usage"][rank]
    if not current_device_usage < golden_ref * std_dev:
        raise RuntimeError(
            "Peak memory usage for cuda device {:d} is {:d} which"
            "is less than golden reference value of {:d}".format(rank, current_device_usage, golden_ref)
        )


def verify_lm_run(wps, golden_config, args):
    """Verify that words per second for a given benchmark run matches the golden data."""

    if dist.get_rank() == dist.get_world_size() - 1:
        # Assert that words per second is within 3 standard deviations of the average
        # of five golden runs
        logging.info("Throughput(wps) is {:.2f}.".format(wps))
        if not wps > (golden_config["avg_wps"] - (3 * golden_config["std_dev_wps"])):
            raise RuntimeError(
                "Throughput(wps):{:.2f} is below the golden threshold of an "
                "average value of {:.2f} and standard dev of {:.2f}.".format(
                    wps, golden_config["avg_wps"], golden_config["std_dev_wps"]
                )
            )

    for i in range(4):
        verify_peak_memory(i, golden_config, 1.1)


def benchmark_language_model(model_config, model, benchmark_config, model_specs, config_class, args):
    golden_config = get_golden_config(args.model_name, config_class, args)
    epoch = benchmark_config["epochs"]
    start_time = time.time()
    if dist.get_rank() == dist.get_world_size() - 1:
        logging.debug("-" * 110)
        logging.debug("| start of epoch {:1d}".format(epoch))
        logging.debug("-" * 110)
    wps, loss = train(model_config, model, benchmark_config, model_specs, args)
    elapsed_time = time.time() - start_time
    if dist.get_rank() == dist.get_world_size() - 1:
        logging.debug("-" * 110)
        logging.debug("| end of epoch {:1d} | time: {:5.2f}s | train loss {:5.2f} ".format(epoch, elapsed_time, loss))
        logging.debug("-" * 110)
        logging.debug("Throughput(wps) is {:.2f}.".format(wps))
    logging.debug(
        "Peak allocated bytes on cuda:{}: {:1d}".format(
            dist.get_rank(), torch.cuda.memory_stats(dist.get_rank())["allocated_bytes.all.peak"]
        )
    )

    if len(model.balance) == 4:
        if args.model_name == "lm":
            verify_lm_run(wps, golden_config, args)
        else:
            raise RuntimeError("Unrecognized args.model_name " % args.model_name)


def generate_balance(num_devices, num_layers):
    balance = []
    layers_assigned = 0
    for i in range(num_devices):
        x = (num_layers - layers_assigned) / (num_devices - i)
        if x.is_integer():
            balance.append(int(x))
            layers_assigned += x
        else:
            balance.append(math.ceil(x))
            layers_assigned += math.ceil(x)
    return balance


def get_golden_config(model_name, config_class, args):
    """Return a dict with the golden data for throughput and memory usage."""

    if model_name == "lm":
        return config_class.get_golden_real_stats()
    else:
        raise RuntimeError("Unrecognized args.model_mame " % args.model_name)


def benchmark_single_process(config_class, args):
    """Benchmark a given model using a single process and multiple devices."""

    init_method_pgroup = "tcp://localhost:{}".format(MPI_PORT)
    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1, init_method=init_method_pgroup)

    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert num_devices > 0
    utils.init_random_seed(0)

    benchmark_config = utils.create_benchmark_config(args.model_name, config_class)
    model_specs = utils.get_model_specs(args.model_name, config_class)
    model_config = utils.create_model_config(args, benchmark_config=benchmark_config, model_specs=model_specs)
    model = model_config["model"]

    balance = generate_balance(min(num_devices, 4), len(model))
    pipe_model = Pipe(model, balance, chunks=args.chunks, checkpoint=args.checkpoint)
    del model
    del model_config["model"]

    if args.dry_run:
        train(model_config, pipe_model, benchmark_config, model_specs, args)
    else:
        benchmark_language_model(model_config, pipe_model, benchmark_config, model_specs, config_class, args)


def run_worker(rank, world_size, args):
    if args.world_size != 0:
        world_size = args.world_size
    dist_init(rank + args.rank_base, world_size, hostname=args.host)
    initialize_model_parallel(1, world_size)
    utils.init_random_seed(0)
    run_mp_worker(args, world_size)

    rpc.shutdown()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = utils.init_args()
    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)

    logging.info(f"Running single process benchmark with args: {args}")
    benchmark_single_process(lm_wikitext2, args)
