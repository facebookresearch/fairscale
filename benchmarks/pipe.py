# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse
from collections import defaultdict
from functools import reduce
import gc
import logging
import math
import operator
import pprint
import time

from datasets.wikitext2_data import get_real_dataloaders as get_real_wikitext2_dataloaders
from datasets.wikitext2_data import get_synthetic_dataloaders as get_synthetic_wikitext2_dataloaders
from golden_configs import lm_wikitext2
from models import transformer_lm
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import rpc
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam

from fairscale.nn import Pipe
from fairscale.nn.model_parallel import initialize_model_parallel
from fairscale.nn.model_parallel.initialize import get_data_parallel_group, get_pipeline_parallel_group
from fairscale.nn.pipe import LazyModule, MultiProcessPipe
from fairscale.optim.oss import OSS
from fairscale.utils.testing import dist_init, get_worker_map

MPI_PORT = 29500
RPC_PORT = 29501


def init_random_seed(seed: int):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_model_and_optimizer(args, device, config):
    """Return instantiated model and optimizer function."""

    if args.model_name == "lm":
        model = get_lm_model(args, device, config)

    lr = config["lr"]

    def make_adam(params):
        if args.ddp_zero:
            return OSS(params=params, optim=Adam, group=get_data_parallel_group(), lr=lr)
        else:
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


def get_tensors_by_size_bucket():

    size_buckets = defaultdict(int)
    for obj in gc.get_objects():
        if not isinstance(obj, torch.Tensor):
            continue
        if obj.device.type == "cuda":
            size_buckets[(*obj.size(),) + (obj.element_size(),)] += 1

    return size_buckets


def dump_size_buckets(size_buckets, prefix=""):

    total = 0
    for key, value in size_buckets.items():
        this = reduce(operator.mul, key) * value
        total += this
        print(prefix + f"{key} : {value}, {this}")

    print(prefix + f"total = {total}")


last_size_buckets = None
once = True


def safe_rank():
    try:
        return torch.distributed.get_rank()
    except AssertionError:
        return 0


def check_size_buckets():
    global last_size_buckets
    global once
    size_buckets = get_tensors_by_size_bucket()
    if last_size_buckets is not None:
        if size_buckets != last_size_buckets:
            print(f"difference is oustanding tensors: {safe-rank()}")
            dump_size_buckets(last_size_buckets, "old: ")
            dump_size_buckets(size_buckets, "new: ")
        if once:
            print(f"dumping buckets for: {safe_rank()}")
            dump_size_buckets(last_size_buckets, "old: ")
            dump_size_buckets(size_buckets, "new: ")
            once = False
    else:
        print(f"size buckets none on {safe_rank()}")
    last_size_buckets = size_buckets


def dump_cuda_tensors():
    print(f"dumping cuda tensors...")

    for obj in gc.get_objects():
        if not isinstance(obj, torch.Tensor):
            continue
        if obj.device.type == "cuda":
            size_buckets[(*obj.size(),) + (obj.element_size(),)] += 1

    print(f"outstanding cuda tensors:")
    total = 0
    for key, value in size_buckets.items():
        this = reduce(operator.mul, key) * value
        total += this
        print(f"{key} : {value}, {this}")
    print(f"total size = {total}")
    pprint.pprint(torch.cuda.memory_stats())


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

    pipe_group = model.group if hasattr(model, "group") else None

    if args.ddp_zero:
        model = DDP(
            model,
            device_ids=[torch.cuda.current_device()],
            process_group=get_data_parallel_group(),
            find_unused_parameters=False,
        )

    # TODO(anj-s): Avoid sending fake data to all replicas except the first and last one.
    if pipe_group and pipe_group.rank() != 0 and pipe_group.rank() != (pipe_group.size() - 1):
        lm_dataloader, _, _ = get_synthetic_dataloaders(args, benchmark_config)

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
            if (pipe_group is None or pipe_group.rank() == 0) and not args.ddp_zero:
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
            if args.ddp_zero:
                ddp_group = get_data_parallel_group()
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM, group=ddp_group)
                loss /= ddp_group.size()
            loss.backward()
            del target
        else:
            if args.ddp_zero:
                model.module.back_helper(output)
            else:
                model.back_helper(output)

        del output

        torch.nn.utils.clip_grad_value_(model.parameters(), benchmark_config["clip_value"])
        optimizer.step()

        if pipe_group is None or pipe_group.rank() == pipe_group.size() - 1:
            total_loss += loss.item()
            log_interval = 1
            total_tokens_per_log_interval += source.numel()
            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                if not args.multiprocess or dist.get_rank() == dist.get_world_size() - 1:
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
    if not args.multiprocess or dist.get_rank() == dist.get_world_size() - 1:
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

    if len(model.balance) == 4:
        if args.model_name == "lm":
            verify_lm_run(wps, golden_config, args)
        else:
            raise RuntimeError("Unrecognized args.model_name " % args.model_name)


def generate_balance_weighted(num_devices, num_layers, fraction=0.5):
    balance = []
    layers_assigned = 0
    average_count = num_layers / num_devices
    last_layers = int(average_count * fraction)

    balance = generate_balance(num_devices - 1, num_layers - last_layers)
    balance.append(last_layers)
    return balance


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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
        return lm_wikitext2.get_golden_real_stats(args.multiprocess)
    else:
        raise RuntimeError("Unrecognized args.model_mame " % args.model_name)


def benchmark_single_process(args):
    """Benchmark a given model using a single process and multiple devices."""

    init_method_pgroup = "tcp://localhost:{}".format(MPI_PORT)
    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1, init_method=init_method_pgroup)

    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert num_devices > 0
    init_random_seed(0)

    benchmark_config = create_benchmark_config(args.model_name)
    model_config = create_model_config(args, benchmark_config=benchmark_config)
    model = model_config["model"]

    balance = generate_balance(min(num_devices, 4), len(model))
    pipe_model = Pipe(model, balance, chunks=args.chunks, checkpoint=args.checkpoint)
    del model
    del model_config["model"]

    if args.dry_run:
        train(model_config, pipe_model, benchmark_config, args)
    else:
        benchmark_language_model(model_config, pipe_model, benchmark_config, args)


def run_mp_worker(args, available_workers):

    benchmark_config = create_benchmark_config(args.model_name)
    model_config = create_model_config(args, benchmark_config=benchmark_config)
    model = model_config["model"]

    balance = generate_balance(get_pipeline_parallel_group().size(), len(model))
    pipe_model = MultiProcessPipe(
        model,
        balance,
        chunks=args.chunks,
        worker_map=get_worker_map(),
        input_device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        checkpoint=args.checkpoint,
        # TODO(anj-s): Do we need to comment this out? loss_fn=benchmark_config["criterion"],
    )
    if torch.cuda.is_available():
        pipe_model = pipe_model.cuda()
    if args.all_at_once and pipe_model.pipeline:
        print(f"running all at once")
        pipe_model.pipeline.all_at_once = True

    if args.dry_run:
        train(model_config, pipe_model, benchmark_config, args)
    else:
        benchmark_language_model(model_config, pipe_model, benchmark_config, args)


def run_worker(rank, world_size, args):
    if args.world_size != 0:
        world_size = args.world_size
    dist_init(rank + args.rank_base, world_size, hostname=args.host)
    initialize_model_parallel(1, world_size)
    init_random_seed(0)
    run_mp_worker(args, world_size)

    rpc.shutdown()
    torch.distributed.destroy_process_group()


def benchmark_multiprocess(rank, world_size, args):

    init_method_pgroup = "tcp://localhost:{}".format(MPI_PORT)
    # TODO(anj-s): Add regression benchmarks for nccl as well.
    torch.distributed.init_process_group(
        backend="gloo", rank=rank, world_size=world_size, init_method=init_method_pgroup
    )

    torch.cuda.set_device(rank % torch.cuda.device_count())
    # TODO(anj-s): Move to TensorPipeRpcBackendOptions.
    rpc.init_rpc(
        f"Test{rank}",
        rank=rank,
        world_size=world_size,
        backend=rpc.BackendType.PROCESS_GROUP,
        rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
            rpc_timeout=20, init_method="tcp://localhost:{}".format(RPC_PORT)
        ),
    )
    initialize_model_parallel(1, world_size)
    init_random_seed(0)
    run_mp_worker(args, world_size)

    rpc.shutdown()
    torch.distributed.destroy_process_group()


parser = argparse.ArgumentParser(description="benchmark")
parser.add_argument("--multiprocess", action="store_true", help="Runs single process benchmarks.")
parser.add_argument("--host", "-o", type=str, default="localhost", help="hostname")
parser.add_argument("--chunks", type=int, default=1, help="number of microbatches per batch")
parser.add_argument("--batch-size", type=int, default=8, help="size of a batch")
parser.add_argument("--all-at-once", action="store_true", default=False, help="do backward pass on whole batch at once")
parser.add_argument("--max-batch", type=int, default=4, help="Max number of batches")
parser.add_argument("--ddp-zero", action="store_true", default=False, help="enable ddp")
parser.add_argument(
    "--lazy-construction", action="store_true", default=False, help="Number of decoder layers in the model"
)
parser.add_argument(
    "--checkpoint", default="never", choices=["always", "except_last", "never"], help="Checkpointing strategy for pipe"
)
parser.add_argument("--use_synthetic_data", action="store_true", help="Uses synthetic data for running benchmarks.")
parser.add_argument("--dry_run", action="store_true", help="Run a sample training run without regression testing.")
parser.add_argument(
    # TODO(anj-s): In the process of adding more models and hence the requirement for a flag.
    "--model_name",
    default="lm",
    help="Language Model(LM) used to benchmark nn.pipe.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    # TODO(anj-s): Remove print statements and introduce logging levels.

    if not args.multiprocess:
        print(f"Running single process benchmark with args: {args}")
        benchmark_single_process(args)
    else:
        world_size = max(torch.cuda.device_count(), 1)
        print(f"Running multiprocess benchmark with args: {args}")
        mp.spawn(benchmark_multiprocess, args=(world_size, args), nprocs=world_size, join=True)
