# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse
from collections import defaultdict
from functools import reduce
import gc
import logging
import math
import operator
import os
import pprint
import time

from benchmark_dataset import BenchmarkLMDataset, collate_sentences_lm
import datasets
import models
import numpy
import torch
from torch.distributed import rpc
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from fairscale.nn import Pipe
from fairscale.nn.model_parallel import initialize_model_parallel
from fairscale.nn.model_parallel.initialize import get_data_parallel_group, get_pipeline_parallel_group
from fairscale.nn.pipe import LazyModule, pipe
from fairscale.optim import GradScaler
from fairscale.optim.oss import OSS
from fairscale.utils.testing import dist_init, get_worker_map

try:
    from fairscale.optim import Adam  # type: ignore

    can_benchmark = True
except ImportError:
    from torch.optim import Adam  # type: ignore

    can_benchmark = False


def init_random_seed(seed: int):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)


def make_model(args, device, config):
    ninp = config["ninp"]
    nhead = config["nhead"]
    initrange = config["initrange"]
    dropout = config["dropout"]
    vocab_size = config["vocab_size"]
    nhid = config["nhid"]
    lr = config["lr"]
    ndecoder = args.num_decoder_layers

    if args.lazy_construction:
        layers = [
            LazyModule(lambda: models.EmbeddingLayer(vocab_size, ninp, initrange)),
            LazyModule(lambda: models.PositionalEncodingLayer(ninp, dropout)),
        ]
        for _ in range(ndecoder):
            layers.append(LazyModule(lambda: models.TransformerDecoderLayer(ninp, nhead, nhid, dropout)))

        layers.append(LazyModule(lambda: models.LinearLayer(ninp, vocab_size, initrange)))
        model = layers
    else:
        model = models.TransformerLMSequntial(vocab_size, ninp, nhead, nhid, dropout, initrange, ndecoder).to(device)

    def make_adam(params):
        if args.ddp_zero:
            return OSS(params=params, optim=Adam, group=get_data_parallel_group(), lr=lr)
        else:
            return Adam(params, lr=lr)

    optimizer = make_adam
    return model, optimizer


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
    if model.group:
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
    if model.devices:
        return model.devices[index]
    else:
        return torch.cuda.current_device()


def get_fake_dataloader(lm_dataloader_len):
    fake_input = {"input": torch.zeros(args.batch_size)}

    class FakeDataset:
        def __getitem__(self, index):
            return fake_input

        def __len__(self):
            return lm_dataloader_len

    return FakeDataset()


def train(data_config, model, benchmark_config, args):
    lm_dataloader = data_config["data"]
    criterion = benchmark_config["criterion"]
    vocab_size = benchmark_config["vocab_size"]
    optimizer = data_config["optimizer"]

    model.train()
    log_number_of_parameters(model)

    total_loss = 0.0
    start_time = time.time()
    word_counter = 0

    optimizer = optimizer(model.parameters())

    pipe_group = model.group

    if args.ddp_zero:
        model = DDP(
            model,
            device_ids=[torch.cuda.current_device()],
            process_group=get_data_parallel_group(),
            find_unused_parameters=False,
        )

    # TODO(anj-s): Avoid sending fake data to all replicas except the first and last one.
    if pipe_group and pipe_group.rank() != 0 and pipe_group.rank() != (pipe_group.size() - 1):
        lm_dataloader = get_fake_dataloader(len(lm_dataloader))

    for i, batch in enumerate(lm_dataloader):
        if args.max_batch and i > args.max_batch:
            break
        optimizer.zero_grad()
        try:
            if (pipe_group is None or pipe_group.rank() == 0) and not args.ddp_zero:
                tmp = batch["input"].to(get_device(model, 0))
                output = model(tmp)
            else:
                output = model(batch["input"])
        except Exception as e:
            raise RuntimeError(f"training failed on {torch.distributed.get_rank()}") from e

        if pipe_group is None or pipe_group.rank() == pipe_group.size() - 1:
            target = batch["target"].to(get_device(model, -1))
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
            word_counter += batch["ntokens"]
            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print(
                    "| batch {:5d} | wps {:5.2f} | loss {:5.2f} | ppl {:8.2f}".format(
                        i, word_counter / elapsed, cur_loss, math.exp(cur_loss)
                    )
                )
                word_counter = 0
                total_loss = 0
                start_time = time.time()


def evaluate(eval_model, data_source, criterion, bptt, ntokens):
    eval_model.eval()
    total_loss = 0.0

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


def benchmark_language_model(model_config, model, benchmark_config, args):
    ntokens, train_data, val_data, test_data = model_config["data"]
    optimizer = model_config["optimizer"]
    criterion = benchmark_config["criterion"]
    epoch = 1
    bptt = 35
    start_time = time.time()

    print("-" * 110)
    print("| start of epoch {:1d}".format(epoch))
    print("-" * 110)
    epoch_start_time = time.time()
    train(train_data, model, criterion, optimizer, bptt, ntokens, args)
    val_loss = 1  # evaluate(model, val_data, criterion, bptt, ntokens)
    print("-" * 89)
    print(
        "| end of epoch {:1d} | time: {:5.2f}s | valid loss {:5.2f} ".format(
            epoch, (time.time() - epoch_start_time), val_loss
        )
    )
    print("-" * 110)

    elapsed_time = time.time() - start_time
    nwords = get_number_of_words(train_data) + get_number_of_words(val_data)
    wps = nwords / elapsed_time

    test_loss = 1  # evaluate(model, test_data, criterion, bptt, ntokens)
    print("=" * 89)
    print(
        "| end of training | test loss {:5.2f} \n| time: {:5.2f}s | words: {:3d} | wps: {:5.2f}".format(
            test_loss, elapsed_time, nwords, wps
        )
    )
    print("=" * 110)

    if can_benchmark and len(model.balance) == 4:
        # Assert that words per second is within 3 standard deviations of the average
        # of six golden runs
        assert wps > 36954.4 - (3 * 116.825)

        print("Peak allocated bytes on cuda:0: {:1d}".format(torch.cuda.memory_stats(0)["allocated_bytes.all.peak"]))
        print("Peak allocated bytes on cuda:1: {:1d}".format(torch.cuda.memory_stats(1)["allocated_bytes.all.peak"]))
        print("Peak allocated bytes on cuda:2: {:1d}".format(torch.cuda.memory_stats(2)["allocated_bytes.all.peak"]))
        print("Peak allocated bytes on cuda:3: {:1d}".format(torch.cuda.memory_stats(3)["allocated_bytes.all.peak"]))

        # Assert that memory usage on each GPU is within 10% of golden run
        # Right-hand-side is golden run bytes * 110%
        assert torch.cuda.memory_stats(0)["allocated_bytes.all.peak"] < 4061909504 * 1.1
        assert torch.cuda.memory_stats(1)["allocated_bytes.all.peak"] < 4050944 * 1.1
        assert torch.cuda.memory_stats(2)["allocated_bytes.all.peak"] < 10427392 * 1.1
        assert torch.cuda.memory_stats(3)["allocated_bytes.all.peak"] < 2031824896 * 1.1
        print("No regression detected")


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


def make_model_and_data(args, config=None):
    """Return a dict with the given model, dataset and optimizer."""

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.use_synthetic_data:
        model, optimizer = make_model(args, device, config)
        lm_dataset = BenchmarkLMDataset()
        lm_dataloader = DataLoader(
            lm_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_sentences_lm
        )
        return {"model": model, "optimizer": optimizer, "data": lm_dataloader}
    else:
        data = datasets.get_wikitext2_data(device)
        ntokens, _, _, _ = data
        config["vocab_size"] = ntokens
        model, optimizer = make_model(args, device, ntokens)
        return {
            "model": model,
            "optimizer": optimizer,
            "data": data,
        }


def create_benchmark_config(model_name):
    """Return a dict with configurations required for benchmarking `model_name` model."""

    if model_name == "seq_pred":
        return {
            "vocab_size": 10000,
            "ninp": 2048,  # embedding dimension
            "nhid": 2048,  # the dimension of the feedforward network model in nn.TransformerEncoder
            "nhead": 32,  # the number of heads in the multiheadattention models
            "dropout": 0,
            "initrange": 0.1,
            "criterion": nn.CrossEntropyLoss(),
            "lr": 0.01,  # learning rate
            "scaler": GradScaler(),
            "clip_value": 0.05,
        }


def benchmark_single_process(args):
    """Benchmark a given model using a single process and multiple devices."""

    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert num_devices > 0
    init_random_seed(0)

    benchmark_config = create_benchmark_config(args.model_name)
    model_config = make_model_and_data(args, config=benchmark_config)
    model = model_config["model"]

    balance = generate_balance(min(num_devices, 4), len(model))
    p = pipe.Pipe(
        model, balance, chunks=args.chunks, pipelined_backward=args.pipelined_backward, checkpoint=args.checkpoint
    )
    del model
    del model_config["model"]

    if args.use_synthetic_data:
        train(model_config, p, benchmark_config, args)
    else:
        benchmark_language_model(model_config, p, benchmark_config, args)


def run_mp_worker(args, available_workers):

    benchmark_config = create_benchmark_config(args.model_name)
    model_config = make_model_and_data(args, config=benchmark_config)
    model = model_config["model"]

    balance = generate_balance_weighted(get_pipeline_parallel_group().size(), len(model), 0.8)
    p = pipe.Pipe(
        model,
        balance,
        style=Pipe.AsyncSchedule,
        chunks=args.chunks,
        worker_map=get_worker_map(),
        input_device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        pipelined_backward=args.pipelined_backward,
        checkpoint=args.checkpoint,
        # TODO(anj-s): Do we need to comment this out? loss_fn=benchmark_config["criterion"],
    )
    if torch.cuda.is_available():
        p = p.cuda()
    if args.all_at_once and p.pipeline:
        print(f"running all at once")
        p.pipeline.all_at_once = True

    if args.use_synthetic_data:
        train(model_config, p, benchmark_config, args)
    else:
        benchmark_language_model(model_config, p, benchmark_config, args)


def run_worker(rank, world_size, args):
    if args.world_size != 0:
        world_size = args.world_size
    dist_init(rank + args.rank_base, world_size, hostname=args.host)
    initialize_model_parallel(1, world_size)
    init_random_seed(0)
    run_mp_worker(args, world_size)

    rpc.shutdown()
    torch.distributed.destroy_process_group()


def bench_multi_process(args, all_at_once=False):
    if args.local_world_size != 0:
        world_size = args.local_world_size
    else:
        world_size = min(torch.cuda.device_count(), 2)
    mp.spawn(run_worker, args=(world_size, args), nprocs=world_size, join=True)


best_device_map = {
    0: "mlx5_0:1",
    1: "mlx5_0:1",
    2: "mlx5_1:1",
    3: "mlx5_1:1",
    4: "mlx5_2:1",
    5: "mlx5_2:1",
    6: "mlx5_3:1",
    7: "mlx5_3:1",
}


def bench_mpi(args):
    guess_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    os.environ["UCX_NET_DEVICES"] = best_device_map[local_rank]

    os.environ["MASTER_ADDR"] = args.host
    os.environ["MASTER_PORT"] = "10638"
    if args.socket_name:
        os.environ["GLOO_SOCKET_IFNAME"] = args.socket_name
        os.environ["TP_SOCKET_IFNAME"] = args.socket_name

    torch.distributed.init_process_group(backend="gloo", rank=guess_rank, world_size=world_size)

    os.environ["MASTER_ADDR"] = args.host
    os.environ["MASTER_PORT"] = "10639"
    init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank % torch.cuda.device_count())

    rpc.init_rpc(
        f"Test{rank}",
        rank=rank,
        world_size=world_size,
        backend=rpc.BackendType.PROCESS_GROUP,
        rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(rpc_timeout=20, init_method=init_method),
    )

    backends = {"model_parallel_backend": "nccl", "pipeline_backend": "mpi", "ddp_backend": "nccl"}

    if args.ddp_zero:
        initialize_model_parallel(1, 4, **backends)
    else:
        initialize_model_parallel(1, world_size, **backends)
    init_random_seed(0)

    run_mp_worker(args, world_size)

    rpc.shutdown()
    torch.distributed.destroy_process_group()


parser = argparse.ArgumentParser(description="benchmark")
parser.add_argument("--local-world-size", "-l", type=int, default=0, help="local world size")
parser.add_argument("--world-size", "-w", type=int, default=0, help="world size")
parser.add_argument("--rank-base", "-r", type=int, help="rank base", default=0)
parser.add_argument("--host", "-o", type=str, default="localhost", help="hostname")
parser.add_argument("--no-mpi", action="store_true", default=False, help="disable mpi")
parser.add_argument("--chunks", type=int, default=1, help="number of microbatches per batch")
parser.add_argument("--batch-size", type=int, default=8, help="size of a batch")
parser.add_argument("--all-at-once", action="store_true", default=False, help="do backward pass on whole batch at once")
parser.add_argument("--max-batch", type=int, default=4, help="Max number of batches")
parser.add_argument("--socket-name", type=str, default=None, help="socket ifname for gloo/tp")
parser.add_argument("--num-decoder-layers", type=int, default=10, help="Number of decoder layers in the model")
parser.add_argument("--ddp-zero", action="store_true", default=False, help="enable ddp")
parser.add_argument(
    "--lazy-construction", action="store_true", default=False, help="Number of decoder layers in the model"
)
parser.add_argument(
    "--checkpoint", default="never", choices=["always", "except_last", "never"], help="Checkpointing strategy for pipe"
)
parser.add_argument(
    "--pipelined-backward", dest="pipelined_backward", action="store_true", help="Pipelined backward pass"
)
parser.add_argument(
    "--no-pipelined-backward", dest="pipelined_backward", action="store_false", help="Pipelined backward pass"
)
parser.add_argument("--use_synthetic_data", default=True, help="Uses synthetic data for a sample training run.")
parser.add_argument(
    "--model_name", default="seq_pred", choices=["seq_pred", "transformer"], help="Model used to benchmark pipe."
)
parser.set_defaults(pipelined_backward=True)

if __name__ == "__main__":
    args = parser.parse_args()
    # TODO(anj-s): Add support for multiprocess benchmarking.
    if args.no_mpi or "OMPI_COMM_WORLD_RANK" not in os.environ:
        print(f"Running benchmark with args: {args}")
        benchmark_single_process(args)
    else:
        if os.environ["OMPI_COMM_WORLD_RANK"] == "0":
            print(f"Running benchmark with args: {args}")
        bench_mpi(args)
