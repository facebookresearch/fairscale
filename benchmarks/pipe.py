# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse
import math
import os
import time
import warnings

from benchmark_dataset import BenchmarkLMDataset, collate_sentences_lm
import torch
from torch.distributed import rpc
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
from torchtext.data.utils import get_tokenizer

from fairscale.nn import Pipe
from fairscale.nn.model_parallel import initialize_model_parallel
from fairscale.nn.pipe import pipe
from fairscale.optim import GradScaler
from tests.nn.model_parallel.commons import dist_init, get_worker_map

try:
    from fairscale.optim import Adam  # type: ignore

    can_benchmark = True
except ImportError:
    from torch.optim import Adam  # type: ignore

    can_benchmark = False


def init_random_seed(seed: int):
    import numpy

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)


PIPE_CHUNKS = 2
iteration_count = 0


class EmbeddingLayer(nn.Embedding):
    def __init__(self, ntoken, ninp, initrange):
        super().__init__(ntoken, ninp)
        self.ninp = ninp
        self.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        return super().forward(src) * math.sqrt(self.ninp)


class PositionalEncodingLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncodingLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerDecoderLayer(nn.TransformerEncoderLayer):
    """Though this class inherits from torch.nn.TransformerEncoderLayer,
    it functions as a decoder in this model"""

    def __init__(self, ninp, nhead, nhid, droupout):
        super().__init__(ninp, nhead, nhid, droupout)
        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        global iteration_count
        iteration_count += 1
        # if iteration_count == 196:
        #    dump_cuda_tensors()

        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        return super().forward(src, self.src_mask)


class LinearLayer(nn.Linear):
    def __init__(self, ninp, ntoken, initrange):
        super().__init__(ninp, ntoken)
        self.bias.data.zero_()
        self.weight.data.uniform_(-initrange, initrange)


class TransformerLMSequntial(nn.Sequential):
    """A small language model based on the design of GPT-2 using nn.Sequeitnal
    for compatability with Pipe"""

    def __init__(self, ntokens, ninp, nhead, nhid, dropout, initrange, ndecoder):
        layers = [
            EmbeddingLayer(ntokens, ninp, initrange),
            PositionalEncodingLayer(ninp, dropout),
        ]
        for _ in range(ndecoder):
            layers.append(TransformerDecoderLayer(ninp, nhead, nhid, dropout))

        layers.append(LinearLayer(ninp, ntokens, initrange))
        super(TransformerLMSequntial, self).__init__(*layers)


def get_data(device):
    with warnings.catch_warnings(record=True) as fjldska:
        TEXT = torchtext.data.Field(
            tokenize=get_tokenizer("basic_english"), init_token="<sos>", eos_token="<eos>", lower=True
        )
        train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
        TEXT.build_vocab(train_txt)
        ntokens = len(TEXT.vocab.stoi)

        batch_size = 20
        eval_batch_size = 10
        train_data = batchify(train_txt, batch_size, TEXT, device)
        val_data = batchify(val_txt, eval_batch_size, TEXT, device)
        test_data = batchify(test_txt, eval_batch_size, TEXT, device)

        return ntokens, train_data, val_data, test_data


def batchify(data, bsz, TEXT, device):
    data = TEXT.numericalize([data.examples[0].text])
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target


def make_model(args, device, ntokens):
    ninp = 2048  # embedding dimension
    nhid = 2048  # the dimension of the feedforward network model in nn.TransformerEncoder
    nhead = 32  # the number of heads in the multiheadattention models
    dropout = 0
    initrange = 0.1
    ndecoder = args.num_decoder_layers

    if args.lazy_construction:
        layers = [
            lambda: EmbeddingLayer(ntokens, ninp, initrange),
            lambda: PositionalEncodingLayer(ninp, dropout),
        ]
        for _ in range(ndecoder):
            layers.append(lambda: TransformerDecoderLayer(ninp, nhead, nhid, dropout))

        layers.append(lambda: LinearLayer(ninp, ntokens, initrange))
        model = layers
    else:
        model = TransformerLMSequntial(ntokens, ninp, nhead, nhid, dropout, initrange, ndecoder).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 0.01  # learning rate

    def make_adam(model):
        return Adam(model.parameters(), lr=lr)

    optimizer = make_adam
    scaler = GradScaler()

    return model, criterion, optimizer, scaler


def get_tensors_by_size_bucket():
    import gc
    from collections import defaultdict

    size_buckets = defaultdict(int)
    for obj in gc.get_objects():
        if not isinstance(obj, torch.Tensor):
            continue
        if obj.device.type == "cuda":
            size_buckets[(*obj.size(),) + (obj.element_size(),)] += 1

    return size_buckets


def dump_size_buckets(size_buckets, prefix=""):
    import operator
    from functools import reduce

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
    from functools import reduce
    import operator
    import gc

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

    import pprint

    pprint.pprint(torch.cuda.memory_stats())


def train(lm_dataloader, model, criterion, optimizer, vocab_size, args):
    model.train()
    from functools import reduce
    import operator

    num_params = reduce(operator.add, (reduce(operator.mul, x.size()) for x in model.parameters()))
    if model.group:
        print(f"training model, #prams = {num_params}, group: {model.group.rank()}, sizes {model.group.size()}")
    else:
        print(f"training model, #prams = {num_params}")
    vocab_size = 10000  # FIXME
    total_loss = 0.0
    start_time = time.time()
    word_counter = 0

    optimizer = optimizer(model)

    def get_first_device(model):
        if model.devices:
            return model.devices[0]
        else:
            return torch.cuda.current_device()

    def get_last_device(model):
        if model.devices:
            return model.devices[-1]
        else:
            return torch.cuda.current_device()

    for i, batch in enumerate(lm_dataloader):
        if args.max_batch and i > args.max_batch:
            break
        optimizer.zero_grad()
        output = model(batch["input"].to(get_first_device(model)))

        if model.group is None or model.group.rank() == model.group.size() - 1:
            target = batch["target"].to(get_last_device(model))
            output = output.to(target.device)
            loss = criterion(output.view(-1, vocab_size), target.view(-1))
            loss.backward()
        else:
            model.back_helper(output)

        del output

        torch.nn.utils.clip_grad_value_(model.parameters(), 0.05)
        optimizer.step()

        if model.group is None or model.group.rank() == model.group.size() - 1:
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
        # if i >= 10:
        #    break
        # torch.cuda.empty_cache()
        # check_size_buckets()


def evaluate(eval_model, data_source, criterion, bptt, ntokens):
    eval_model.eval()
    total_loss = 0.0
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


def benchmark_language_model(train_data, val_data, test_data, model, criterion, optimizer, ntokens, args):
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


def make_model_and_data(args, device, new_data: bool = True):
    if new_data:
        device = torch.device("cuda")
        vocab_size = 10000
        model, criterion, optimizer, scaler = make_model(args, device, vocab_size)
        lm_dataset = BenchmarkLMDataset()
        lm_dataloader = DataLoader(
            lm_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_sentences_lm
        )
        return {
            "model": model,
            "criterion": criterion,
            "optimizer": optimizer,
            "data": lm_dataloader,
            "vocab_size": vocab_size,
        }
    else:
        device = torch.device("cuda")
        data = get_data(device)
        ntokens, train_data, val_data, test_data = data
        model, criterion, optimizer, scaler = make_model(args, device, ntokens)
        return {
            "model": model,
            "criterion": criterion,
            "optimizer": optimizer,
            "data": data,
        }


def bench_single_process(args):
    num_devices = torch.cuda.device_count()
    assert num_devices > 0
    init_random_seed(0)
    device = torch.device("cuda")

    new_data = True

    blob = make_model_and_data(args, None, new_data=new_data)
    model = blob["model"]

    balance = generate_balance(min(num_devices, 8), len(model))
    p = pipe.Pipe(
        model, balance, chunks=args.chunks, pipelined_backward=args.pipelined_backward, checkpoint=args.checkpoint
    )
    del model
    del blob["model"]

    if new_data:
        train(blob["data"], p, blob["criterion"], blob["optimizer"], blob["vocab_size"], args)
    else:
        ntokens, train_data, val_data, test_data = blob["data"]
        benchmark_language_model(train_data, val_data, test_data, p, criterion, optimizer, ntokens, args)


def run_mp_worker(args, available_workers):
    new_data = True

    blob = make_model_and_data(args, None, new_data=new_data)
    model = blob["model"]

    balance = generate_balance(min(available_workers, 8), len(model))
    p = pipe.Pipe(
        model,
        balance,
        style=Pipe.MultiProcess,
        chunks=args.chunks,
        worker_map=get_worker_map(),
        input_device=torch.cuda.current_device(),
        pipelined_backward=args.pipelined_backward,
        checkpoint=args.checkpoint,
    ).cuda()

    if args.all_at_once and p.pipeline:
        print(f"running all at once")
        p.pipeline.all_at_once = True

    if new_data:
        train(blob["data"], p, blob["criterion"], blob["optimizer"], blob["vocab_size"], args)
    else:
        ntokens, train_data, val_data, test_data = blob["data"]
        benchmark_language_model(train_data, val_data, test_data, p, criterion, optimizer, ntokens, args)


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
    os.environ["UCX_NET_DEVICES"] = best_device_map[guess_rank]

    torch.distributed.init_process_group(backend="mpi")
    os.environ["MASTER_ADDR"] = args.host
    os.environ["MASTER_PORT"] = "10639"
    if args.socket_name:
        os.environ["GLOO_SOCKET_IFNAME"] = args.socket_name
        os.environ["TP_SOCKET_IFNAME"] = args.socket_name
    init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    rpc.init_rpc(
        f"Test{rank}",
        rank=rank,
        world_size=world_size,
        backend=rpc.BackendType.PROCESS_GROUP,
        rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(rpc_timeout=20, init_method=init_method),
    )

    initialize_model_parallel(1, world_size)
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
parser.set_defaults(pipelined_backward=True)

if __name__ == "__main__":
    args = parser.parse_args()
    # bench_multi_process(args, all_at_once=True)
    if args.no_mpi or "OMPI_COMM_WORLD_RANK" not in os.environ:
        print(f"Running benchmark with args: {args}")
        bench_single_process(args)
    else:
        if os.environ["OMPI_COMM_WORLD_RANK"] == "0":
            print(f"Running benchmark with args: {args}")
        bench_mpi(args)
