# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Based on https://github.com/pytorch/tutorials/blob/master/beginner_source/transformer_tutorial.py
# Apply CPU offload and problem sharding to a big transformer model

import argparse
import logging
import tempfile
import time
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import math

import io
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from fairscale.utils.testing import generate_square_subsequent_mask, get_sequential_transformer

TEMPDIR = tempfile.gettempdir()
BPTT = 35
OPTIM = torch.optim.SGD

from fairscale.nn.data_parallel.offload_ddp import OffloadDataParallelExperimental
from torch.nn.parallel import DistributedDataParallel as DDP


def default_data_download():
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
    data_path = Path(TEMPDIR + "/" + "data")
    data_path.mkdir(exist_ok=True)
    filepath = download_from_url(url, root=TEMPDIR)
    logging.info(f"download to {filepath} done")

    test_filepath, valid_filepath, train_filepath = extract_archive(from_path=filepath, to_path=TEMPDIR)
    return test_filepath, valid_filepath, train_filepath


def batchify(data, batch_size):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // batch_size

    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)

    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data


def get_train_val_test(args: argparse.Namespace):

    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(map(tokenizer, iter(io.open(args.train_filepath, encoding="utf8"))))

    def data_process(raw_text_iter):
        data = [torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_data = data_process(iter(io.open(args.train_filepath, encoding="utf8")))
    val_data = data_process(iter(io.open(args.valid_filepath, encoding="utf8")))
    test_data = data_process(iter(io.open(args.test_filepath, encoding="utf8")))

    train_data = batchify(train_data, args.batch_size)
    val_data = batchify(val_data, args.eval_batch_size)
    test_data = batchify(test_data, args.eval_batch_size)

    return train_data, val_data, test_data, vocab


def get_batch(source, i, device):
    seq_len = min(BPTT, len(source) - 1 - i)
    data = source[i : i + seq_len].to(device)
    target = source[i + 1 : i + 1 + seq_len].reshape(-1).to(device)
    return data, target


def get_transformer(ntokens):
    emsize = 200  # embedding dimension
    nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value

    return get_sequential_transformer(ntokens, emsize, nhead, nhid, nlayers, dropout)


def train(rank: int, args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO)

    # DDP
    dist.init_process_group(backend="nccl", init_method="tcp://localhost:29501", rank=rank, world_size=args.world_size)
    torch.cuda.device(rank)
    device = torch.device(rank)

    # Setup the problem
    logging.info("Building dataset")
    train_data, val_data, test_data, vocab = get_train_val_test(args)
    ntokens = len(vocab.stoi)  # the size of vocabulary

    # Optim loop
    criterion = nn.CrossEntropyLoss()
    lr = 5.0  # learning rate

    # Get a model
    logging.info("Building model")
    model = get_transformer(ntokens)  # CPU by default

    if args.offload:
        logging.info("Using sharded offloading for training")
        offload_model = OffloadDataParallelExperimental(
            model_cpu=model,
            optimizer=OPTIM,
            optimizer_params={"lr": lr},
            world_size=args.world_size,
            device=device,
            offload_device=torch.device("cpu"),
        )

        optimizer = offload_model.optimizer
        model = offload_model

    else:
        logging.info("Using DDP for training")
        model = model.to(device)
        model = DDP(module=model, device_ids=[rank], find_unused_parameters=False)  # type: ignore
        optimizer = OPTIM(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    logging.info(f"Rank {rank} starts training")

    def train_epoch():
        model.train()  # Turn on the train mode
        total_loss = 0.0
        start_time = time.time()
        src_mask = generate_square_subsequent_mask(BPTT).to(device)

        for batch, i in enumerate(range(0, train_data.size(0) - 1, BPTT)):
            data, targets = get_batch(train_data, i, device)

            optimizer.zero_grad()
            if data.size(0) != BPTT:
                src_mask = generate_square_subsequent_mask(data.size(0)).to(device)

            output, _ = model((data, src_mask))

            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            log_interval = 200
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                logging.info(
                    "| epoch {:3d} | {:5d}/{:5d} batches | "
                    "lr {:02.2f} | ms/batch {:5.2f} | "
                    "loss {:5.2f} | ppl {:8.2f}".format(
                        epoch,
                        batch,
                        len(train_data) // BPTT,
                        scheduler.get_last_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss,
                        math.exp(cur_loss),
                    )
                )
                total_loss = 0
                start_time = time.time()

    def evaluate(eval_model, data_source):
        eval_model.eval()  # Turn on the evaluation mode
        total_loss = 0.0
        src_mask = generate_square_subsequent_mask(BPTT).to(device)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, BPTT):
                data, targets = get_batch(data_source, i, device=device)
                if data.size(0) != BPTT:
                    src_mask = generate_square_subsequent_mask(data.size(0)).to(device)
                output = eval_model(data, src_mask)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output_flat, targets).item()
        return total_loss / (len(data_source) - 1)

    ######################################################################
    # Loop over epochs. Save the model if the validation loss is the best
    # we've seen so far. Adjust the learning rate after each epoch.

    best_val_loss = float("inf")
    best_model = None

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_epoch()
        val_loss = evaluate(model, val_data)
        logging.info("-" * 89)
        logging.info(
            "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
            "valid ppl {:8.2f}".format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss))
        )
        logging.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()

    ######################################################################
    # Evaluate the model with the test dataset
    # -------------------------------------
    #
    # Apply the best model to check the result with the test dataset.

    test_loss = evaluate(best_model, test_data)
    logging.info("=" * 89)
    logging.info("| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(test_loss, math.exp(test_loss)))
    logging.info("=" * 89)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the CPU offload + sharding with a Transformer training")
    parser.add_argument("--world_size", action="store", default=2, type=int)
    parser.add_argument("--epochs", action="store", default=10, type=int)
    parser.add_argument("--batch_size", action="store", default=20, type=int)
    parser.add_argument("--eval_batch_size", action="store", default=10, type=int)
    parser.add_argument("--train_filepath", action="store", default=None, type=str)
    parser.add_argument("--valid_filepath", action="store", default=None, type=str)
    parser.add_argument("--test_filepath", action="store", default=None, type=str)
    parser.add_argument("--offload", action="store_true", default=False)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Benchmark arguments: %s" % args)

    if args.valid_filepath is None or args.test_filepath is None or args.train_filepath is None:
        logging.info("Fetching default dataset")
        args.test_filepath, args.valid_filepath, args.train_filepath = default_data_download()

    # Kick training, one process per rank
    logging.info("Starting training")
    mp.spawn(
        train, args=(args,), nprocs=args.world_size, join=True,
    )
