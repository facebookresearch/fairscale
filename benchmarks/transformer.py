# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import math
import time

from models import transformerModel as transformer
import torch
import torch.nn as nn
import torchtext
from torchtext.data.utils import get_tokenizer


def get_data(device):
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


def make_model(device, ntokens):
    emsize = 50  # embedding dimension
    nhid = 50  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 1  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value
    model = transformer.TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    lr = 5.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return model, criterion, optimizer


def train(train_data, model, criterion, optimizer, bptt, ntokens):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(
                "| {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}".format(
                    batch, len(train_data) // bptt, elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss),
                )
            )
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source, criterion, bptt, ntokens):
    eval_model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def get_number_of_words(data):
    return data.size()[0] * data.size()[1]


def benchmark_language_model(train_data, val_data, test_data, model, criterion, optimizer, ntokens):
    epoch = 1
    bptt = 35
    start_time = time.time()

    print("-" * 89)
    print("| start of epoch {:1d}".format(epoch))
    print("-" * 89)
    epoch_start_time = time.time()
    train(train_data, model, criterion, optimizer, bptt, ntokens)
    val_loss = evaluate(model, val_data, criterion, bptt, ntokens)
    print("-" * 89)
    print(
        "| end of epoch {:1d} | time: {:5.2f}s | valid loss {:5.2f} | "
        "valid ppl {:8.2f}".format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss))
    )
    print("-" * 89)

    elapsed_time = time.time() - start_time
    nwords = get_number_of_words(train_data) + get_number_of_words(val_data)
    wps = nwords / elapsed_time

    test_loss = evaluate(model, test_data, criterion, bptt, ntokens)
    print("=" * 89)
    print(
        "| end of training | test loss {:5.2f} | test ppl {:8.2f}\n| time: {:5.2f}s | words: {:3d} | wps: {:5.2f}".format(
            test_loss, math.exp(test_loss), elapsed_time, nwords, wps
        )
    )
    print("=" * 89)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    ntokens, train_data, val_data, test_data = get_data(device)
    model, criterion, optimizer = make_model(device, ntokens)
    benchmark_language_model(train_data, val_data, test_data, model, criterion, optimizer, ntokens)
