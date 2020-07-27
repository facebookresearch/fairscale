# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import math
import time

import torch
import torch.nn as nn
import torchtext
from torchtext.data.utils import get_tokenizer

import fairscale.nn.pipe.pipe as pipe

try:
    from fairscale.optim.adam import Adam  # type: ignore
except ImportError:
    from torch.optim import Adam  # type: ignore


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

    def __init__(self, ntokens, ninp, nhead, nhid, dropout, initrange):
        super(TransformerLMSequntial, self).__init__(
            EmbeddingLayer(ntokens, ninp, initrange),
            PositionalEncodingLayer(ninp, dropout),
            TransformerDecoderLayer(ninp, nhead, nhid, dropout),
            LinearLayer(ninp, ntokens, initrange),
        )


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
    ninp = 50  # embedding dimension
    nhid = 50  # the dimension of the feedforward network model in nn.TransformerEncoder
    nhead = 2  # the number of heads in the multiheadattention models
    dropout = 0
    initrange = 0.1

    model = TransformerLMSequntial(ntokens, ninp, nhead, nhid, dropout, initrange).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 0.01  # learning rate
    optimizer = Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer


def train(train_data, model, criterion, optimizer, bptt, ntokens):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        optimizer.zero_grad()
        output = model(data)
        output = output.to(targets.device)

        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 0.05)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(
                "| {:5d}/{:5d} batches | ms/batch {:5.2f} | "
                "loss {:5.2f} | ppl {:8.2f}".format(
                    batch, len(train_data) // bptt, elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)
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
            output = output.to(targets.device)
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
        "| end of epoch {:1d} | time: {:5.2f}s | valid loss {:5.2f} ".format(
            epoch, (time.time() - epoch_start_time), val_loss
        )
    )
    print("-" * 89)

    elapsed_time = time.time() - start_time
    nwords = get_number_of_words(train_data) + get_number_of_words(val_data)
    wps = nwords / elapsed_time

    test_loss = evaluate(model, test_data, criterion, bptt, ntokens)
    print("=" * 89)
    print(
        "| end of training | test loss {:5.2f} \n| time: {:5.2f}s | words: {:3d} | wps: {:5.2f}".format(
            test_loss, elapsed_time, nwords, wps
        )
    )
    print("=" * 89)

    if len(model.balance) == 4:
        # Assert that words per second is within 3 standard deviations of the average
        # of five golden runs
        assert wps > 19276.1 - (3 * 88)

        print("Peak allocated bytes on cuda:0: {:1d}".format(torch.cuda.memory_stats(0)["allocated_bytes.all.peak"]))
        print("Peak allocated bytes on cuda:1: {:1d}".format(torch.cuda.memory_stats(1)["allocated_bytes.all.peak"]))
        print("Peak allocated bytes on cuda:2: {:1d}".format(torch.cuda.memory_stats(2)["allocated_bytes.all.peak"]))
        print("Peak allocated bytes on cuda:3: {:1d}".format(torch.cuda.memory_stats(3)["allocated_bytes.all.peak"]))

        # Assert that memory usage on each GPU is within 10% of golden run
        # Right-hand-side is golden run bytes * 110%
        assert torch.cuda.memory_stats(0)["allocated_bytes.all.peak"] < 365915648 * 1.1
        assert torch.cuda.memory_stats(1)["allocated_bytes.all.peak"] < 1281024 * 1.1
        assert torch.cuda.memory_stats(2)["allocated_bytes.all.peak"] < 2788864 * 1.1
        assert torch.cuda.memory_stats(3)["allocated_bytes.all.peak"] < 190724608 * 1.1
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


if __name__ == "__main__":
    num_devices = torch.cuda.device_count()
    assert num_devices > 0

    torch.manual_seed(0)
    device = torch.device("cuda")
    ntokens, train_data, val_data, test_data = get_data(device)
    model, criterion, optimizer = make_model(device, ntokens)
    balance = generate_balance(min(num_devices, 4), len(model))
    p = pipe.Pipe(model, balance)
    benchmark_language_model(train_data, val_data, test_data, p, criterion, optimizer, ntokens)
    del p
