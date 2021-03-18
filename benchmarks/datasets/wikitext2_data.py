# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import io
import tempfile

import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import build_vocab_from_iterator


def _batchify(data, batch_size):
    data = torch.tensor(data)
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the bsz batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data


def _get_total_batch_size(benchmark_config, model_specs):
    return model_specs["seq_len"] * benchmark_config["batch_size"]


def get_real_dataloaders(args, benchmark_config, model_specs):
    """Return real dataloaders for training, testing and validation."""

    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
    tmpdir = tempfile.TemporaryDirectory()
    test_filepath, valid_filepath, train_filepath = extract_archive(download_from_url(url, root=tmpdir.name))
    tokenizer = get_tokenizer("basic_english")

    def data_process(raw_text_iter):
        data = [torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    vocab = build_vocab_from_iterator(map(tokenizer, iter(io.open(train_filepath, encoding="utf8"))))

    train_dataset = data_process(iter(io.open(train_filepath, encoding="utf8")))
    valid_dataset = data_process(iter(io.open(valid_filepath, encoding="utf8")))
    test_dataset = data_process(iter(io.open(test_filepath, encoding="utf8")))

    def batchify(data):
        batch_size = benchmark_config["batch_size"]
        return _batchify(data, batch_size)

    total_batch_size = _get_total_batch_size(benchmark_config, model_specs)
    train_dataloader = DataLoader(train_dataset, batch_size=total_batch_size, collate_fn=batchify)
    valid_dataloader = DataLoader(valid_dataset, batch_size=total_batch_size, collate_fn=batchify)
    test_dataloader = DataLoader(test_dataset, batch_size=total_batch_size, collate_fn=batchify)
    return len(vocab.stoi), train_dataloader, valid_dataloader, test_dataloader


def get_synthetic_dataloaders(args, benchmark_config, model_specs):
    """Return synthetic dataloaders for training, testing and validation."""

    def batchify(data):
        batch_size = benchmark_config["batch_size"]
        return _batchify(data, batch_size)

    total_batch_size = total_batch_size = _get_total_batch_size(benchmark_config, model_specs)
    # vocab_size is 10000 and length of the real data is 2049990.
    lm_dataset = torch.randint(1, 10000, (2049990,))

    lm_dataloader = DataLoader(
        lm_dataset, batch_size=total_batch_size, shuffle=True, num_workers=0, collate_fn=batchify
    )
    return lm_dataloader, lm_dataloader, lm_dataloader
