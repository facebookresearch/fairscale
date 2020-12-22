# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import warnings

import torchtext
from torchtext.data.utils import get_tokenizer


def get_wikitext2_data(device):
    """Return batched data from wikitext2 dataset for training, validation and testing."""
    with warnings.catch_warnings(record=True) as _:
        text_field = torchtext.data.Field(
            tokenize=get_tokenizer("basic_english"), init_token="<sos>", eos_token="<eos>", lower=True
        )
        train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(text_field)
        text_field.build_vocab(train_txt)
        ntokens = len(text_field.vocab.stoi)

        batch_size = 20
        eval_batch_size = 10
        train_data = batchify(train_txt, batch_size, text_field, device)
        val_data = batchify(val_txt, eval_batch_size, text_field, device)
        test_data = batchify(test_txt, eval_batch_size, text_field, device)

        return ntokens, train_data, val_data, test_data


def batchify(data, bsz, text_field, device):
    """Return batched data that is placed on the specified device."""
    data = text_field.numericalize([data.examples[0].text])
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)
