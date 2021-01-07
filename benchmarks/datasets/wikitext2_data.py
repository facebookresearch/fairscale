# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import io
import warnings

import torch
from torch.utils.data import Dataset
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import build_vocab_from_iterator


class SyntheticLMDataset(Dataset):
    """
    Dataset to benchmark a translation like seq2seq task.
    Args:
        vocab_size (int, optional): size of the vocabulary (default 10000).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        total_samples (int, optional): the total number of rows in the
            dataset (default: 10000).
    """

    def __init__(
        self, vocab_size=10000, max_source_positions=1024, total_samples=10000,
    ):
        self.vocab_size = vocab_size
        self.max_source_positions = max_source_positions
        self.total_samples = total_samples
        self.sizes = [self.max_source_positions] * self.total_samples

    def __getitem__(self, index):
        length = self.sizes[index]

        source = torch.randint(1, self.vocab_size, (length,))
        target = source.clone()
        return {
            "id": index,
            "source": source,
            "target": target,
        }

    def __len__(self):
        return self.total_samples


class Wikitext2Data:
    def get_real_dataloaders(args):
        """Return dataloaders for training, testing and validation."""

        url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
        test_filepath, valid_filepath, train_filepath = extract_archive(download_from_url(url))
        tokenizer = get_tokenizer("basic_english")

        def data_process(raw_text_iter):
            data = [
                torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long) for item in raw_text_iter
            ]
            return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

        vocab = build_vocab_from_iterator(map(tokenizer, iter(io.open(train_filepath, encoding="utf8"))))

        train_dataset = data_process(iter(io.open(train_filepath, encoding="utf8")))
        valid_dataset = data_process(iter(io.open(valid_filepath, encoding="utf8")))
        test_dataset = data_process(iter(io.open(test_filepath, encoding="utf8")))

        # TODO(anj-s): We need to pass a device argument if we want this to work
        # on multiple devices.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO(anj-s): Batch size needs to be argument that we pass in.
        def batchify(data):
            batch_size = args.batch_size
            data = torch.tensor(data)
            # Divide the dataset into bsz parts.
            nbatch = data.size(0) // batch_size
            # Trim off any extra elements that wouldn't cleanly fit (remainders).
            data = data.narrow(0, 0, nbatch * batch_size)
            # Evenly divide the data across the bsz batches.
            data = data.view(batch_size, -1).t().contiguous()
            return data.to(device)

        seq_len = 512
        total_batch_size = seq_len * args.batch_size
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=total_batch_size, collate_fn=batchify)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=total_batch_size, collate_fn=batchify)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=total_batch_size, collate_fn=batchify)
        return len(vocab.stoi), train_dataloader, valid_dataloader, test_dataloader

    def get_raw_real_data(device):
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

    def get_synthetic_dataset():
        return SyntheticLMDataset()

    def get_synthetic_dataloader(args):
        
        # TODO(anj-s): We need to pass a device argument if we want this to work
        # on multiple devices.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def batchify(data):
                batch_size = args.batch_size
                data = torch.tensor(data)
                # Divide the dataset into bsz parts.
                nbatch = data.size(0) // batch_size
                # Trim off any extra elements that wouldn't cleanly fit (remainders).
                data = data.narrow(0, 0, nbatch * batch_size)
                # Evenly divide the data across the bsz batches.
                data = data.view(batch_size, -1).t().contiguous()
                return data.to(device)

        # TODO(anj-s): Both seq_len and batch size should be part of the golden config.
        seq_len = 512
        total_batch_size = seq_len * args.batch_size
        # vocab_size is 10000 and length of the real data is 2049990.
        lm_dataset = torch.randint(1, 10000, (2049990,))
        
        lm_dataloader = torch.utils.data.DataLoader(
            lm_dataset, batch_size=total_batch_size, shuffle=True, num_workers=0, collate_fn=batchify
        )
        return lm_dataloader, lm_dataloader, lm_dataloader
