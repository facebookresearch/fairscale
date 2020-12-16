import warnings

import torchtext
from torchtext.data.utils import get_tokenizer


def get_wikitext2_data(device):
    with warnings.catch_warnings(record=True) as _:
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