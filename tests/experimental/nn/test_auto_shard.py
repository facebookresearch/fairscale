# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing Auto Shard functionality of non nn.Sequential models.
"""

import math
import sys

import pytest
import torch
import torch.nn
import torch.nn as nn

from fairscale.internal import torch_version


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = torch.nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, *args):
        src = args[0]
        src_mask = args[1]
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


bptt = 35
ntokens = 28783  # the size of vocabulary
emsize = 200  # embedding dimension
nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 1  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # the number of heads in the multiheadattention models
dropout = 0.2  # the dropout value


def test_single_run():
    if sys.version_info.major == 3 and sys.version_info.minor > 10:
        pytest.skip("torch.fx doesn't seem to work 3.11 yet")
    if torch_version() < (1, 8, 0):
        pytest.skip("requires torch version >= 1.8.0")
    from fairscale.experimental.nn.auto_shard import shard_model

    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)
    sharded_model = shard_model(model)
    assert len(sharded_model) == 2, "Length is sharded model is incorrect."
    expected_param_nums = [5998600, 5785383]
    for i, model in enumerate(sharded_model):
        param_count = {}
        for name, module in model.named_modules():
            if "." in name:
                continue

            param_count[name] = sum([x.numel() for x in module.parameters()])
        assert expected_param_nums[i] == param_count[""]

    src_mask = torch.randn((35, 35), dtype=torch.float32)
    src = torch.randint(1, ntokens, (35, 20))
    input = [src, src_mask]
    for model in sharded_model:
        if type(input) == list:
            input = model(*input)
        else:
            input = model(input)

    assert input.size() == torch.Size([35, 20, 28783])


class Branch(torch.nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.left = nn.Linear(in_features=features, out_features=features)
        self.right = nn.Linear(in_features=features, out_features=features)

    def forward(self, x):
        if x.sum() > 1000:
            return self.left(x)
        else:
            return self.right(x)


class BranchedNetwork(torch.nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.net = torch.nn.ModuleList([Branch(features) for _ in range(10)])

    def forward(self, x):
        for module in self.net:
            x = module(x)
        return x


def test_dynaimc_conditionals_auto_wrapped():
    if torch_version() < (1, 8, 0):
        pytest.skip("requires torch version >= 1.8.0")
    from fairscale.experimental.nn.auto_shard import shard_model

    features = 10

    model = BranchedNetwork(features)
    sharded_model = shard_model(model, 3)
    assert len(sharded_model) == 3

    input_ = torch.randn(3, features)
    model_output = model(input_)
    sharded_model_output = input_
    for shard in sharded_model:
        sharded_model_output = shard(sharded_model_output)
    assert torch.allclose(model_output, sharded_model_output)
