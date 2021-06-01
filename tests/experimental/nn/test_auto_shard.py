# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing Auto Shard functionality of non nn.Sequential models.
"""

import math

import torch
import torch.nn
import torch.nn as nn

from fairscale.experimental.nn.auto_shard import shard_model


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
        # TODO(anj): Fix the following error when using autoshard
        # Error: TypeError: slice indices must be integers or None or have an __index__ method
        # x = x + self.pe[:x.size(0), self.d_model]
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
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)
    sharded_model = shard_model(model)
    assert len(sharded_model) == 2, "Length is sharded model is incorrect."
    expected_param_nums = [5998600, 5785383]
    for i, model in enumerate(sharded_model):
        param_count = {}
        for named_mods in model.named_modules():
            sum = 0
            for x in named_mods[1].parameters():
                mul_dims = math.prod(x.size())
                sum += mul_dims

            name = named_mods[0].split(".")[0]
            if name in param_count:
                param_count[name] += sum
            else:
                param_count[name] = sum

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
