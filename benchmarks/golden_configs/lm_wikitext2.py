# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch.nn as nn

from fairscale.optim import GradScaler


def get_benchmark_config():

    return {
        "epochs": 1,
        "vocab_size": 10000,
        "ninp": 2048,  # embedding dimension
        "nhid": 2048,  # the dimension of the feedforward network model in nn.TransformerEncoder
        "nhead": 32,  # the number of heads in the multiheadattention models
        "dropout": 0,
        "initrange": 0.1,
        "criterion": nn.CrossEntropyLoss(),
        "lr": 0.001,  # learning rate
        "scaler": GradScaler(),
        "clip_value": 0.05,
        "batch_size": 8,
    }


def get_golden_real_stats():

    return {
        "avg_wps": 703.778,
        "std_dev_wps": 5.732,
        "peak_mem_usage": [2320996352, 1396742144, 1396742144, 2340010496],
    }


def get_golden_synthetic_stats():
    # TODO(anj-s): Add support for synthetic regression benchmarks
    raise NotImplementedError("Synthetic data benchmarks are not supported.")
