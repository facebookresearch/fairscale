# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from fairscale.optim import GradScaler


class Offload_Transformer:
    def get_model_config():
        return {
            "vocab_size": 10000,
            "ninp": 2048,  # embedding dimension
            "nhid": 2048,  # the dimension of the feedforward network model in nn.TransformerEncoder
            "nhead": 32,  # the number of heads in the multiheadattention models
            "dropout": 0,
            "initrange": 0.1,
            "scaler": GradScaler(),
            "clip_value": 0.05,
            "num_decoder_layers": 10,
            "seq_len": 32,
        }

    def get_benchmark_config(checkpoint_activation=True):

        return {
            "epochs": 1,
            "lr": 0.001,  # learning rate
            "batch_size": 8,
            "criterion": nn.CrossEntropyLoss(),
            "checkpoint_activation": checkpoint_activation,
            "num_microbatches": 1,
            "slices": 3,
        }

    def get_golden_real_stats():
        return {
            "avg_wps": 192.105,
            "std_dev_wps": 39.56,
            "peak_mem_usage": 1180848128,
        }


class Offload_Sequential:
    def get_model_config():
        return {
            "inputs": 100,
            "outputs": 5,
            "hidden": 1000,
            "layers": 100,
            "clip_value": 0.05,
        }

    def get_benchmark_config():

        return {
            "epochs": 1,
            "lr": 0.001,  # learning rate
            "batch_size": 8,
            "criterion": nn.CrossEntropyLoss(),
            "slices": 3,
            "checkpoint_activation": True,
            "num_microbatches": 1,
        }


class Pipe:
    def get_model_config():
        return {
            "vocab_size": 10000,
            "ninp": 2048,  # embedding dimension
            "nhid": 2048,  # the dimension of the feedforward network model in nn.TransformerEncoder
            "nhead": 32,  # the number of heads in the multiheadattention models
            "dropout": 0,
            "initrange": 0.1,
            "scaler": GradScaler(),
            "clip_value": 0.05,
            "num_decoder_layers": 10,
            "seq_len": 32,
        }

    def get_benchmark_config():

        return {
            "epochs": 1,
            "lr": 0.001,  # learning rate
            "batch_size": 8,
            "criterion": nn.CrossEntropyLoss(),
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


class MOE:
    def get_model_config():
        return {
            "vocab_size": 10000,
            "ninp": 1024,  # embedding dimension
            "nhid": 4096,  # the dimension of the feedforward network model in nn.TransformerEncoder
            "nhead": 32,  # the number of heads in the multiheadattention models
            "dropout": 0,
            "initrange": 0.1,
            "scaler": GradScaler(),
            "clip_value": 0.05,
            "num_decoder_layers": 20,
            "seq_len": 33,  # (seq_len - 1) needs to be divisible by num_local_experts
            "is_moe": True,
            "num_local_experts": 2,
        }

    def get_benchmark_config():
        return {
            "epochs": 1,
            "lr": 0.001,  # learning rate
            "batch_size": 32,
            "criterion": nn.CrossEntropyLoss(),
        }
