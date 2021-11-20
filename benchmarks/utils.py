# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from functools import reduce
import logging
import operator

import datasets.wikitext2_data as wikitext2_data
from models import transformer_lm
import numpy as np
import torch
from torch.optim import Adam


def init_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def init_args():
    parser = argparse.ArgumentParser(description="benchmark")
    parser.add_argument("--host", "-o", type=str, default="localhost", help="hostname")
    parser.add_argument("--chunks", type=int, default=1, help="number of microbatches per batch")
    parser.add_argument("--batch-size", type=int, default=8, help="size of a batch")
    parser.add_argument(
        "--checkpoint",
        default="never",
        choices=["always", "except_last", "never"],
        help="Checkpointing strategy for pipe",
    )
    parser.add_argument(
        "--lazy-construction", action="store_true", default=False, help="Number of decoder layers in the model"
    )
    parser.add_argument("--max-batch", type=int, default=4, help="Max number of batches")
    parser.add_argument("--use_synthetic_data", action="store_true", help="Uses synthetic data for running benchmarks.")
    parser.add_argument("--dry_run", action="store_true", help="Run a sample training run without regression testing.")
    parser.add_argument(
        # TODO(anj-s): In the process of adding more models and hence the requirement for a flag.
        "--model_name",
        default="lm",
        help="Language Model(LM) used to benchmark nn.pipe.",
    )
    parser.add_argument("--debug", action="store_true", default=False, help="Display additional debug information")
    args = parser.parse_args()
    return args


def create_benchmark_config(model_name, config_class):
    """Return a dict with configurations required for benchmarking `model_name` model."""

    if model_name == "lm":
        return config_class.get_benchmark_config()
    else:
        raise RuntimeError("Unrecognized args.model_mame " % args.model_name)


def get_model_specs(model_name, config_class):
    """Return a dict with configurations required for configuring `model_name` model."""

    if model_name == "lm":
        return config_class.get_model_config()
    else:
        raise RuntimeError("Unrecognized args.model_mame " % model_name)


def create_model_config(args, benchmark_config=None, model_specs=None, device=None):
    """Return a dict with the given model, dataset and optimizer."""

    if not device:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset_info = get_dataset_info(args)
    assert model_specs is not None
    model_specs["vocab_size"] = dataset_info.ntokens
    model, optimizer = get_model_and_optimizer(args, device, benchmark_config, model_specs)
    return {
        "model": model,
        "optimizer": optimizer,
        "dataset_info": dataset_info,
    }


def get_model_and_optimizer(args, device, benchmark_config, model_config):
    """Return instantiated model and optimizer function."""

    if args.model_name == "lm":
        model = get_lm_model(args, device, model_config)

    lr = benchmark_config["lr"]

    def make_adam(params):
        return Adam(params, lr=lr)

    optimizer = make_adam
    return model, optimizer


def get_lm_model(args, device, config):
    """Get language model(based on GPT-2) used for sequence prediction."""

    ninp = config["ninp"]
    nhead = config["nhead"]
    initrange = config["initrange"]
    dropout = config["dropout"]
    vocab_size = config["vocab_size"]
    nhid = config["nhid"]
    ndecoder = config["num_decoder_layers"]
    is_moe = config.get("is_moe", False)
    num_local_experts = config.get("num_local_experts", 1)

    if args.lazy_construction:
        layers = [
            LazyModule(lambda: transformer_lm.EmbeddingLayer(vocab_size, ninp, initrange)),
            LazyModule(lambda: transformer_lm.PositionalEncodingLayer(ninp, dropout)),
        ]
        for _ in range(ndecoder):
            layers.append(
                LazyModule(
                    lambda: transformer_lm.TransformerDecoderLayer(
                        ninp, nhead, nhid, dropout, is_moe, num_local_experts
                    )
                )
            )

        layers.append(LazyModule(lambda: transformer_lm.LinearLayer(ninp, vocab_size, initrange)))
        model = layers
    else:
        model = transformer_lm.TransformerLM(
            vocab_size, ninp, nhead, nhid, dropout, initrange, ndecoder, is_moe, num_local_experts
        ).to(device)

    return model


def log_number_of_parameters(model, logger=None):
    if not logger:
        logger = logging
    num_params = reduce(operator.add, (reduce(operator.mul, x.size()) for x in model.parameters()))
    if hasattr(model, "group"):
        total = torch.Tensor([num_params])
        if torch.cuda.is_available():
            total = total.cuda()
        torch.distributed.all_reduce(total, group=model.group)
        logger.debug(
            f"training model, #params = {num_params}, group: {model.group.rank()}, grank:"
            f" {torch.distributed.get_rank()}, sizes {model.group.size()}"
        )
        torch.distributed.barrier()
        if model.group.rank() == 0:
            logger.debug(f"total #prams = {total.item()}")
    else:
        logger.debug(f"training model, #params = {num_params}")


def get_dataset_info(args):
    assert args.model_name == "lm"
    if args.use_synthetic_data:
        return wikitext2_data.get_synthetic_datasets()
    else:
        return wikitext2_data.get_real_datasets()


def get_data_loader(dataset_info, args, benchmark_config, model_specs, num_replicas=1, rank=0):
    return wikitext2_data.get_dataloaders(dataset_info, benchmark_config, model_specs, num_replicas, rank)
