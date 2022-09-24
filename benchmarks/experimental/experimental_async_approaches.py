# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
import sys
import time
import warnings

from benchmark_dataset import BenchmarkLMDataset, collate_sentences_lm
import torch
from torch.distributed import rpc
import torch.multiprocessing as mp
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import torchtext
from torchtext.data.utils import get_tokenizer

from fairscale.experimental.nn.ampnet_pipe import pipe
from fairscale.fair_dev.testing.testing import dist_init, get_worker_map
from fairscale.nn.model_parallel import initialize_model_parallel
from fairscale.nn.model_parallel.initialize import get_pipeline_parallel_group
from fairscale.nn.pipe import LazyModule
from fairscale.optim import GradScaler

try:
    from fairscale.optim import Adam  # type: ignore

    can_benchmark = True
except ImportError:
    from torch.optim import Adam  # type: ignore

    can_benchmark = False


def init_random_seed(seed: int):
    import numpy

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)


PIPE_CHUNKS = 2
iteration_count = 0


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
        global iteration_count
        iteration_count += 1

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

    def __init__(self, ntokens, ninp, nhead, nhid, dropout, initrange, ndecoder):
        layers = [
            EmbeddingLayer(ntokens, ninp, initrange),
            PositionalEncodingLayer(ninp, dropout),
        ]
        for _ in range(ndecoder):
            layers.append(TransformerDecoderLayer(ninp, nhead, nhid, dropout))

        layers.append(LinearLayer(ninp, ntokens, initrange))
        super(TransformerLMSequntial, self).__init__(*layers)


class MySGD(Optimizer):
    r"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (required)
    """

    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(MySGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MySGD, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(d_p, alpha=-group["lr"])
        return loss


class SpectrainSGDMomentum(Optimizer):
    r"""
    Implements a SGD with momentum optimizer with Spectrain based weight
    prediction. Please refer to the spectrain paper: https://arxiv.org/pdf/1809.02839.pdf
    for more details.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (required)
        momentum (float): momentum (default=0.9)
    """

    def __init__(self, params, lr, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        params = list(params)
        super(SpectrainSGDMomentum, self).__init__(params, defaults)
        self.old_weights = None

        self.cur_params, self.reference_params = self.prep_param_copies(params)
        for group in self.param_groups:
            for p in group["params"]:
                if momentum != 0:
                    param_state = self.state[p]
                    param_state["momentum_buffer"] = torch.zeros_like(p.data)

    def __setstate__(self, state):
        super(SpectrainSGDMomentum, self).__setstate__(state)

    def prep_param_copies(self, params):
        model_params = [param for param in params if param.requires_grad]
        reference_params = [param.clone().detach() for param in model_params]
        for param in reference_params:
            param.requires_grad = True
        return model_params, reference_params

    def copy_params(self, master_params, model_params):
        for model, master in zip(model_params, master_params):
            model.data.copy_(master.data)

    def modify_reference_params_using_current_params(self):
        self.copy_params(self.cur_params, self.reference_params)

    def modify_current_params_using_reference_params(self):
        self.copy_params(self.reference_params, self.cur_params)

    # chunk_index and chunks parameters are for unused for spectrain usecase
    def update_weight_using_future_predictions(self, model_index, num_gpus, chunk_index, chunks, forward):
        if forward:

            # In forward pass:
            # 1. clone weights to self.old_weights
            # 2. predict new weights and modify

            self.modify_reference_params_using_current_params()
            for group in self.param_groups:
                multiplier = group["lr"] * (model_index // 2 + num_gpus - model_index - 1)
                for p in group["params"]:
                    param_state = self.state[p]
                    p.data.sub_(param_state["momentum_buffer"].data, alpha=multiplier)

        else:
            # In backward pass:
            # 1. load old weights
            # 2. predict new weights and modify

            self.modify_current_params_using_reference_params()
            for group in self.param_groups:
                multiplier = group["lr"] * (model_index // 2)
                for p in group["params"]:
                    param_state = self.state[p]
                    p.data.sub_(param_state["momentum_buffer"].data, alpha=multiplier)

    def step(self, weight_prediction=True, closure=None):
        """Performs a single optimization step.
        Args:
            weight_prediction (bool, optional): Enable weight prediction based updates
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        if weight_prediction:
            self.modify_current_params_using_reference_params()

        for group in self.param_groups:
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if momentum != 0:
                    param_state = self.state[p]
                    buf = param_state["momentum_buffer"]
                    buf.data.mul_(momentum).add_(d_p, alpha=1 - momentum)

                    d_p = buf

                p.data.add_(d_p, alpha=-group["lr"])
        return loss


class XpipeAdam(Optimizer):
    r"""Implements Xpipe approach on top of Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.

    Xpipe details can be found here: https://arxiv.org/abs/1911.04610

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        params = list(params)

        super(XpipeAdam, self).__init__(params, defaults)
        self.cur_params, self.master_params = self.prep_param_copies(params)
        _, self.forward_params = self.prep_param_copies(params)
        _, self.backward_params = self.prep_param_copies(params)
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["step"] = 0
                # Exponential moving average of gradient values
                param_state["exp_avg"] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                param_state["exp_avg_sq"] = torch.zeros_like(p.data)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def prep_param_copies(self, params):
        model_params = [param for param in params if param.requires_grad]
        reference_params = [param.clone().detach() for param in model_params]
        for param in reference_params:
            param.requires_grad = True
        return model_params, reference_params

    def copy_params(self, master_params, model_params):
        for model, master in zip(model_params, master_params):
            model.data.copy_(master.data)

    def update_weight_using_future_predictions(
        self, model_index, num_gpus, current_microbatch_index, microbatches_per_minibatch, forward
    ):

        if forward:

            # Forward pass overview:
            # if bell-weather:
            #   1. read from master copy
            #   2. predict and modify
            #   3. flush updates to forward copy

            # else:
            #   1. read from forward copy

            if current_microbatch_index % microbatches_per_minibatch == 0:

                # read from master copy
                self.copy_params(self.master_params, self.cur_params)

                microbatch_index = current_microbatch_index + 1

                # predict and modify
                for group in self.param_groups:
                    multiplier = group["lr"] * round(
                        (microbatch_index + num_gpus - model_index / 2 - 2) / microbatch_index
                    )

                    beta1, beta2 = group["betas"]
                    eps = group["eps"]
                    for p in group["params"]:
                        param_state = self.state[p]
                        temp1 = param_state["exp_avg"].data / (1 - beta1)
                        temp2 = ((param_state["exp_avg_sq"].data / (1 - beta2)) + eps).sqrt()
                        p.data.addcdiv_(temp1, temp2, value=-multiplier)

                # flush updates to forward copy
                self.copy_params(self.cur_params, self.forward_params)

            else:
                self.copy_params(self.forward_params, self.cur_params)

        else:
            # Backward pass overview:
            # if bell-weather:
            #   1. read from master copy
            #   2. predict and modify
            #   3. flush updates to backward copy

            # else:
            #   1. read from backward copy

            if current_microbatch_index % microbatches_per_minibatch == 0:

                # read from master copy
                self.copy_params(self.master_params, self.cur_params)

                microbatch_index = current_microbatch_index + 1

                # predict and modify
                for group in self.param_groups:
                    multiplier = group["lr"] * (microbatch_index + model_index // 2 - 1) // microbatch_index

                    beta1, beta2 = group["betas"]
                    eps = group["eps"]
                    for p in group["params"]:
                        param_state = self.state[p]
                        temp1 = param_state["exp_avg"].data / (1 - beta1)
                        temp2 = ((param_state["exp_avg_sq"].data / (1 - beta2)) + eps).sqrt()
                        p.data.addcdiv_(temp1, temp2, value=-multiplier)

                # flush updates to forward copy
                self.copy_params(self.cur_params, self.backward_params)

            else:
                self.copy_params(self.backward_params, self.cur_params)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                amsgrad = group.get("amsgrad", False)

                p_data = p.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p_data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p_data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p_data)
                else:
                    state["exp_avg"] = state["exp_avg"].to(p_data)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(p_data)
                    if amsgrad:
                        state["max_exp_avg_sq"] = state["max_exp_avg_sq"].to(p_data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg_data = exp_avg.data
                exp_avg_sq_data = exp_avg_sq.data

                # Decay the first and second moment running average coefficient
                exp_avg_data.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq_data.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq_data, out=max_exp_avg_sq_data)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq_data.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                if group["weight_decay"] != 0:
                    p_data.add_(p_data, alpha=-group["weight_decay"] * group["lr"])
                p_data.addcdiv_(exp_avg_data, denom, value=-step_size)

        return loss


def get_data(device):
    with warnings.catch_warnings(record=True) as fjldska:
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


def make_model(args, device, ntokens):
    ninp = 2048  # embedding dimension
    nhid = 2048  # the dimension of the feedforward network model in nn.TransformerEncoder
    nhead = 32  # the number of heads in the multiheadattention models
    dropout = 0
    initrange = 0.1
    ndecoder = args.num_decoder_layers

    if args.lazy_construction:
        layers = [
            LazyModule(lambda: EmbeddingLayer(ntokens, ninp, initrange)),
            LazyModule(lambda: PositionalEncodingLayer(ninp, dropout)),
        ]
        for _ in range(ndecoder):
            layers.append(LazyModule(lambda: TransformerDecoderLayer(ninp, nhead, nhid, dropout)))

        layers.append(LazyModule(lambda: LinearLayer(ninp, ntokens, initrange)))
        model = layers
    else:
        model = TransformerLMSequntial(ntokens, ninp, nhead, nhid, dropout, initrange, ndecoder).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 0.01  # learning rate

    def make_adam(model):
        return Adam(model.parameters(), lr=lr)

    def make_custom_optimizer(model, args):
        if args.xpipe:
            return XpipeAdam(model.parameters(), lr=lr)
        elif args.spectrain:
            return SpectrainSGDMomentum(model.parameters(), lr=lr)
        else:
            return MySGD(model.parameters(), lr=lr)

    optimizer = make_custom_optimizer
    scaler = GradScaler()

    return model, criterion, optimizer, scaler


class AsyncDelegate(object):
    def __init__(self, vocab_size, iteration_per_batch=1000):
        self.cur_epoch = 0
        self.cur_iteration = 0
        self.iteration_per_batch = iteration_per_batch
        self.vocab_size = vocab_size
        self.word_counter = 0
        self.start_time = time.time()
        self.log_interval = 1
        self.total_loss = 0

    def transform_input(self, cur_batch):
        return cur_batch["input"]

    def transform_target(self, cur_batch):
        return cur_batch["target"].view(-1)

    def log_loss(self, cur_batch, loss, count):
        self.word_counter += cur_batch["ntokens"]
        if count % self.log_interval == 0 and count > 0:
            self.total_loss += loss.item()
            cur_loss = self.total_loss / self.log_interval
            elapsed = time.time() - self.start_time
            print(
                "| batch {:5d} | wps {:5.2f} | loss {:5.2f} | ppl {:8.2f}".format(
                    count, self.word_counter / elapsed, cur_loss, math.exp(cur_loss)
                )
            )
            self.word_counter = 0
            self.total_loss = 0
            self.start_time = time.time()

    def transform_output_before_loss(self, output_tensor):
        return output_tensor.view(-1, self.vocab_size)

    def check_and_save_weights(self, num_gradients):
        pass


def train(lm_dataloader, model, criterion, optimizer, vocab_size, args):
    model.train()
    from functools import reduce
    import operator

    num_params = reduce(operator.add, (reduce(operator.mul, x.size()) for x in model.parameters()))
    if model.group:
        total = torch.Tensor([num_params])
        if torch.cuda.is_available():
            total = total.cuda()
        torch.distributed.all_reduce(total, group=model.group)
        logging.info(
            f"training model, #prams = {num_params}, group: {model.group.rank()}, grank:"
            f" {torch.distributed.get_rank()}, sizes {model.group.size()}"
        )
        torch.distributed.barrier()
        if model.group.rank() == 0:
            logging.info(f"total #prams = {total.item()}")
    else:
        logging.info(f"training model, #prams = {num_params}")
    vocab_size = 10000  # FIXME
    total_loss = 0.0
    start_time = time.time()
    word_counter = 0

    optimizer = optimizer(model, args)
    transform_and_log = AsyncDelegate(vocab_size)
    model.interleave(
        lm_dataloader, criterion, optimizer, transform_and_log, args.min_update_interval, args.spectrain or args.xpipe
    )
    if model.group.rank() == model.group.size() - 1:
        print("Done with an epoch")


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


def benchmark_language_model(train_data, val_data, test_data, model, criterion, optimizer, ntokens, args):
    epoch = 1
    bptt = 35
    start_time = time.time()

    print("-" * 110)
    print("| start of epoch {:1d}".format(epoch))
    print("-" * 110)
    epoch_start_time = time.time()
    train(train_data, model, criterion, optimizer, bptt, ntokens, args)
    val_loss = 1  # evaluate(model, val_data, criterion, bptt, ntokens)
    print("-" * 89)
    print(
        "| end of epoch {:1d} | time: {:5.2f}s | valid loss {:5.2f} ".format(
            epoch, (time.time() - epoch_start_time), val_loss
        )
    )
    print("-" * 110)

    elapsed_time = time.time() - start_time
    nwords = get_number_of_words(train_data) + get_number_of_words(val_data)
    wps = nwords / elapsed_time

    test_loss = 1  # evaluate(model, test_data, criterion, bptt, ntokens)
    print("=" * 89)
    print(
        "| end of training | test loss {:5.2f} \n| time: {:5.2f}s | words: {:3d} | wps: {:5.2f}".format(
            test_loss, elapsed_time, nwords, wps
        )
    )
    print("=" * 110)


def generate_balance_weighted(num_devices, num_layers, fraction=0.5):
    balance = []
    layers_assigned = 0
    average_count = num_layers / num_devices
    last_layers = int(average_count * fraction)

    balance = generate_balance(num_devices - 1, num_layers - last_layers)
    balance.append(last_layers)
    return balance


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


def make_model_and_data(args, device, new_data: bool = True):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if new_data:
        vocab_size = 10000
        model, criterion, optimizer, scaler = make_model(args, device, vocab_size)
        lm_dataset = BenchmarkLMDataset()
        lm_dataloader = DataLoader(
            lm_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_sentences_lm
        )
        return {
            "model": model,
            "criterion": criterion,
            "optimizer": optimizer,
            "data": lm_dataloader,
            "vocab_size": vocab_size,
        }
    else:
        data = get_data(device)
        ntokens, train_data, val_data, test_data = data
        model, criterion, optimizer, scaler = make_model(args, device, ntokens)
        return {
            "model": model,
            "criterion": criterion,
            "optimizer": optimizer,
            "data": data,
        }


def run_mp_worker(args, available_workers):
    new_data = True

    blob = make_model_and_data(args, None, new_data=new_data)
    model = blob["model"]

    balance = generate_balance(get_pipeline_parallel_group().size(), len(model))
    p = pipe.AMPnetPipe(
        module=model,
        balance=balance,
        chunks=args.chunks,
        worker_map=get_worker_map(),
        input_device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        checkpoint=args.checkpoint,
    )
    if torch.cuda.is_available():
        p = p.cuda()

    if new_data:
        train(blob["data"], p, blob["criterion"], blob["optimizer"], blob["vocab_size"], args)
    else:
        ntokens, train_data, val_data, test_data = blob["data"]
        benchmark_language_model(train_data, val_data, test_data, p, criterion, optimizer, ntokens, args)


def run_worker(rank, world_size, args):
    if args.world_size != 0:
        world_size = args.world_size
    dist_init(rank + args.rank_base, world_size, hostname=args.host)
    initialize_model_parallel(1, world_size)
    init_random_seed(0)
    run_mp_worker(args, world_size)

    rpc.shutdown()
    torch.distributed.destroy_process_group()


def bench_multi_process(args, all_at_once=False):
    if args.local_world_size != 0:
        world_size = args.local_world_size
    else:
        world_size = min(torch.cuda.device_count(), 2)
    mp.spawn(run_worker, args=(world_size, args), nprocs=world_size, join=True)


best_device_map = {
    0: "mlx5_0:1",
    1: "mlx5_0:1",
    2: "mlx5_1:1",
    3: "mlx5_1:1",
    4: "mlx5_2:1",
    5: "mlx5_2:1",
    6: "mlx5_3:1",
    7: "mlx5_3:1",
}


def bench_mpi(args):
    guess_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    os.environ["UCX_NET_DEVICES"] = best_device_map[local_rank]

    os.environ["MASTER_ADDR"] = args.host
    os.environ["MASTER_PORT"] = "10638"
    if args.socket_name:
        os.environ["GLOO_SOCKET_IFNAME"] = args.socket_name
        os.environ["TP_SOCKET_IFNAME"] = args.socket_name

    torch.distributed.init_process_group(backend="gloo", rank=guess_rank, world_size=world_size)

    os.environ["MASTER_ADDR"] = args.host
    os.environ["MASTER_PORT"] = "10639"
    init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    rpc.init_rpc(
        f"Test{rank}",
        rank=rank,
        world_size=world_size,
        backend=rpc.BackendType.TENSORPIPE,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(rpc_timeout=20, init_method=init_method),
    )

    backends = {"model_parallel_backend": "nccl", "pipeline_backend": "mpi", "ddp_backend": "nccl"}

    initialize_model_parallel(1, world_size, **backends)
    init_random_seed(0)

    run_mp_worker(args, world_size)

    rpc.shutdown()
    torch.distributed.destroy_process_group()


parser = argparse.ArgumentParser(description="benchmark")
parser.add_argument("--local-world-size", "-l", type=int, default=0, help="local world size")
parser.add_argument("--world-size", "-w", type=int, default=0, help="world size")
parser.add_argument("--rank-base", "-r", type=int, help="rank base", default=0)
parser.add_argument("--host", "-o", type=str, default="localhost", help="hostname")
parser.add_argument("--no-mpi", action="store_true", default=False, help="disable mpi")
parser.add_argument("--chunks", type=int, default=1, help="number of microbatches per batch")
parser.add_argument("--batch-size", type=int, default=8, help="size of a batch")
parser.add_argument("--max-batch", type=int, default=4, help="Max number of batches")
parser.add_argument("--socket-name", type=str, default=None, help="socket ifname for gloo/tp")
parser.add_argument("--num-decoder-layers", type=int, default=10, help="Number of decoder layers in the model")
parser.add_argument("--spectrain", action="store_true", default=False, help="Use spectrain based weight prediction")
parser.add_argument("--xpipe", action="store_true", default=False, help="Use xpipe based weight prediction")
parser.add_argument(
    "--lazy-construction", action="store_true", default=False, help="Number of decoder layers in the model"
)
parser.add_argument(
    "--checkpoint", default="never", choices=["always", "except_last", "never"], help="Checkpointing strategy for pipe"
)
parser.add_argument("--min-update-interval", type=int, default=1, help="min update interval for ampnet")

"""
To run the script,
   1. please build a suitable version of OpenMPI with a cuda-enabled UCX backend.
   2. For running on 2 gpus:
   <open-mpi-installed-dir>/bin/mpirun --host localhost:8 -np 2 --map-by node --mca pml ucx -x UCX_TLS=rc,sm,cuda_ipc,cuda_copy -x PYTHONPATH=$PWD -x PATH=$PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -x UCX_RNDV_SCHEME=put_zcopy -x UCX_MEMTYPE_CACHE=n python3 benchmarks/experimental/experimental_async_approaches.py --num-decoder-layers=8 --host localhost --batch-size 4
   3. For doing Spectrain based weight prediction, add `--spectrain` to the training command line argument.
   4. For doing Xpipe based weight prediction, add `--xpipe` to the training command line argument.
"""

if __name__ == "__main__":
    args = parser.parse_args()
    # bench_multi_process(args, all_at_once=True)
    if args.no_mpi or "OMPI_COMM_WORLD_RANK" not in os.environ:
        print("Can't run benchmark")
        sys.exit(1)

    else:
        if os.environ["OMPI_COMM_WORLD_RANK"] == "0":
            print(f"Running benchmark with args: {args}")
        bench_mpi(args)
