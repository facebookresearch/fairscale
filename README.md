![FairScale Logo](./docs/source/_static/img/fairscale-logo.png)

![PyPI](https://img.shields.io/pypi/v/fairscale)
[![Documentation Status](https://readthedocs.org/projects/fairscale/badge/?version=latest)](https://fairscale.readthedocs.io/en/latest/?badge=latest)
[![CircleCI](https://circleci.com/gh/facebookresearch/fairscale.svg?style=shield)](https://app.circleci.com/pipelines/github/facebookresearch/fairscale/) ![PyPI - License](https://img.shields.io/pypi/l/fairscale) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/facebookresearch/fairscale/blob/master/CONTRIBUTING.md)
--------------------------------------------------------------------------------

## Description
FairScale is a PyTorch extension library for high performance and large scale training on one or multiple machines/nodes. This library extends basic PyTorch capabilities while adding new experimental ones.

FairScale supports:
* Parallelism:
   * Pipeline parallelism (`fairscale.nn.pipe`)
   * Asynchronous Pipeline parallelism (`fairscale.nn.async_pipe`)
   * Model Parallelism (`fairscale.nn.model_parallel.layers`)
   * _experimental_ AmpNet (`fairscale.experimental.nn.ampnet_pipe`)
* Sharded training:
   * Optimizer state sharding (`fairscale.optim.OSS`)
   * Sharded Data Parallel (SDP) (`fairscale.nn.ShardedDataParallel`)
   * Fully Sharded Data Parallel (FSDP) (`fairscale.nn.FullyShardedDataParallel`) (PyTorch >= 1.6)
   * OffloadModel (`fairscale.experimental.nn.OffloadModel`)
* Optimization at scale:
   * AdaScale SGD (`fairscale.optim.AdaScale`)
* GPU memory optimization:
   * Activation checkpointing wrapper (`fairscale.nn.misc.checkpoint_wrapper`)
* GPU speed optimization:
   * Sharded grad scaler - automatic mixed precision (`fairscale.optim.grad_scaler`)

## Requirements

* PyTorch >= 1.5.1

## Installation

Normal installation:
```bash
pip install fairscale
```

Development mode:
```bash
cd fairscale
pip install -r requirements.txt
pip install -e .
```

If either of the above fails, add `--no-build-isolation` to the `pip install` command (this could be a problem with recent versions of pip).


## Getting Started
The full documentation (https://fairscale.readthedocs.io/) contains instructions for getting started and extending fairscale.

## Examples
### Pipe

Run a 4-layer model on 2 GPUs. The first two layers run on cuda:0 and the next two layers run on cuda:1.

```python
import torch

import fairscale

model = torch.nn.Sequential(a, b, c, d)
model = fairscale.nn.Pipe(model, balance=[2, 2], devices=[0, 1], chunks=8)
```

### Optimizer state sharding (ZeRO)
See a more complete example [here](https://github.com/facebookresearch/fairscale/blob/master/benchmarks/oss.py), but a minimal example could look like the following :

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP

def train(
    rank: int,
    world_size: int,
    epochs: int):

    # DDP init example
    dist.init_process_group(backend='nccl', init_method="tcp://localhost:29501", rank=rank, world_size=world_size)

    # Problem statement
    model = myAwesomeModel().to(rank)
    dataloader = mySuperFastDataloader()
    loss_fn = myVeryRelevantLoss()
    base_optimizer = torch.optim.SGD # pick any pytorch compliant optimizer here
    base_optimizer_arguments = {} # pass any optimizer specific arguments here, or directly below when instantiating OSS

    # Wrap the optimizer in its state sharding brethren
    optimizer = OSS(params=model.parameters(), optim=base_optimizer, **base_optimizer_arguments)

    # Wrap the model into ShardedDDP, which will reduce gradients to the proper ranks
    model = ShardedDDP(model, optimizer)

    # Any relevant training loop, nothing specific to OSS. For example:
    model.train()
    for e in range(epochs):
        for batch in dataloader:
            # Train
            model.zero_grad()
            outputs = model(batch["inputs"])
            loss = loss_fn(outputs, batch["label"])
            loss.backward()
            optimizer.step()

    dist.destroy_process_group()

if __name__ == "__main__":
    # Supposing that WORLD_SIZE and EPOCHS are somehow defined somewhere
    mp.spawn(
        train,
        args=(
            WORLD_SIZE,
            EPOCHS,
        ),
        nprocs=WORLD_SIZE,
        join=True,
    )
```

### AdaScale SGD

AdaScale can be used to wrap a SGD optimizer and to be used in DDP (Distributed Data Parallel)
training or non-DDP with gradient accumulation. The benefit is to re-use the same LR
schedule from a baseline batch size when effective batch size is bigger.

Note that AdaScale does _not_ help increase per-GPU batch size.

```python
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR  # or your scheduler
from fairscale.optim import AdaScale

...
optim = AdaScale(SGD(model.parameters(), lr=0.1))
scheduler = LambdaLR(optim, ...)
...
# Note: the train loop should be with DDP or with gradient accumulation.
last_epoch = 0
step = 0
done = False
while not done:
    for sample in dataset:
        ...
        step += optim.gain()
        optim.step()
        epoch = step // len(dataset)
        if last_epoch != epoch:
            scheduler.step()
            last_epoch = epoch
        if epoch > max_epoch:
            done = True
```

Primary goal is to allow scaling to bigger batch sizes without losing model accuracy.
(However, training time might be longer comparing to without AdaScale.)

At a high level, we want ML researchers to:
  * go parallel more easily (i.e. no need to find new learning rate schedules)
  * not worrying about losing accuracy
  * potentially higher GPU efficiency (fewer steps, less networking overhead, etc.)

# Testing

We use circleci to test on PyTorch versions 1.6.0, 1.7.1, and 1.8.1. Please create an [issue](https://github.com/facebookresearch/fairscale/issues) if you are having trouble with installation.

## Contributors

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License

fairscale is licensed under the [BSD-3-Clause License](LICENSE).

fairscale.nn.pipe is forked from [torchgpipe](https://github.com/kakaobrain/torchgpipe), Copyright 2019, Kakao Brain, licensed under [Apache License](http://www.apache.org/licenses/LICENSE-2.0).

fairscale.nn.model_parallel is forked from [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), Copyright 2020, NVIDIA CORPORATION, licensed under [Apache License](http://www.apache.org/licenses/LICENSE-2.0).

fairscale.optim.adascale is forked from [AdaptDL](https://github.com/petuum/adaptdl), Copyright 2020, Petuum, Inc., licensed under [Apache License](http://www.apache.org/licenses/LICENSE-2.0).

fairscale.nn.misc.flatten_params_wrapper is forked from [PyTorch-Reparam-Module](https://github.com/SsnL/PyTorch-Reparam-Module), Copyright 2018, Tongzhou Wang, licensed under [MIT License](https://github.com/SsnL/PyTorch-Reparam-Module/blob/master/LICENSE).

## References

Here is a list of all authors on relevant research papers this work is based on:

* torchgpipe: Chiheon Kim, Heungsub Lee, Myungryong Jeong, Woonhyuk Baek, Boogeon Yoon, Ildoo Kim, Sungbin Lim, Sungwoong Kim. [[Paper](https://arxiv.org/pdf/2004.09910.pdf)] [[Code](https://github.com/kakaobrain/torchgpipe)]
* ZeRO: Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. [[Paper](https://arxiv.org/pdf/1910.02054.pdf)] [[Code](https://github.com/microsoft/DeepSpeed)]
* Megatron-LM: Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro. [[Paper](https://arxiv.org/pdf/1909.08053.pdf)][[Code](https://github.com/NVIDIA/Megatron-LM)]
* AdaScale SGD: Tyler B. Johnson, Pulkit Agrawal, Haijie Gu, Carlos Guestrin. [[Paper](https://proceedings.icml.cc/static/paper_files/icml/2020/4682-Paper.pdf)]
* GShard: Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen [[Paper]](https://arxiv.org/abs/2006.16668)
* AMPNet:Alexander L. Gaunt, Matthew A. Johnson, Maik Riechert, Daniel Tarlow, Ryota Tomioka, Dimitrios Vytiniotis, Sam Webster [[Paper]](https://arxiv.org/abs/1705.09786)
* L2L: Training large Neural networks with constant Memory using a new execution Algorithm, 2020, [[Paper](https://arxiv.org/abs/2002.05645)]
* ZeRO-Offload: Democratizing Billion-Scale Model Training. 2021, [[Paper](https://arxiv.org/abs/2101.06840)]
