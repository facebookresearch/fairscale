# fairscale
![PyPI](https://img.shields.io/pypi/v/fairscale)
[![Documentation Status](https://readthedocs.org/projects/fairscale/badge/?version=latest)](https://fairscale.readthedocs.io/en/latest/?badge=latest)
[![CircleCI](https://circleci.com/gh/facebookresearch/fairscale.svg?style=shield)](https://app.circleci.com/pipelines/github/facebookresearch/fairscale/) ![PyPI - License](https://img.shields.io/pypi/l/fairscale) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/facebookresearch/fairscale/blob/master/CONTRIBUTING.md)

## Description
fairscale is a PyTorch extension library for high performance and large scale training for optimizing training on one or across multiple machines/nodes. This library extend basic pytorch capabilities while adding new experimental ones.

fairscale supports:
* Parallelism:
   * pipeline parallelism (fairscale.nn.Pipe)
   * tensor parallelism (fairscale.nn.model_parallel)
* Optimization:
   * optimizer state sharding (fairscale.optim.oss)


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
from torch.nn.parallel import DistributedDataParallel as DDP

def train(
    rank: int,
    world_size: int,
    epochs: int):

    # DDP init example
    dist.init_process_group(backend='nccl', init_method="tcp://localhost:29501", rank=rank, world_size=world_size)

    # Problem statement
    model = myAwesomeModel().to(rank)
    model = DDP(model, device_ids=[rank])
    dataloader = mySuperFastDataloader()
    loss_fn = myVeryRelevantLoss()
    base_optimizer = torch.optim.SGD # pick any pytorch compliant optimizer here
    base_optimizer_arguments = {} # pass any optimizer specific arguments here, or directly below when instantiating OSS

    optimizer = OSS(params=model.parameters(), optim=base_optimizer, **base_optimizer_arguments)

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



# Testing

We use circleci to test on PyTorch versions 1.5.1, 1.6.0 and 1.7.0 and CUDA version 10.1. Please create an [issue](https://github.com/facebookresearch/fairscale/issues) if you are having trouble with installation.

## Contributors

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License

fairscale is licensed under the [BSD-3-Clause License](LICENSE).

fairscale.nn.pipe is forked from [torchgpipe](https://github.com/kakaobrain/torchgpipe), Copyright 2019, Kakao Brain, licensed under [Apache License](http://www.apache.org/licenses/LICENSE-2.0).

fairscale.nn.model_parallel is forked from [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), Copyright 2020, NVIDIA CORPORATION, licensed under [Apache License](http://www.apache.org/licenses/LICENSE-2.0).

fairscale.optim.adascale is forked from [AdaptDL](https://github.com/petuum/adaptdl), Copyright 2020, Petuum, Inc., licensed under [Apache License](http://www.apache.org/licenses/LICENSE-2.0).

## References

Here is a list of all authors on relevant research papers this work is based on:

* torchgpipe: Chiheon Kim, Heungsub Lee, Myungryong Jeong, Woonhyuk Baek, Boogeon Yoon, Ildoo Kim, Sungbin Lim, Sungwoong Kim. [[Paper](https://arxiv.org/pdf/2004.09910.pdf)] [[Code](https://github.com/kakaobrain/torchgpipe)]
* ZeRO: Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. [[Paper](https://arxiv.org/pdf/1910.02054.pdf)] [[Code](https://github.com/microsoft/DeepSpeed)]
* Megatron-LM: Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro. [[Paper](https://arxiv.org/pdf/1909.08053.pdf)][[Code](https://github.com/NVIDIA/Megatron-LM)]
* AdaScale SGD: Tyler B. Johnson, Pulkit Agrawal, Haijie Gu, Carlos Guestrin. [[Paper](https://proceedings.icml.cc/static/paper_files/icml/2020/4682-Paper.pdf)]
