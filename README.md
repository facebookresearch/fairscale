# fairscale
fairscale is a PyTorch extension library for high performance and large scale training.

fairscale supports:
* pipeline parallelism (fairscale.nn.Pipe)
* tensor parallelism (fairscale.nn.model_parallel)
* optimizer state sharding (fairscale.optim.oss)

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
See a more complete example [here](https://github.com/facebookresearch/fairscale/blob/oss_async_broadcast/benchmarks/oss.py), but a minimal example could look like the following :

```python
import torch
from fairscale.optim.oss import OSS

def train(
    rank: int,
    world_size: int,
    epochs: int):

    # DDP
    dist_init(rank, world_size)

    # Problem statement
    model = myAwesomeModel()
    dataloader = mySuperFastDataloader()
    loss = myVeryRelevantLoss()
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
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            loss /= world_size
            loss.backward()
            optimizer.step()

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


## Requirements

* PyTorch >= 1.4

## Installation

Normal installation:
```bash
pip install .
```

Development mode:
```bash
pip install -e .
```

## Contributors

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License

fairscale is licensed under the [BSD-3-Clause License](LICENSE).

fairscale.nn.pipe is forked from [torchgpipe](https://github.com/kakaobrain/torchgpipe), Copyright 2019, Kakao Brain, licensed under [Apache License](http://www.apache.org/licenses/LICENSE-2.0).

fairscale.nn.model_parallel is forked from [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), Copyright 2020, NVIDIA CORPORATION, licensed under [Apache License](http://www.apache.org/licenses/LICENSE-2.0).

## References

Here is a list of all authors on relevant research papers this work is based on:

* torchgpipe: Chiheon Kim, Heungsub Lee, Myungryong Jeong, Woonhyuk Baek, Boogeon Yoon, Ildoo Kim, Sungbin Lim, Sungwoong Kim. [[Paper](https://arxiv.org/pdf/2004.09910.pdf)] [[Code](https://github.com/kakaobrain/torchgpipe)]
* ZeRO: Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. [[Paper](https://arxiv.org/pdf/1910.02054.pdf)] [[Code](https://github.com/microsoft/DeepSpeed)]
* Megatron-LM: Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro. [[Paper](https://arxiv.org/pdf/1909.08053.pdf)][[Code](https://github.com/NVIDIA/Megatron-LM)]
