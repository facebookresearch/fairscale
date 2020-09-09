# fairscale
fairscale is a PyTorch extension library for high performance and large scale training.

fairscale supports:
* pipeline parallelism (fairscale.nn.Pipe)
* tensor parallelism (fairscale.nn.model_parallel)
* optimizer state sharding (fairscale.optim.oss)

## Examples

Run a 4-layer model on 2 GPUs. The first two layers run on cuda:0 and the next two layers run on cuda:1.

```bash
import torch

import fairscale

model = torch.nn.Sequential(a, b, c, d)
model = fairscale.nn.Pipe(model, balance=[2, 2], devices=[0, 1], chunks=8)
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

* torchgpipe: Chiheon Kim, Heungsub Lee, Myungryong Jeong, Woonhyuk Baek, Boogeon Yoon, Ildoo Kim, Sungbin Lim, Sungwoong Kim. [[Paper](https://arxiv.org/pdf/abs/2004.09910)] [[Code](https://github.com/kakaobrain/torchgpipe)]
* ZeRO: Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. [[Paper](https://arxiv.org/abs/1910.02054)] [[Code](https://github.com/microsoft/DeepSpeed)]
* Megatron-LM: Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro. [Paper](https://arxiv.org/abs/1909.08053)[Code](https://github.com/NVIDIA/Megatron-LM)
