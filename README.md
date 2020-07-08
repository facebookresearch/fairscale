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
