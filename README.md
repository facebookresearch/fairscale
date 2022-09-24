![FairScale Logo](./docs/source/_static/img/fairscale-logo.png)

[![Support Ukraine](https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB)](https://opensource.facebook.com/support-ukraine)
![PyPI](https://img.shields.io/pypi/v/fairscale)
[![Documentation Status](https://readthedocs.org/projects/fairscale/badge/?version=latest)](https://fairscale.readthedocs.io/en/latest/?badge=latest)
[![CircleCI](https://circleci.com/gh/facebookresearch/fairscale.svg?style=shield)](https://app.circleci.com/pipelines/github/facebookresearch/fairscale/) ![PyPI - License](https://img.shields.io/pypi/l/fairscale) [![Downloads](https://pepy.tech/badge/fairscale)](https://pepy.tech/project/fairscale) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/facebookresearch/fairscale/blob/main/CONTRIBUTING.md)
--------------------------------------------------------------------------------

## Description
FairScale is a PyTorch extension library for high performance and large scale training.
This library extends basic PyTorch capabilities while adding new SOTA scaling techniques.
FairScale makes available the latest distributed training techniques in the form of composable
modules and easy to use APIs. These APIs are a fundamental part of a researcher's toolbox as
they attempt to scale models with limited resources.

FairScale was designed with the following values in mind:

* **Usability** -  Users should be able to understand and use FairScale APIs with minimum cognitive overload.

* **Modularity** - Users should be able to combine multiple FairScale APIs as part of their training loop seamlessly.

* **Performance** - FairScale APIs provide the best performance in terms of scaling and efficiency.

## Watch Introductory Video

[![Explain Like I’m 5: FairScale](https://img.youtube.com/vi/oDt7ebOwWIc/0.jpg)](https://www.youtube.com/watch?v=oDt7ebOwWIc)

## Installation

To install FairScale, please see the following [instructions](https://github.com/facebookresearch/fairscale/blob/main/docs/source/installation_instructions.rst).
You should be able to install a package with pip or conda, or build directly from source.

## Getting Started
The full [documentation](https://fairscale.readthedocs.io/) contains instructions for getting started, deep dives and tutorials about the various FairScale APIs.

## FSDP

FullyShardedDataParallel (FSDP) is the recommended method for scaling to large NN models.
This library has been [upstreamed to PyTorch](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/).
The version of FSDP here is for historical references as well as for experimenting with
new and crazy ideas in research of scaling techniques. Please see the following blog
for [how to use FairScale FSDP and how does it work](https://engineering.fb.com/2021/07/15/open-source/fsdp/).

## Testing

We use circleci to test FairScale with the following PyTorch versions (with CUDA 11.2):
* the latest stable release (e.g. 1.10.0)
* the latest LTS release (e.g. 1.8.1)
* a recent nightly release (e.g. 1.11.0.dev20211101+cu111)

Please create an [issue](https://github.com/facebookresearch/fairscale/issues) if you are having trouble with installation.

## Contributors

We welcome contributions! Please see the [CONTRIBUTING](CONTRIBUTING.md) instructions for how you can contribute to FairScale.

## License

FairScale is licensed under the [BSD-3-Clause License](LICENSE).

fairscale.nn.pipe is forked from [torchgpipe](https://github.com/kakaobrain/torchgpipe), Copyright 2019, Kakao Brain, licensed under [Apache License](http://www.apache.org/licenses/LICENSE-2.0).

fairscale.nn.model_parallel is forked from [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), Copyright 2020, NVIDIA CORPORATION, licensed under [Apache License](http://www.apache.org/licenses/LICENSE-2.0).

fairscale.optim.adascale is forked from [AdaptDL](https://github.com/petuum/adaptdl), Copyright 2020, Petuum, Inc., licensed under [Apache License](http://www.apache.org/licenses/LICENSE-2.0).

fairscale.nn.misc.flatten_params_wrapper is forked from [PyTorch-Reparam-Module](https://github.com/SsnL/PyTorch-Reparam-Module), Copyright 2018, Tongzhou Wang, licensed under [MIT License](https://github.com/SsnL/PyTorch-Reparam-Module/blob/master/LICENSE).


## Citing FairScale

If you use FairScale in your publication, please cite it by using the following BibTeX entry.

```BibTeX
@Misc{FairScale2021,
  author =       {FairScale authors},
  title =        {FairScale:  A general purpose modular PyTorch library for high performance and large scale training},
  howpublished = {\url{https://github.com/facebookresearch/fairscale}},
  year =         {2021}
}
```
