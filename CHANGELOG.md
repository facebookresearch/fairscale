# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [next rel] - TBD
### Added
- Pytorch compatibility for OSS checkpoints (#310)
- Elastic checkpoints for OSS, world size can vary in between save and loads (#310)
- Tensor views for OSS bucketing, reduced CPU use (#300)
- Bucket calls in ShardedDDP, for faster inter node communications (#327)

## [0.1.4] - 2021-01-07
### Fixed
- Missing cu files in the pip package


## [0.1.3] - 2021-01-04
### Fixed
- Release numbering within python and from pypi

## [0.1.2] - 2021-01-04
### Added
- AdaScale:
  . Added gradient accumulation feature (#202)
  . Added support of torch.lr_scheduler (#229)
  . Added support for add_param_groups (#266)
  . Added support for scale != world_size (#266)

### Fixed
- AdaScale: smoothing factor value fixed when using gradient accumulation (#235)
- Pipe: documentation on balancing functions (#243)
- ShardedDDP: handle typical NLP models
- ShardedDDP: better partitioning when finetuning


## [0.1.1] - 2020-12-01
### Fixed
- make sure pip package includes header files (#221)

## [0.1.0] - 2020-12-01
### Added
- ShardedDataParallel with autoreduce (#157)
- cpu support for Pipe (#188)
- ShardedOptim: Distributed Grad Scaler (for torch AMP)  (#182)
- OSS-aware clip grads, bridge sharded states (#167)
- oss: add rank_local_state_dict staticmethod (#174)
- support for PyTorch 1.7.0 (#171)
- Add implementation of AdaScale (#139)

### Fixed
- pip package install (#196, #200)

## [0.0.3] - 2020-10-14
### Added
- multi-process pipe

### Fixed
- multiple OSS fixes
- MegaTron+OSS DDP fix

## [0.0.2] - 2020-08-28
### Added
- add ddp that works with oss with reduce() not all_reduce() (#19)
- support for PyTorch v1.6
- add mixed precision Adam (#40)
- Adam optimizer state scaling (#44)

### Fixed
- properly restore a sharded optim state (#39)
- OSS restore state to proper device (#46)
- optim/oss: support optimizers with additional step kwargs (#53)
- optim/oss: fix state cast (#56)
- fix eval for oss_ddp (#55)
- optim/oss: work correctly with LRScheduler (#58)

## [0.0.1] - 2020-07-31
- Initial release.
