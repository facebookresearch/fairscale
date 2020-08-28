# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


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
