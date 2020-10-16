# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
:mod:`fairgc.optim` is a package implementing various torch optimization algorithms.
"""

try:
    from .adam import Adam, Precision
except ImportError:  # pragma: no cover
    pass  # pragma: no cover
from .adascale import AdaScale  # type: ignore
from .grad_scaler import GradScaler
from .oss import OSS
