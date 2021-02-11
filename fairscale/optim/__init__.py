# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
:mod:`fairscale.optim` is a package implementing various torch optimization algorithms.
"""
import logging

from .adascale import AdaScale, AdaScaleWrapper
from .oss import OSS

try:
    from .adam import Adam, Precision
except ImportError:  # pragma: no cover
    pass  # pragma: no cover
try:
    from .grad_scaler import GradScaler
except ImportError:
    logging.warning("Torch AMP is not available on this platform")
