# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import sys
from typing import List

# Check for user requirements before we import our code.
try:
    import pygit2
except ImportError:
    print("Error: please pip install pygit2 module to use wgit")
    sys.exit(1)

try:
    import pgzip
except ImportError:
    print("Error: please pip install pgzip module to use wgit")
    sys.exit(1)


from .repo import Repo
from .signal_sparsity import Algo, SignalSparsity, random_sparse_mask
from .signal_sparsity_profiling import EnergyConcentrationProfile
from .version import __version_tuple__

__version__ = ".".join([str(x) for x in __version_tuple__])
__all__: List[str] = []
