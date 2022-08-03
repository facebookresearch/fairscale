# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Import most common subpackages
################################################################################

from typing import List

# Don't import sub-modules as experimental stuff otherwise gets imported directly
# when user does an `import fairscale`. This can cause experimental code's import
# dependencies (like pygit2) to leak into the fairscale main dependency.

__all__: List[str] = []
