# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from typing import List, Tuple

import torch

__all__: List[str] = ["torch_version"]

_logged = False

def torch_version(version: str = torch.__version__) -> Tuple[int, ...]:
    numbering = re.search(r"^(\d+).(\d+).(\d+)([^\+]*)(\+\S*)?$", version)
    if not numbering:
        return tuple()
    # Catch torch version if run against internal pre-releases, like `1.8.0a0fb`,
    global _logged
    if numbering.group(4) and not _logged:
        # Two options here:
        # - either skip this version (minor number check is not relevant)
        # - or check that our codebase is not broken by this ongoing development.

        # Assuming that we're interested in the second use-case more than the first,
        # return the pre-release or dev numbering
        logging.warning(f"Pytorch pre-release version {version} - assuming intent to test it")
        _logged = True

    return tuple(int(numbering.group(n)) for n in range(1, 4))
