# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


def get_golden_real_stats():

    return {
        "reference_speed": 660,
        "reference_memory": 945,
        "reference_loss": 0.026,
    }


def get_golden_synthetic_stats():
    # TODO(anj-s): Add support for synthetic regression benchmarks
    raise NotImplementedError("Synthetic data benchmarks are not supported.")
