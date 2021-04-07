# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


def get_golden_real_stats():

    return {
        "reference_speed": 660,
        "reference_memory": 945,
        "reference_loss": 0.026,
    }


def get_golden_synthetic_stats():
    # TODO(anj-s): Add support for synthetic regression benchmarks
    raise NotImplementedError("Synthetic data benchmarks are not supported.")
