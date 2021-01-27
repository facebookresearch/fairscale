# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


def get_golden_real_stats():

    return {
        "reference_speed": 1430,
        "reference_memory": 1220,
        "reference_loss": 0.006,
    }


def get_golden_synthetic_stats():
    # TODO(anj-s): Add support for synthetic regression benchmarks
    raise NotImplementedError("Synthetic data benchmarks are not supported.")
