# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


""" Golden data used in unit tests. """

adascale_test_data = [
    # "input" value is a list of input tensors for micro-batch/rank 0 and micro-batch/rank 1.
    {
        "input": [[1.0, 0], [0, 1.0]],
        "expected_gain": 4.0 / 3,
        "expected_grad": [[0.5, 0.5], [0.5, 0.5]],
        "expected_bias_grad": [1.0, 1.0],
    },
    {
        "input": [[1.0, 1.0], [1.0, 1.0]],
        "expected_gain": 1.0000001249999846,
        "expected_grad": [[1.0, 1.0], [1.0, 1.0]],
        "expected_bias_grad": [1.0, 1.0],
    },
    {
        "input": [[-1.0, 1.0], [1.0, -1.0]],
        "expected_gain": 2.0,
        "expected_grad": [[0.0, 0.0], [0.0, 0.0]],
        "expected_bias_grad": [1.0, 1.0],
    },
    {
        "input": [[1.0, 4.0], [5.0, 0.5]],
        "expected_gain": 1.4688796680497926,
        "expected_grad": [[3.0, 2.25], [3.0, 2.25]],
        "expected_bias_grad": [1.0, 1.0],
    },
    {
        "input": [[-0.2, 3.0], [5.0, 0.5]],
        "expected_gain": 1.8472893901708,
        "expected_grad": [[2.4000000953674316, 1.75], [2.4000000953674316, 1.75]],
        "expected_bias_grad": [1.0, 1.0],
    },
    # "inputs" to trigger multiple iteration tests, which make sure the
    # smoothing factor calculation is also covered.
    {
        "inputs": [[[-0.2, 3.3], [5.2, 0.7]], [[1.0, 4.0], [3.1, 0.1]]],
        "expected_gain": 1.6720968158031417,
        "expected_grad": [[2.049999952316284, 2.049999952316284], [2.049999952316284, 2.049999952316284]],
        "expected_bias_grad": [1.0, 1.0],
    },
]

corr_mean_test_data = [
    {
        "inputs": [
            [[1.0, 0.0, 2.0], [2.0, 0.0, 1.0]],
            [[0.0, 1.0, 2.0], [2.0, 1.0, 0]],
            [[3.0, 1.0, 2.0], [2.0, 1.0, -1.0]],
        ],
        "expected_grad": [[1.5, 0.0, 1.5], [1.0, 1.0, 1.0], [2.5, 1.0, 0.5]],
        # expected pearson correlation of two micro-batches
        "expected_corr": [0.5, -1.0, 0.327327],
        "expected_cos_similarity": [float("nan"), 0.8165, 0.8433],
    }
]
