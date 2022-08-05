import random
import unittest

import numpy as np
import torch

from fair_dev.fairtest import common_assertions, random_utils, tensor_assertions


class WithRandomSeedTest(unittest.TestCase):
    def test(self) -> None:
        expected_py_state = random.getstate()
        expected_torch_state = torch.get_rng_state()
        expected_np_state = np.random.get_state()

        # comparing numpy structures is still hard,
        # capture a side-effect for later.
        j = np.random.rand(2, 3)
        np.random.set_state(expected_np_state)

        with random_utils.with_random_seed(0):
            a = torch.rand(2, 3)
            x = np.random.rand(2, 3)
            m = random.random()

        with random_utils.with_random_seed(0):
            b = torch.rand(2, 3)
            y = np.random.rand(2, 3)
            n = random.random()

        with random_utils.with_random_seed(0):
            state = random_utils.CommonRngState.get_state()

        with random_utils.with_random_seed(state=state):
            c = torch.rand(2, 3)
            z = np.random.rand(2, 3)
            o = random.random()

        tensor_assertions.assert_matches_tensor(a, b)
        tensor_assertions.assert_matches_tensor(a, c)

        common_assertions.assert_true((x == y).all())
        common_assertions.assert_true((x == z).all())

        common_assertions.assert_match(m, n)
        common_assertions.assert_match(m, o)

        common_assertions.assert_match(
            random.getstate(),
            expected_py_state,
        )

        tensor_assertions.assert_matches_tensor(
            torch.get_rng_state(),
            expected_torch_state,
        )

        # TODO: nice numpy matchers similar to the Tensor matchers.
        k = np.random.rand(2, 3)
        common_assertions.assert_true((j == k).all())
