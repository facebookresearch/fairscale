import unittest

import torch

from fair_dev.fairtest import random_utils, tensor_assertions


class WithRandomSeedTest(unittest.TestCase):
    def test(self) -> None:
        expected_state = torch.get_rng_state()

        with random_utils.with_random_seed(0):
            a = torch.rand(2, 3)

        with random_utils.with_random_seed(0):
            b = torch.rand(2, 3)

        tensor_assertions.assert_matches_tensor(a, b)

        tensor_assertions.assert_matches_tensor(
            torch.get_rng_state(),
            expected_state,
        )
