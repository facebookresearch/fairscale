import unittest

import hamcrest
import torch

from fair_dev import fairtest


class ExamplesTest(unittest.TestCase):
    def test_split(self) -> None:
        a = torch.ones(3)
        fairtest.assert_match(
            a.split([1, 2]),
            hamcrest.contains_exactly(
                fairtest.tensor_with_structure(size=[1], device="cpu"),
                fairtest.tensor_with_structure(size=[2], device="cpu"),
            ),
        )

    def test_random(self) -> None:
        with fairtest.with_random_seed(0):
            a = torch.rand(2, 3)

        with fairtest.with_random_seed(0):
            b = torch.rand(2, 3)

        fairtest.assert_matches_tensor(a, b)
