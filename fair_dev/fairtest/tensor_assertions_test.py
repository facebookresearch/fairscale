import unittest

import torch

from fair_dev.fairtest import common_assertions, tensor_assertions


class SameTensorStorageTest(unittest.TestCase):
    def test(self):
        a = torch.ones(2)
        a_view = a.split([1, 1])[0]
        b = torch.ones(2)

        common_assertions.assert_match(
            a,
            tensor_assertions.same_tensor_storage(a),
        )

        common_assertions.assert_match(
            a,
            tensor_assertions.same_tensor_storage(a_view),
        )

        common_assertions.assert_raises(
            lambda: common_assertions.assert_match(
                a,
                tensor_assertions.TensorStorageMatcher(b),
            ),
            AssertionError,
            (r"(?ms)same storage as.*but: was"),
        )


class AssertTensorsShareStorageTest(unittest.TestCase):
    def test(self):
        a = torch.ones(3, 2)
        tensor_assertions.assert_tensors_share_storage(
            a,
            *a.split([1, 2]),
        )

        common_assertions.assert_raises(
            lambda: tensor_assertions.assert_tensors_share_storage(
                a,
                torch.tensor([[1.0, 1.0]]),
                torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            ),
            AssertionError,
            (r"(?ms)same storage as.*but: was"),
        )


class AssertTensorStorageDiffersTest(unittest.TestCase):
    def test(self):
        a = torch.ones(2)
        a_view = a.split([1, 1])[0]
        b = torch.ones(2)

        tensor_assertions.assert_tensor_storage_differs(
            a,
            b,
        )

        common_assertions.assert_raises(
            lambda: tensor_assertions.assert_tensor_storage_differs(
                a,
                a_view,
            ),
            AssertionError,
            r"(?ms)not same storage as .* but: was",
        )
