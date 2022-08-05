import unittest

import hamcrest
import torch

from fair_dev.fairtest import common_assertions, tensor_assertions


class SameTensorStorageTest(unittest.TestCase):
    def test(self) -> None:
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
    def test(self) -> None:
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
    def test(self) -> None:
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


class MatchesTensorStructureTest(unittest.TestCase):
    def test_bad_type(self) -> None:
        common_assertions.assert_raises(
            lambda: hamcrest.assert_that(
                7,
                tensor_assertions.tensor_with_structure(
                    size=(5, 3),
                ),
            ),
            AssertionError,
            "(?ms)Expected: an instance of Tensor.* but: was <7>",
        )

    def test_size(self) -> None:
        a = torch.ones(2, 3)

        hamcrest.assert_that(
            a,
            tensor_assertions.tensor_with_structure(
                size=(2, 3),
            ),
        )
        hamcrest.assert_that(
            a,
            tensor_assertions.tensor_with_size((2, 3)),
        )
        hamcrest.assert_that(
            a,
            tensor_assertions.tensor_with_structure(
                size=torch.Size((2, 3)),
            ),
        )

        common_assertions.assert_raises(
            lambda: hamcrest.assert_that(
                a,
                tensor_assertions.tensor_with_structure(
                    size=(5, 3),
                ),
            ),
            AssertionError,
            r"(?ms)Expected: tensor\(size=\(5, 3\)\).*but: tensor\(\[size=\(2, 3\)\]\)",
        )
        common_assertions.assert_raises(
            lambda: hamcrest.assert_that(
                a,
                tensor_assertions.tensor_with_size((5, 3)),
            ),
            AssertionError,
            r"(?ms)Expected: tensor\(size=\(5, 3\)\).*but: tensor\(\[size=\(2, 3\)\]\)",
        )

    def test_dtype(self) -> None:
        a = torch.ones(2, 3, dtype=torch.int64)

        hamcrest.assert_that(
            a,
            tensor_assertions.tensor_with_structure(
                dtype=torch.int64,
            ),
        )
        hamcrest.assert_that(
            a,
            tensor_assertions.tensor_with_dtype(torch.int64),
        )

        common_assertions.assert_raises(
            lambda: hamcrest.assert_that(
                a,
                tensor_assertions.tensor_with_structure(
                    dtype=torch.float16,
                ),
            ),
            AssertionError,
            r"(?ms)Expected: tensor\(dtype=torch.float16\).*but: tensor\(\[dtype=torch.int64\]\)",
        )
        common_assertions.assert_raises(
            lambda: hamcrest.assert_that(
                a,
                tensor_assertions.tensor_with_dtype(torch.float16),
            ),
            AssertionError,
            r"(?ms)Expected: tensor\(dtype=torch.float16\).*but: tensor\(\[dtype=torch.int64\]\)",
        )

    def test_device(self) -> None:
        a = torch.ones(2, 3)

        hamcrest.assert_that(
            a,
            tensor_assertions.tensor_with_structure(
                device="cpu",
            ),
        )
        hamcrest.assert_that(
            a,
            tensor_assertions.tensor_with_device("cpu"),
        )
        hamcrest.assert_that(
            a,
            tensor_assertions.tensor_with_structure(
                device=torch.device("cpu"),
            ),
        )

        common_assertions.assert_raises(
            lambda: hamcrest.assert_that(
                a,
                tensor_assertions.tensor_with_structure(
                    device="cuda",
                ),
            ),
            AssertionError,
            r"(?ms)Expected: tensor\(device='cuda'\).*but: tensor\(\[device='cpu'\]\)",
        )
        common_assertions.assert_raises(
            lambda: hamcrest.assert_that(
                a,
                tensor_assertions.tensor_with_device("cuda"),
            ),
            AssertionError,
            r"(?ms)Expected: tensor\(device='cuda'\).*but: tensor\(\[device='cpu'\]\)",
        )

    def test_layout(self) -> None:
        a = torch.ones(2, 3)

        hamcrest.assert_that(
            a,
            tensor_assertions.tensor_with_structure(
                layout=torch.strided,
            ),
        )
        hamcrest.assert_that(
            a,
            tensor_assertions.tensor_with_layout(torch.strided),
        )

        common_assertions.assert_raises(
            lambda: hamcrest.assert_that(
                a,
                tensor_assertions.tensor_with_structure(
                    layout=torch.sparse_coo,  # type: ignore
                ),
            ),
            AssertionError,
            r"(?ms)Expected: tensor\(layout=torch.sparse_coo\).*but: tensor\(\[layout=torch.strided\]\)",
        )
        common_assertions.assert_raises(
            lambda: hamcrest.assert_that(
                a,
                tensor_assertions.tensor_with_layout(torch.sparse_coo),  # type: ignore
            ),
            AssertionError,
            r"(?ms)Expected: tensor\(layout=torch.sparse_coo\).*but: tensor\(\[layout=torch.strided\]\)",
        )

    def test_everything(self) -> None:
        a = torch.ones(2, 3, dtype=torch.int64)

        hamcrest.assert_that(
            a,
            tensor_assertions.tensor_with_structure(
                size=(2, 3),
                dtype=torch.int64,
                device="cpu",
                layout=torch.strided,
            ),
        )

        tensor_assertions.assert_tensor_structure(
            a,
            size=(2, 3),
            dtype=torch.int64,
            device="cpu",
            layout=torch.strided,
        )

        hamcrest.assert_that(
            a,
            tensor_assertions.tensor_with_structure(
                size=torch.Size((2, 3)),
                dtype=torch.int64,
                device=torch.device("cpu"),
                layout=torch.strided,
            ),
        )

        common_assertions.assert_raises(
            lambda: hamcrest.assert_that(
                a,
                tensor_assertions.tensor_with_structure(
                    size=(5, 3),  # mismatch
                    dtype=torch.float16,  # mismatch
                    device="cpu",
                    layout=torch.strided,
                ),
            ),
            AssertionError,
            (
                r"(?ms)Expected: tensor\(size=\(5, 3\), dtype=torch.float16, device='cpu', layout=torch.strided\)"
                + r".*"
                + r"but: tensor\(\[size=\(2, 3\)\], \[dtype=torch.int64\], device='cpu', layout=torch.strided\)"
            ),
        )
