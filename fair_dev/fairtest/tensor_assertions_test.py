import unittest

import hamcrest
from hamcrest.core.string_description import StringDescription
import torch

from fair_dev.fairtest import common_assertions, tensor_assertions


def test_example() -> None:
    a = torch.ones(3)
    hamcrest.assert_that(
        a.split([1, 2]),
        hamcrest.contains_exactly(
            tensor_assertions.tensor_with_structure(size=[1], device="cpu"),
            tensor_assertions.tensor_with_structure(size=[2], device="cpu"),
        ),
    )


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
    def test_match_description(self) -> None:
        desc = StringDescription()
        tensor_assertions.tensor_with_structure(size=[2]).describe_match(torch.ones(2), desc)
        hamcrest.assert_that(
            str(desc),
            hamcrest.matches_regexp(
                r"tensor structure matched::\ntensor\(..., size=\[2\]\)",
            ),
        )

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

        for matcher in [
            tensor_assertions.tensor_with_structure(size=(5, 3)),
            tensor_assertions.tensor_with_size((5, 3)),
        ]:
            common_assertions.assert_raises(
                lambda: hamcrest.assert_that(a, matcher),
                AssertionError,
                r"(?ms)matching structure::.*\(..., size=\[5, 3\]\)"
                + r".*structure did not match::.*\(..., size=\[2, 3\]\)",
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

        for matcher in [
            tensor_assertions.tensor_with_structure(dtype=torch.float16),
            tensor_assertions.tensor_with_dtype(torch.float16),
        ]:
            common_assertions.assert_raises(
                lambda: hamcrest.assert_that(a, matcher),
                AssertionError,
                r"(?ms)matching structure::.*\(..., dtype=torch.float16\)"
                + r".*structure did not match::.*\(..., dtype=torch.int64\)",
            )

    def test_device(self) -> None:
        a = torch.ones(2, 3)

        hamcrest.assert_that(
            a,
            tensor_assertions.tensor_with_structure(device="cpu"),
        )
        hamcrest.assert_that(
            a,
            tensor_assertions.tensor_with_device("cpu"),
        )
        hamcrest.assert_that(
            a,
            tensor_assertions.tensor_with_structure(device=torch.device("cpu")),
        )

        for matcher in [
            tensor_assertions.tensor_with_structure(device="cuda"),
            tensor_assertions.tensor_with_device("cuda"),
        ]:
            common_assertions.assert_raises(
                lambda: hamcrest.assert_that(a, matcher),
                AssertionError,
                r"(?ms)matching structure::.*\(..., device='cuda'\)"
                + r".*structure did not match::.*\(..., device='cpu'\)",
            )

    def test_layout(self) -> None:
        a = torch.ones(2, 3)

        hamcrest.assert_that(
            a,
            tensor_assertions.tensor_with_structure(layout=torch.strided),
        )
        hamcrest.assert_that(
            a,
            tensor_assertions.tensor_with_layout(torch.strided),
        )

        for matcher in [
            tensor_assertions.tensor_with_layout(torch.sparse_coo),  # type: ignore
            tensor_assertions.tensor_with_structure(layout=torch.sparse_coo),  # type: ignore
        ]:
            common_assertions.assert_raises(
                lambda: hamcrest.assert_that(a, matcher),
                AssertionError,
                r"(?ms)matching structure::.*\(..., layout=torch.sparse_coo\)"
                + r".*structure did not match::.*\(..., layout=torch.strided\)",
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
            r"(?ms)matching structure::"
            + r".*\(..., size=\[5, 3\], dtype=torch.float16, device='cpu', layout=torch.strided\)"
            + r".*structure did not match::.*\(..., size=\[2, 3\], dtype=torch.int64\)",
        )


class TensorMatcherTest(unittest.TestCase):
    def test_match_description(self) -> None:
        desc = StringDescription()
        tensor_assertions.matches_tensor([1, 1]).describe_match(torch.ones(2), desc)
        hamcrest.assert_that(
            str(desc),
            hamcrest.matches_regexp(
                r"tensor structure matched::\n"
                + r"tensor\(..., size=\[2\], dtype=torch.float32, layout=torch.strided\)",
            ),
        )

    def test_bad_type(self) -> None:
        for bad in [
            lambda: hamcrest.assert_that(7, tensor_assertions.matches_tensor([1, 1])),
            lambda: tensor_assertions.assert_matches_tensor(7, [1, 1]),  # type: ignore
        ]:
            common_assertions.assert_raises(
                bad,
                AssertionError,
                "(?ms)Expected: an instance of Tensor.* but: was <7>",
            )

    def test_everything(self) -> None:
        a = torch.ones(2, dtype=torch.int64)
        hamcrest.assert_that(
            a,
            tensor_assertions.matches_tensor(
                [1, 1],
            ),
        )

    def test_structure(self) -> None:
        a = torch.ones(2)
        # bad size
        expected = [1.0, 1.0, 1.0]
        for bad in [
            lambda: hamcrest.assert_that(a, tensor_assertions.matches_tensor(expected)),
            lambda: tensor_assertions.assert_matches_tensor(a, expected),
        ]:
            common_assertions.assert_raises(
                bad,
                AssertionError,
                r"(?ms)matching structure and values::.*\(..., size=\[3\], dtype=torch.float32, layout=torch.strided\)"
                + r".*structure did not match::.*\(..., size=\[2\]\)",
            )

        # bad dtype
        expected = [1, 1]
        for bad in [
            lambda: hamcrest.assert_that(a, tensor_assertions.matches_tensor(expected)),
            lambda: tensor_assertions.assert_matches_tensor(a, expected),
        ]:
            common_assertions.assert_raises(
                bad,
                AssertionError,
                r"(?ms)matching structure and values::.*\(..., size=\[2\], dtype=torch.int64, layout=torch.strided\)"
                + r".*structure did not match::.*\(..., dtype=torch.float32\)",
            )

        b = a.to_sparse()
        for bad in [
            lambda: hamcrest.assert_that(b, tensor_assertions.matches_tensor(a)),
            lambda: tensor_assertions.assert_matches_tensor(b, a),
        ]:
            common_assertions.assert_raises(
                bad,
                AssertionError,
                r"(?ms)matching structure and values::.*\(..., size=\[2\], dtype=torch.float32, layout=torch.strided\)"
                + r".*structure did not match::.*\(..., layout=torch.sparse_coo\)",
            )

    def test_structure_cuda(self) -> None:
        # bad device
        a = torch.ones(2)
        b = a.to(device="cuda")
        expected = [1.0, 1.0]

        hamcrest.assert_that(
            b,
            tensor_assertions.matches_tensor(expected, ignore_device=True),
        )

        for bad in [
            lambda: hamcrest.assert_that(
                b,
                tensor_assertions.matches_tensor(
                    expected,
                    ignore_device=False,
                ),
            ),
            lambda: tensor_assertions.assert_matches_tensor(
                b,
                expected,
                ignore_device=False,
            ),
        ]:
            common_assertions.assert_raises(
                bad,
                AssertionError,
                r"(?ms)matching structure and values::.*\(..., size=\[2\], dtype=torch.float32, device='cpu', layout=torch.strided\)"
                + r".*structure did not match::.*\(..., device='cuda:.'\)",
            )

    def test_values(self) -> None:
        a = torch.ones(3, 2)
        expected = torch.zeros(3, 2)
        for bad in [
            lambda: hamcrest.assert_that(
                a,
                tensor_assertions.matches_tensor(
                    expected,
                ),
            ),
            lambda: tensor_assertions.assert_matches_tensor(
                a,
                expected,
            ),
        ]:
            common_assertions.assert_raises(
                bad,
                AssertionError,
                r"(?ms)matching structure and values::\n"
                + r".*\(..., size=\[3, 2\], dtype=torch.float32, layout=torch.strided\)"
                + r".*\[\[0"
                + r".*tensor values did not match::\n"
                + r".*\[\[1",
            )
