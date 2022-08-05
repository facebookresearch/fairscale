import numbers
import typing

import hamcrest
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
import torch

from fair_dev.fairtest import common_assertions, tracebacks

# int is not a Number?
# https://github.com/python/mypy/issues/3186
# https://stackoverflow.com/questions/69334475/how-to-hint-at-number-types-i-e-subclasses-of-number-not-numbers-themselv/69383462#69383462kk

NumberLike = typing.Union[numbers.Number, numbers.Complex, typing.SupportsFloat]

TensorConvertable = typing.Any

# TensorConvertable = typing.Union[
#    torch.Tensor,
#    NumberLike,
#    typing.Sequence,
#    typing.List,
#    typing.Tuple,
#    nptyping.NDArray,
# ]
"Types which torch.as_tensor(T) can convert."


def hide_tracebacks(mode: bool = True) -> None:
    """
    Hint that some unittest stacks (unittest, pytest) should remove
    frames from tracebacks that include this module.

    :param mode: optional, the traceback mode.
    """
    # globals(), called within the module, grabs the module dict.
    tracebacks.hide_module_tracebacks(globals(), mode)


# default to hiding.
hide_tracebacks(True)


class TensorStorageMatcher(BaseMatcher[torch.Tensor]):
    expect_same: bool
    expected: torch.Tensor

    def __init__(
        self,
        expected: torch.Tensor,
        *,
        expect_same: bool = True,
    ) -> None:
        self.expected = expected
        self.expect_same = expect_same

    def _data_ptr(self, tensor: torch.Tensor):
        return tensor.storage().data_ptr()  # type: ignore

    def _matches(self, item: torch.Tensor) -> bool:
        return self.expect_same == (self._data_ptr(item) == self._data_ptr(self.expected))

    def describe_to(self, description: Description) -> None:
        if not self.expect_same:
            description.append_text("not ")

        description.append_text(
            f"same storage as {self._data_ptr(self.expected)} <{self.expected}>",
        )

    def describe_mismatch(
        self,
        item: torch.Tensor,
        mismatch_description: Description,
    ) -> None:
        mismatch_description.append_text(f"was {self._data_ptr(item)} <{item}>")


def same_tensor_storage(expected: torch.Tensor) -> TensorStorageMatcher:
    return TensorStorageMatcher(expected)


def different_tensor_storage(expected: torch.Tensor) -> TensorStorageMatcher:
    return TensorStorageMatcher(expected, expect_same=False)


def assert_tensors_share_storage(actual: torch.Tensor, *views: torch.Tensor) -> None:
    """
    Assert that each tensor is a view of the same storage.

    :param views: a series of child Tensors which must all be views of source.
    """
    for t in views:
        common_assertions.assert_match(
            t,
            same_tensor_storage(actual),
        )


def assert_tensor_storage_differs(
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    """
    Assert that two tensors are not views of each other, and have different storage.

    :param actual: the tensor.
    :param reference: the reference tensor.
    """
    common_assertions.assert_match(
        actual,
        different_tensor_storage(reference),
    )


class TensorStructureMatcher(BaseMatcher):
    """
    PyHamcrest matcher for comparing the structure of a tensor to an exemplar.

    Matches:
      - device
      - size
      - dtype
      - layout
    """

    expected: torch.Tensor

    def __init__(self, expected: TensorConvertable) -> None:
        self.expected = torch.as_tensor(expected)

    def _matches(self, item: typing.Any) -> bool:
        # Todo: structural miss-match that still shows expected tensor.

        try:
            common_assertions.assert_match(
                item.device,
                self.expected.device,
            )
            common_assertions.assert_match(
                item.size(),
                self.expected.size(),
            )
            common_assertions.assert_match(
                item.dtype,
                self.expected.dtype,
            )
            common_assertions.assert_match(
                item.layout,
                self.expected.layout,
            )
            return True
        except AssertionError:
            return False

    def describe_to(self, description: Description) -> None:
        description.append_description_of(self.expected)


def matches_tensor_structure(
    expected: TensorConvertable,
) -> TensorStructureMatcher:
    """
    Construct a matcher for comparing the structure of a tensor to an exemplar.

    Matches on:
      - device
      - size
      - dtype
      - layout

    :return: a matcher.
    """
    return TensorStructureMatcher(expected)


def assert_tensor_structure(
    actual: torch.Tensor,
    expected: TensorConvertable,
) -> None:
    """
    Assert that the `actual` matches the structure (not data) of the `expected`.

    :param actual: a tensor.
    :param expected: an expected structure.
    """
    hamcrest.assert_that(
        actual,
        matches_tensor_structure(expected),
    )


class TensorMatcher(TensorStructureMatcher):
    """
    PyHamcrest matcher for comparing the structure and data a tensor to an exemplar.

    Matches:
      - device
      - size
      - dtype
      - layout
    """

    close: bool = False
    "Should <close> values be considered identical?"

    def __init__(
        self,
        expected: TensorConvertable,
        *,
        close: bool = False,
    ):
        super().__init__(expected=expected)
        self.close = close

        if self.expected.is_sparse and not self.expected.is_coalesced():  # type: ignore
            self.expected = self.expected.coalesce()

    def _matches(self, item: typing.Any) -> bool:
        if not super()._matches(item):
            return False

        if self.close:
            try:
                torch.testing.assert_close(
                    item,
                    self.expected,
                    equal_nan=True,
                )
                return True
            except AssertionError:
                return False

        else:
            if self.expected.is_sparse:  # type: ignore
                common_assertions.assert_true(item.is_sparse)

                # TODO: it may be necessary to sort the indices and values.
                if not item.is_coalesced():
                    item = item.coalesce()

                assert_tensor_equals(
                    item.indices(),
                    self.expected.indices(),
                )
                assert_tensor_equals(
                    item.values(),
                    self.expected.values(),
                )
                return True
            else:
                # torch.equal(item, self.expected) does not support nan.
                try:
                    torch.testing.assert_close(
                        item,
                        self.expected,
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    )
                except AssertionError:
                    return False
                return True

    def describe_to(self, description: Description) -> None:
        description.append_text("\n")
        description.append_description_of(self.expected)

    def describe_match(self, item: typing.Any, match_description: Description) -> None:
        match_description.append_text("was \n")
        match_description.append_description_of(item)

    def describe_mismatch(self, item: typing.Any, mismatch_description: Description) -> None:
        torch.set_printoptions(
            precision=10,
        )
        mismatch_description.append_text("was \n")
        mismatch_description.append_description_of(item)


def matches_tensor(
    expected: TensorConvertable,
    close: bool = False,
) -> TensorMatcher:
    """
    Returns a matcher for structure and value of a Tensor.

    :param expected: the expected Tensor.
    :param close: should *close* values be acceptable?
    """
    return TensorMatcher(expected, close=close)


def assert_tensor_equals(
    actual: torch.Tensor,
    expected: TensorConvertable,
    *,
    close: bool = False,
    view_of: typing.Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Assert that the `actual` tensor equals the `expected` tensor.

    :param actual: the actual tensor.
    :param expected: the value (to coerce to a Tensor) to compare to.
    :param close: should *close* values match?
    :param view_of: if present, also check that actual is a view of the reference Tensor.
    :returns: the `actual` value.
    """
    hamcrest.assert_that(
        actual,
        matches_tensor(
            expected,
            close=close,
        ),
    )
    if view_of is not None:
        assert_tensors_share_storage(view_of, actual)

    return actual


def match_tensor_sequence(
    *expected: TensorConvertable,
) -> Matcher:
    """
    Returns a matcher which expects a sequence that matches the tensors.
    :param expected: the expected tensors.
    """
    return hamcrest.contains_exactly(*[matches_tensor(e) for e in expected])


def assert_tensor_sequence_equals(
    actual: typing.Sequence[torch.Tensor],
    *expected: TensorConvertable,
    view_of: typing.Optional[torch.Tensor] = None,
) -> typing.Sequence[torch.Tensor]:
    """
    Assert that the `actual` is a sequence that equals the given `expected` tensor values.

    :param actual: the `actual` to test.
    :param expected: the expected values.
    :param view_of: if present, also check that actual is a view of the reference Tensor.
    :return: the `actual`
    """
    hamcrest.assert_that(
        actual,
        match_tensor_sequence(*expected),
    )
    if view_of is not None:
        assert_tensors_share_storage(view_of, *actual)
    return actual
