from typing import Any, Optional, Sequence, Union

import hamcrest
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
import torch

from fair_dev.fairtest import common_assertions, tracebacks

# TensorConvertible is difficult to express in mypy
# This is a broken attempt that doesn't quite get there:
#
# int is not a Number?
# https://github.com/python/mypy/issues/3186
# https://stackoverflow.com/questions/69334475/how-to-hint-at-number-types-i-e-subclasses-of-number-not-numbers-themselv/69383462#69383462kk
#
# NumberLike = Union[numbers.Number, numbers.Complex, SupportsFloat]
#
# TensorConvertable = Union[
#    torch.Tensor,
#    NumberLike,
#    Sequence,
#    List,
#    Tuple,
#    npNDArray,
# ]
#
# Instead, we punt:
TensorConvertable = Any
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
    """
    This is a `hamcrest.same_instance(expected)` matcher, but for torch.Tensor storage.

    It has both expect match, and expect no match mechanics.
    """

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

    def _data_ptr(self, tensor: torch.Tensor) -> int:
        return tensor.storage().data_ptr()  # type: ignore

    def _matches(self, item: torch.Tensor) -> bool:
        return self.expect_same == (self._data_ptr(item) == self._data_ptr(self.expected))

    def describe_to(self, description: Description) -> None:
        if not self.expect_same:
            description.append_text("not ")

        description.append_text(
            f"same storage as {self._data_ptr(self.expected)} <{self.expected}>",
        )

    def describe_match(
        self,
        item: torch.Tensor,
        match_description: Description,
    ) -> None:
        match_description.append_text(f"was {self._data_ptr(item)} <{item}>")

    def describe_mismatch(
        self,
        item: torch.Tensor,
        mismatch_description: Description,
    ) -> None:
        mismatch_description.append_text(f"was {self._data_ptr(item)} <{item}>")


def same_tensor_storage(expected: torch.Tensor) -> TensorStorageMatcher:
    """
    Matcher constructor to match Tensors which share storage (are views of the same data).

    :param expected: the reference Tensor.
    :return: a `Matcher[Tensor]`.
    """
    return TensorStorageMatcher(expected)


def different_tensor_storage(expected: torch.Tensor) -> TensorStorageMatcher:
    """
    Matcher constructor to assert Tensors do not share storage (are not views of the same data).

    :param expected: the reference Tensor.
    :return: a `Matcher[Tensor]`.
    """
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


class TensorStructureMatcher(BaseMatcher[torch.Tensor]):
    """
    Matcher for comparing the structure of a tensor to an exemplar.

    The point of a unified matcher, over individual matchers, is
    unified descriptions of match expectations.

    Optionally Matches:
      - size
      - dtype
      - device
      - layout
    """

    size: Optional[torch.Size] = None
    dtype: Optional[torch.dtype] = None
    device: Optional[torch.device] = None
    layout: Optional[torch.layout] = None

    def __init__(
        self,
        size: Optional[Union[torch.Size, Sequence[int]]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str]] = None,
        layout: Optional[torch.layout] = None,
    ):
        if size:
            self.size = torch.Size(size)

        self.dtype = dtype

        if device:
            self.device = torch.device(device)

        self.layout = layout

    def _matches(
        self,
        item: Any,
    ) -> bool:
        hamcrest.assert_that(
            item,
            hamcrest.instance_of(torch.Tensor),
        )

        return all(
            (
                self.size is None or self.size == item.size(),
                self.dtype is None or self.dtype == item.dtype,
                self.device is None or self.device == item.device,
                self.layout is None or self.layout == item.layout,
            ),
        )

    def _describe_expected_structure(self, description: Description) -> str:
        parts = []
        if self.size is not None:
            parts.append(f"size={list(self.size)}")

        if self.dtype is not None:
            parts.append(f"dtype={self.dtype}")

        if self.device is not None:
            parts.append(f"device='{self.device}'")

        if self.layout is not None:
            parts.append(f"layout={self.layout}")

        return f"tensor(..., {', '.join(parts)})"

    def describe_to(self, description: Description) -> None:
        description.append_text("tensor matching structure::\n")
        description.append_text(self._describe_expected_structure(description))

    def _describe_structure_mismatch(
        self,
        item: torch.Tensor,
        *,
        filter_match_fields: bool = True,
    ) -> str:

        parts = []

        if self.size is not None and (not filter_match_fields or item.size() != self.size):
            parts.append(f"size={list(item.size())}")

        if self.dtype is not None and (not filter_match_fields or item.dtype != self.dtype):
            parts.append(f"dtype={item.dtype}")

        if self.device is not None and (not filter_match_fields or item.device != self.device):
            parts.append(f"device='{item.device}'")

        if self.layout is not None and (not filter_match_fields or item.layout != self.layout):
            parts.append(f"layout={item.layout}")

        return f"tensor(..., {', '.join(parts)})"

    def describe_match(
        self,
        item: torch.Tensor,
        match_description: Description,
    ) -> None:
        match_description.append_text("tensor structure matched::\n")
        match_description.append_text(
            self._describe_structure_mismatch(
                item,
                filter_match_fields=False,
            ),
        )

    def describe_mismatch(
        self,
        item: torch.Tensor,
        mismatch_description: Description,
    ) -> None:
        mismatch_description.append_text("tensor structure did not match::\n")
        mismatch_description.append_text(self._describe_structure_mismatch(item))


def tensor_with_structure(
    size: Optional[Union[torch.Size, Sequence[int]]] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None,
    layout: Optional[torch.layout] = None,
) -> TensorStructureMatcher:
    """
    Match on the structure of a Tensor.

    :param size: (Optional) the expected size/shape.
    :param dtype: (Optional) the expected dtype.
    :param device: (Optional) the expected torch.device (or str name).
    :param layout: (optional) the expected torch.layout.
    :return: a Matcher[Tensor].
    """
    return TensorStructureMatcher(
        size=size,
        dtype=dtype,
        device=device,
        layout=layout,
    )


def assert_tensor_structure(
    actual: torch.Tensor,
    size: Optional[Union[torch.Size, Sequence[int]]] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None,
    layout: Optional[torch.layout] = None,
) -> None:
    """
    Assert that a tensor matches the expected structure.

    :param actual: the Tensor.
    :param size: (Optional) the expected size/shape.
    :param dtype: (Optional) the expected dtype.
    :param device: (Optional) the expected torch.device (or str name).
    :param layout: (optional) the expected torch.layout.
    """
    common_assertions.assert_match(
        actual,
        tensor_with_structure(
            size=size,
            dtype=dtype,
            device=device,
            layout=layout,
        ),
    )


def tensor_with_size(
    size: Union[torch.Size, Sequence[int]],
) -> TensorStructureMatcher:
    """
    Match a Tensor with the given size.

    :param size: the expected size/shape.
    :return: a Matcher[Tensor].
    """
    return TensorStructureMatcher(
        size=size,
    )


def tensor_with_dtype(
    dtype: torch.dtype,
) -> TensorStructureMatcher:
    """
    Match a Tensor with the given dtype.

    :param dtype: the expected dtype.
    :return: a Matcher[Tensor].
    """
    return TensorStructureMatcher(
        dtype=dtype,
    )


def tensor_with_device(
    device: Union[torch.device, str],
) -> TensorStructureMatcher:
    """
    Match a Tensor with the given device.

    :param device: the expected device.
    :return: a Matcher[Tensor].
    """
    return TensorStructureMatcher(
        device=device,
    )


def tensor_with_layout(
    layout: torch.layout,
) -> TensorStructureMatcher:
    """
    Match a Tensor with the given layout.

    :param layout: the expected layout.
    :return: a Matcher[Tensor].
    """
    return TensorStructureMatcher(
        layout=layout,
    )


class TensorMatcher(TensorStructureMatcher):
    """
    Matcher for comparing the structure and values of a Tensor.
    """

    expected: torch.Tensor
    close: bool

    def __init__(
        self,
        expected: TensorConvertable,
        *,
        close: bool = True,
        ignore_device: bool = True,
        ignore_layout: bool = False,
    ):
        self.expected = torch.as_tensor(expected)

        self.close = close

        device = self.expected.device
        if ignore_device:
            device = None

        layout = self.expected.layout
        if ignore_layout:
            layout = None

        super().__init__(
            size=self.expected.size(),
            dtype=self.expected.dtype,
            device=device,
            layout=layout,
        )

    def _matches(self, item: torch.Tensor) -> bool:
        if not super()._matches(item):
            return False

        if item.device != self.expected.device:
            # honor the ignore_device semantics from the base class.
            item = item.clone().detach().to(device=self.expected.device)

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
            # TODO: handle is_sparse.
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
        description.append_text("tensor matching structure and values::\n")
        description.append_text(self._describe_expected_structure(description))
        description.append_text("\n")
        description.append_text(self.expected.cpu().numpy())
        description.append_text("\n")

    # describe_match() fall-through.

    def describe_mismatch(
        self,
        item: torch.Tensor,
        mismatch_description: Description,
    ) -> None:
        if not super(TensorMatcher, self)._matches(item):
            super(TensorMatcher, self).describe_mismatch(
                item,
                mismatch_description,
            )
            return

        mismatch_description.append_text("tensor values did not match::\n")
        mismatch_description.append_text(item.cpu().numpy())

        # TODO: structural value diff


def matches_tensor(
    expected: TensorConvertable,
    *,
    close: bool = True,
    ignore_device: bool = True,
    ignore_layout: bool = False,
) -> TensorMatcher:
    """
    Construct a tensor structure and value matcher.

    :param expected: the expected values.
    :param close: should we do "close" matching, or exact?
    :param ignore_device: Should we ignore the device of the expected Tensor?
    :param ignore_layout: Should we ignore the layout of the expected Tensor?
    :return: a TensorMatcher.
    """
    return TensorMatcher(
        expected=expected,
        close=close,
        ignore_device=ignore_device,
        ignore_layout=ignore_layout,
    )


def assert_matches_tensor(
    actual: torch.Tensor,
    expected: TensorConvertable,
    *,
    close: bool = True,
    ignore_device: bool = True,
    ignore_layout: bool = False,
) -> None:
    """
    Assert that a Tensor matches expected values and structure.

    :param actual: the Tensor.
    :param expected: the expected values.
    :param close: should we do "close" matching, or exact?
    :param ignore_device: Should we ignore the device of the expected Tensor?
    :param ignore_layout: Should we ignore the layout of the expected Tensor?
    """
    common_assertions.assert_match(
        actual,
        matches_tensor(
            expected=expected,
            close=close,
            ignore_device=ignore_device,
            ignore_layout=ignore_layout,
        ),
    )
