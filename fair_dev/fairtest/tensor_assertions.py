import numbers
from typing import Any, Optional, Sequence, SupportsFloat, Union

import hamcrest
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
import torch

from fair_dev.fairtest import common_assertions, tracebacks

# int is not a Number?
# https://github.com/python/mypy/issues/3186
# https://stackoverflow.com/questions/69334475/how-to-hint-at-number-types-i-e-subclasses-of-number-not-numbers-themselv/69383462#69383462kk

NumberLike = Union[numbers.Number, numbers.Complex, SupportsFloat]

TensorConvertable = Any

# TensorConvertable = Union[
#    torch.Tensor,
#    NumberLike,
#    Sequence,
#    List,
#    Tuple,
#    npNDArray,
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

    def _matches(self, item: Any) -> bool:
        hamcrest.assert_that(
            item,
            hamcrest.instance_of(torch.Tensor),
        )

        if self.size is not None and self.size != item.size():
            return False

        if self.dtype is not None and self.dtype != item.dtype:
            return False

        if self.device is not None and self.device != item.device:
            return False

        if self.layout is not None and self.layout != item.layout:
            return False

        return True

    def describe_to(self, description: Description) -> None:
        parts = []
        if self.size is not None:
            parts.append(f"size={tuple(self.size)}")

        if self.dtype is not None:
            parts.append(f"dtype={self.dtype}")

        if self.device is not None:
            parts.append(f"device='{self.device}'")

        if self.layout is not None:
            parts.append(f"layout={self.layout}")

        description.append_text(f"tensor({', '.join(parts)})")

    def describe_mismatch(
        self,
        item: torch.Tensor,
        mismatch_description: Description,
    ) -> None:
        def emphasize(test: bool, desc: str) -> str:
            if test:
                return f"[{desc}]"
            return desc

        parts = []
        if self.size is not None:
            parts.append(
                emphasize(
                    item.size() != self.size,
                    f"size={tuple(item.size())}",
                )
            )

        if self.dtype is not None:
            parts.append(
                emphasize(
                    item.dtype != self.dtype,
                    f"dtype={item.dtype}",
                )
            )

        if self.device is not None:
            parts.append(
                emphasize(
                    item.device != self.device,
                    f"device='{item.device}'",
                )
            )

        if self.layout is not None:
            parts.append(
                emphasize(
                    item.layout != self.layout,
                    f"layout={item.layout}",
                )
            )

        mismatch_description.append_text(f"tensor({', '.join(parts)})")


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
