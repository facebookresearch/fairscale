import hamcrest

from fair_dev.fairtest.common_assertions import (
    assert_false,
    assert_match,
    assert_raises,
    assert_true,
    calling_method,
    when_called,
)
from fair_dev.fairtest.numeric_assertions import assert_close_to, close_to
from fair_dev.fairtest.random_utils import CommonRngState, set_random_seed, with_random_seed
from fair_dev.fairtest.tensor_assertions import (
    assert_matches_tensor,
    assert_tensor_storage_differs,
    assert_tensor_structure,
    assert_tensors_share_storage,
    different_tensor_storage,
    matches_tensor,
    same_tensor_storage,
    tensor_with_device,
    tensor_with_dtype,
    tensor_with_layout,
    tensor_with_size,
    tensor_with_structure,
)
from fair_dev.fairtest.tracebacks import hide_module_tracebacks
from fair_dev.fairtest.warnings import ignore_warnings

hide_module_tracebacks(hamcrest.core.base_matcher.__dict__)

__all__ = [
    # common_assertions
    "assert_false",
    "assert_match",
    "assert_raises",
    "assert_true",
    "calling_method",
    "when_called",
    # numeric_assertions
    "assert_close_to",
    "close_to",
    # random_utils
    "CommonRngState",
    "set_random_seed",
    "with_random_seed",
    # tensor_assertions
    "assert_matches_tensor",
    "assert_tensor_storage_differs",
    "assert_tensor_structure",
    "assert_tensors_share_storage",
    "different_tensor_storage",
    "matches_tensor",
    "same_tensor_storage",
    "tensor_with_device",
    "tensor_with_dtype",
    "tensor_with_layout",
    "tensor_with_size",
    "tensor_with_structure",
    # tracebacks
    "hide_module_tracebacks",
    # warnings
    "ignore_warnings",
]
