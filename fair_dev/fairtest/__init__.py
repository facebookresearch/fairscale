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
from fair_dev.fairtest.tracebacks import hide_module_tracebacks
from fair_dev.fairtest.warnings import ignore_warnings

hide_module_tracebacks(hamcrest.core.base_matcher.__dict__)

__all__ = [
    "assert_close_to",
    "assert_match",
    "assert_raises",
    "assert_true",
    "calling_method",
    "close_to",
    "hide_module_tracebacks",
    "when_called",
]
