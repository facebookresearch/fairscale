import typing
from typing import Any, Dict, NoReturn
import unittest

import hamcrest

from fair_dev.fairtest import common_assertions


def _throw(exception: Exception) -> NoReturn:
    """
    Throw an exception, useful inside a lambda.

    :param exception: the exception.
    """
    raise exception


class WhenCalledTest(unittest.TestCase):
    def test_when_called(self) -> None:
        def example(a: Any, b: Any, c: Any) -> Dict[str, Any]:
            return dict(a=a, b=b, c=c)

        hamcrest.assert_that(
            example,
            common_assertions.when_called(1, 2, c=3).matches(dict(a=1, b=2, c=3)),
        )

        common_assertions.assert_raises(
            lambda: hamcrest.assert_that(
                example,
                common_assertions.when_called(1, 2, c=3).matches(
                    dict(a=1, b=2, c=4),
                ),
            ),
            AssertionError,
            (
                r"Expected: <callable>\(1, 2, c=3\) => <{'a': 1, 'b': 2, 'c': 4}>"
                r"\s*"
                r"but: was =><{'a': 1, 'b': 2, 'c': 3}>"
            ),
        )

    def test_calling_method(self) -> None:
        class Example:
            def foo(self, a: Any, b: Any, c: Any) -> Dict[str, Any]:
                return dict(a=a, b=b, c=c)

        hamcrest.assert_that(
            Example(),
            common_assertions.calling_method("foo", 1, 2, c=3).matches(
                dict(a=1, b=2, c=3),
            ),
        )

        common_assertions.assert_raises(
            lambda: hamcrest.assert_that(
                Example(),
                common_assertions.calling_method("foo", 1, 2, c=3).matches(
                    dict(a=1, b=2, c=4),
                ),
            ),
            AssertionError,
            (
                r"Expected: <obj>.foo\(1, 2, c=3\) => <{'a': 1, 'b': 2, 'c': 4}>"
                r"\s*"
                r"but: was =><{'a': 1, 'b': 2, 'c': 3}>"
            ),
        )


class AssertMatchTest(unittest.TestCase):
    def test(self) -> None:
        common_assertions.assert_match("abc", "abc")

        hamcrest.assert_that(
            typing.cast(Any, lambda: common_assertions.assert_match("abc", "xyz")),
            hamcrest.raises(
                AssertionError,
                "Expected: 'xyz'",
            ),
        )


class AssertTruthTest(unittest.TestCase):
    def test(self) -> None:
        common_assertions.assert_true(True)
        common_assertions.assert_true("abc")
        common_assertions.assert_true(1)
        common_assertions.assert_true([1])

        hamcrest.assert_that(
            typing.cast(Any, lambda: common_assertions.assert_true(False)),
            hamcrest.raises(
                AssertionError,
            ),
        )

        hamcrest.assert_that(
            typing.cast(
                Any,
                lambda: common_assertions.assert_true("", reason="meh"),
            ),
            hamcrest.raises(
                AssertionError,
                "meh",
            ),
        )


class AssertFalseTest(unittest.TestCase):
    def test(self) -> None:
        common_assertions.assert_false(False)
        common_assertions.assert_false("")
        common_assertions.assert_false(0)
        common_assertions.assert_false([])

        hamcrest.assert_that(
            typing.cast(Any, lambda: common_assertions.assert_false(True)),
            hamcrest.raises(
                AssertionError,
            ),
        )

        hamcrest.assert_that(
            typing.cast(
                Any,
                lambda: common_assertions.assert_false("abc", reason="meh"),
            ),
            hamcrest.raises(
                AssertionError,
                "meh",
            ),
        )


class AssertRaisesTest(unittest.TestCase):
    def test_simple(self) -> None:
        common_assertions.assert_raises(
            lambda: _throw(ValueError("abc")),
            ValueError,
        )

        common_assertions.assert_raises(
            lambda: _throw(ValueError("abc")),
            ValueError,
            "abc",
        )

        # No exception.
        hamcrest.assert_that(
            typing.cast(
                Any,
                lambda: common_assertions.assert_raises(
                    lambda: (),
                    ValueError,
                ),
            ),
            hamcrest.raises(
                AssertionError,
                "No exception raised",
            ),
        )

        # Wrong exception type.
        hamcrest.assert_that(
            typing.cast(
                Any,
                lambda: common_assertions.assert_raises(
                    lambda: _throw(ValueError("abc 123")), IndexError, "abc [0-9]+"
                ),
            ),
            hamcrest.raises(
                AssertionError,
                "was raised instead",
            ),
        )

    def test_regex(self) -> None:
        common_assertions.assert_raises(
            lambda: _throw(ValueError("abc 123")),
            ValueError,
            "abc [0-9]+",
        )

        hamcrest.assert_that(
            typing.cast(
                Any,
                lambda: common_assertions.assert_raises(
                    lambda: _throw(ValueError("abc xyz")), ValueError, "abc [0-9]+"
                ),
            ),
            hamcrest.raises(
                AssertionError,
                "the expected pattern .* not found",
            ),
        )

    def test_matching(self) -> None:
        class ExampleException(ValueError):
            code: int

        e = ExampleException("abc 123")
        e.code = 123

        common_assertions.assert_raises(
            lambda: _throw(e),
            ValueError,
            matching=hamcrest.has_properties(code=123),
        )

        hamcrest.assert_that(
            typing.cast(
                Any,
                lambda: common_assertions.assert_raises(
                    lambda: _throw(e),
                    ValueError,
                    matching=hamcrest.has_properties(code=9),
                ),
            ),
            hamcrest.raises(
                AssertionError,
                "Correct assertion type .* but an object with .* not found",
            ),
        )
