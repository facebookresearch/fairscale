from dataclasses import dataclass
import typing
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type

import hamcrest
from hamcrest.core.assert_that import _assert_bool, _assert_match
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher

from fair_dev.fairtest.tracebacks import hide_module_tracebacks


def hide_tracebacks(mode: bool = True) -> None:
    """
    Hint that some unittest stacks (unittest, pytest) should remove
    frames from tracebacks that include this module.

    :param mode: optional, the traceback mode.
    """
    # globals(), called within the module, grabs the module dict.
    hide_module_tracebacks(globals(), mode)


# default to hiding.
hide_tracebacks(True)


@dataclass
class WhenCalledMatcher(BaseMatcher[Callable[..., Any]]):
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    method: Optional[str] = None

    def __init__(
        self,
        args: Sequence[Any],
        kwargs: Dict[str, Any],
        matcher: Matcher,
        method: Optional[str] = None,
    ) -> None:
        self.args = tuple(args)
        self.kwargs = dict(kwargs)
        self.matcher = matcher
        self.method = method

    def _matches(self, item: Callable) -> bool:
        return self.matcher.matches(self._call_item(item))

    def _call_item(self, item: Callable) -> Any:
        if self.method is None:
            f = item
        else:
            f = getattr(item, self.method)

        return f(*self.args, **self.kwargs)

    def describe_to(self, description: Description) -> None:
        call_params = ", ".join(
            tuple([repr(a) for a in self.args] + [f"{k}={repr(v)}" for k, v in self.kwargs.items()])
        )

        if self.method is None:
            f = "<callable>"
        else:
            f = "<obj>." + self.method

        description.append_text(f"{f}({call_params}) => ")
        description.append_description_of(self.matcher)

    def describe_mismatch(self, item: Callable, mismatch_description: Description) -> None:
        val = self._call_item(item)
        mismatch_description.append_text("was =>").append_description_of(val)


@dataclass
class WhenCalledBuilder:
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    method: Optional[str] = None

    def matches(self, matcher: Any) -> WhenCalledMatcher:
        return WhenCalledMatcher(
            args=self.args,
            kwargs=self.kwargs,
            matcher=_as_matcher(matcher),
            method=self.method,
        )


def when_called(*args: Any, **kwargs: Any) -> WhenCalledBuilder:
    """
    Build a matcher that asserts that a given callable, when called
    with the supplied arguments, will yield a result matching the
    expected matcher.

    Usage:

    >>> def example(a, b, c):
    >>>     return dict(a=a, b=b, c=c)
    >>>
    >>> hamcrest.assert_that(
    >>>     example,
    >>>     assertions.when_called(1, 2, c=3) \
    >>>         .matches(dict(a=1, b=2, c=3))
    >>> )

    :param args: the arguments to the function.
    :param kwargs: the keyword arguments to the function.
    :return: the builder (needs to have `.matches(<Matcher>)` called).
    """
    return WhenCalledBuilder(args, kwargs)


def calling_method(_method: str, *args: Any, **kwargs: Any) -> WhenCalledBuilder:
    """
    Build a matcher that asserts that calling a given method on an object,
    when called with the supplied arguments, will yield a result matching the
    expected matcher.

    Usage:

    >>> class Example:
    >>>   def foo(self, a, b, c):
    >>>     return dict(a=a, b=b, c=c)
    >>>
    >>> hamcrest.assert_that(
    >>>     Example(),
    >>>     assertions.calling_method('foo', 1, 2, c=3) \
    >>>         .matches(dict(a=1, b=2, c=3))
    >>> )

    :param args: the arguments to the function.
    :param kwargs: the keyword arguments to the function.
    :return: the builder (needs to have `.matches(<Matcher>)` called).
    """
    return WhenCalledBuilder(args, kwargs, method=_method)


def _as_matcher(matcher: Any) -> Matcher:
    if not isinstance(matcher, Matcher):
        matcher = hamcrest.is_(matcher)

    return matcher


def assert_match(actual: Any, matcher: Any, reason: str = "") -> None:
    """
    Asserts that the actual value matches the matcher.

    Similar to hamcrest.assert_that(), but if the matcher is not a Matcher,
    will fallback to ``hamcrest.is_(matcher)`` rather than boolean matching.

    :param actual: the value to match.
    :param matcher: a matcher, or a value that will be converted to an ``is_`` matcher.
    :param reason: Optional explanation to include in failure description.
    """
    _assert_match(
        actual=actual,
        matcher=_as_matcher(matcher),
        reason=reason,
    )


def assert_true(actual: Any, reason: str = "") -> None:
    """
    Asserts that the actual value is truthy.

    :param actual: the value to match.
    :param reason: Optional explanation to include in failure description.
    """
    _assert_bool(actual, reason=reason)


def assert_false(actual: Any, reason: str = "") -> None:
    """
    Asserts that the actual value is falsey.

    :param actual: the value to match.
    :param reason: Optional explanation to include in failure description.
    """
    _assert_bool(not actual, reason=reason)


def assert_raises(
    func: Callable[[], Any],
    exception: Type[Exception],
    pattern: Optional[str] = None,
    matching: Any = None,
) -> None:
    """
    Typing utility wrapper for asserting that a lambda throws an expected error.

    Usage:

    >>> def foo():
    >>>   raise AssertionError("Big time error: 12")
    >>>
    >>> assert_raises(
    >>>   lambda: foo,
    >>>   AssertionError,
    >>>   "error: 12",
    >>> )

    Equivalent to the longer and more annoying:

    >>> hamcrest.assert_that(
    >>>     # needed to silence mypy
    >>>     typing.cast(Any, func),
    >>>     hamcrest.raises(
    >>>         exception=exception,
    >>>         pattern=pattern,
    >>>         matching=matching,
    >>>     ),
    >>> )

    :param func: the function to call.
    :param exception: the exception class to expect.
    :param pattern: an optional regex to match against the exception message.
    :param matching: optional Matchers to match against the exception.
    """

    hamcrest.assert_that(
        typing.cast(Any, func),
        hamcrest.raises(
            exception=exception,
            pattern=pattern,
            matching=matching,
        ),
    )
