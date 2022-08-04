from typing import Any, Optional, SupportsFloat

import hamcrest

from fair_dev.fairtest import assert_match, hide_module_tracebacks


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


def close_to(
    expected: Any,
    delta: Optional[SupportsFloat] = None,
    *,
    rtol: SupportsFloat = 1e-05,
    atol: SupportsFloat = 1e-08,
):
    """
    `close_to()` variant matcher with a default dynamic delta, based on `numpy.isclose()`,

    This clones the numpy notion of `np.isclose()` pattern, where the default delta is:

    >>> delta = atol + rtol * abs(expected)

    :param expected: the expected value.
    :param delta: (optional) the tolerance.
    :param rtol: if delta is None, the relative tolerance.
    :param atol: if delta is None, the absolute tolerance.
    :return: a close_to() matcher.
    """

    if delta is None:
        # numpy.isclose() pattern:
        delta = atol + rtol * abs(expected)

    delta = float(delta)

    return hamcrest.close_to(float(expected), delta)


def assert_close_to(
    actual: Any,
    expected: Any,
    delta: Optional[SupportsFloat] = None,
    *,
    rtol: SupportsFloat = 1e-05,
    atol: SupportsFloat = 1e-08,
) -> None:
    """
    Asserts that two values are close to each other, with automagic delta selection.

    This clones the numpy notion of `np.isclose()` pattern, where the default delta is:

    >>> delta = atol + rtol * abs(expected)

    :param actual: the actual value.
    :param expected: the expected value.
    :param delta: (optional) the tolerance.
    :param rtol: if delta is None, the relative tolerance.
    :param atol: if delta is None, the absolute tolerance.
    """
    assert_match(
        actual,
        close_to(
            expected=expected,
            delta=delta,
            rtol=rtol,
            atol=atol,
        ),
    )
