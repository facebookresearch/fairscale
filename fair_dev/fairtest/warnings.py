import contextlib
from typing import Generator
import warnings


@contextlib.contextmanager
def ignore_warnings() -> Generator[None, None, None]:
    """
    This is a general context manager to suppress all python warnings.

    The use case is that, when testing an API, sometimes it is valuable
    to call that API in an incorrect way, to verify that error handling
    behaves as excepted.

    There is equally a desire to not spam unittest results with warnings,
    so sometimes we avoid writing those tests.

    This permits tests (such as `assert_raises()`) to be written which
    trigger warnings, without escalating those warnings to the test environment.

    Usage:

    >>> with ignore_warnings():
    >>>   # something that will warn ...
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
