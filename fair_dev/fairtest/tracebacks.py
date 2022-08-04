import typing

_TRACEBACK_SENSE_FLAGS = [
    # unittest integration; hide these frames from tracebacks
    "__unittest",
    # py.test integration; hide these frames from tracebacks
    "__tracebackhide__",
]


def hide_module_tracebacks(module: typing.Any, mode: bool = True) -> None:
    """
    Set traceback behavior flags on a module.

    Various python testing tools will pay attention to certain sense flags
    in stack frames, and selectively remove stack frames they are present in.

    This is useful to strip the stack frames from a unittesting library out
    of the error reports; because we generally assume the library is correct,
    and the error should be attributed to the user's call site, and not
    the unittest library.

    When _developing_ a unittest library, this isn't true.

    This machinery lets us toggle those flags for a given module.

    Suggested usage in a test library:

    >>> from fair_dev.fairtest import hide_module_tracebacks
    >>>
    >>> def hide_tracebacks(mode: bool = True) -> None:
    >>>     '''
    >>>     Hint that some unittest stacks (unittest, pytest) should remove
    >>>     frames from tracebacks that include this module.
    >>>
    >>>     :param mode: optional, the traceback mode.
    >>>     '''
    >>>     # globals(), called within the module, grabs the module dict.
    >>>     hide_module_tracebacks(globals(), mode)
    >>>
    >>> # default to hiding.
    >>> hide_tracebacks(True)

    :param module: the module to toggle.
    :param mode: (Optional) the mode; defaults to True.
    """
    for flag in _TRACEBACK_SENSE_FLAGS:
        module[flag] = mode
