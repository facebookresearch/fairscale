from fair_dev.fairtest import common_assertions, tracebacks


def test_hide_tracebacks() -> None:
    # we can verify the behavior of setting the flags here,
    # but we can't easily trigger the behavior of stripping tracebacks.

    tracebacks.hide_module_tracebacks(globals())

    common_assertions.assert_true(
        globals()["__unittest"],
    )
    common_assertions.assert_true(
        globals()["__tracebackhide__"],
    )

    tracebacks.hide_module_tracebacks(globals(), False)

    common_assertions.assert_false(
        globals()["__unittest"],
    )
    common_assertions.assert_false(
        globals()["__tracebackhide__"],
    )
