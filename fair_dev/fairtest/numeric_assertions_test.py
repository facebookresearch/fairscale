import unittest

from fair_dev.fairtest import common_assertions, numeric_assertions


class AssertCloseToTest(unittest.TestCase):
    def test(self):
        numeric_assertions.assert_close_to(
            12.5,
            12.49999,
        )

        common_assertions.assert_raises(
            lambda: numeric_assertions.assert_close_to(
                12.5,
                12.499,
            ),
            AssertionError,
            r"within <0.000125> of <12.499>\s*but: <12.5> differed by <0.0009",
        )
