# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any

# Deprecate allclose when we move to newer versions.
def assert_allclose(actual: Any, expected: Any, rtol: float = ..., atol: float = ..., equal_nan: bool = ..., msg: str = ...) -> None: ...
def assert_close(actual: Any, expected: Any, rtol: float = ..., atol: float = ..., equal_nan: bool = ..., msg: str = ...) -> None: ...
