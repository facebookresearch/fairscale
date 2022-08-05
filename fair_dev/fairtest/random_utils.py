import contextlib
import typing

import torch


@contextlib.contextmanager
def reset_generator_seed(seed: int = 3 * 17 * 53 + 1) -> typing.Iterator:
    """
    Context manager which resets the `torch.manual_seed()` seed on entry.

    :param seed: optional seed.
    """
    torch.manual_seed(seed)
    yield
