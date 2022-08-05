import contextlib
import random
from typing import Any, Iterator, Optional

import numpy as np
import torch


def set_random_seed(
    seed: int = 0,
    py_rng_state: Optional[Any] = None,
    torch_rng_state: Optional[Any] = None,
    np_rng_state: Optional[Any] = None,
) -> None:
    """
    Sets the python, torch, and numpy global random number generators.

    :param seed: the seed.
    :param py_rng_state: (optional) the python rng state.
    :param torch_rng_state: (optional) the torch rng state.
    :param np_rng_state: (optional) the numpy rng state.
    """
    if py_rng_state is not None:
        random.setstate(py_rng_state)
    else:
        random.seed(seed)

    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state)
    else:
        torch.manual_seed(seed)

    if np_rng_state is not None:
        np.random.set_state(np_rng_state)
    else:
        np.random.seed(seed)


@contextlib.contextmanager
def with_random_seed(
    seed: int = 0,
    py_rng_state: Optional[Any] = None,
    torch_rng_state: Optional[Any] = None,
    np_rng_state: Optional[Any] = None,
) -> Iterator:
    """
    Context manager which resets the python, torch, and numpy global random number generators,
    and restores them on exit.

    :param seed: the seed.
    :param py_rng_state: (optional) the python rng state.
    :param torch_rng_state: (optional) the torch rng state.
    :param np_rng_state: (optional) the numpy rng state.
    """
    old_py_state = random.getstate()
    old_torch_state = torch.get_rng_state()
    old_np_state = np.random.get_state()

    set_random_seed(
        seed=seed,
        py_rng_state=py_rng_state,
        torch_rng_state=torch_rng_state,
        np_rng_state=np_rng_state,
    )

    yield

    random.setstate(old_py_state)
    torch.set_rng_state(old_torch_state)
    np.random.set_state(old_np_state)
