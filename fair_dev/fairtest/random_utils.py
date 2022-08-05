import contextlib
from dataclasses import dataclass
import random
from typing import Any, Iterator, Optional

import numpy as np
import torch


@dataclass
class CommonRngState:
    py_rng_state: Optional[Any] = None
    torch_rng_state: Optional[Any] = None
    np_rng_state: Optional[Any] = None

    @classmethod
    def get_state(cls) -> "CommonRngState":
        return CommonRngState(
            py_rng_state=random.getstate(),
            torch_rng_state=torch.get_rng_state(),
            np_rng_state=np.random.get_state(),
        )

    def set_state(self) -> None:
        """
        Apply the state to the random, torch, and numpy global generators.
        """
        if self.py_rng_state is not None:
            random.setstate(self.py_rng_state)
        if self.torch_rng_state is not None:
            torch.set_rng_state(self.torch_rng_state)
        if self.np_rng_state is not None:
            np.random.set_state(self.np_rng_state)


def set_random_seed(
    seed: int = 0,
    *,
    state: Optional[CommonRngState] = None,
) -> None:
    """
    Sets the python, torch, and numpy global random number generators.

    :param seed: the seed.
    :param state: (optional) a CommonRngState.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if state is not None:
        state.set_state()


@contextlib.contextmanager
def with_random_seed(
    seed: int = 0,
    *,
    state: Optional[CommonRngState] = None,
) -> Iterator:
    """
    Context manager which resets the python, torch, and numpy global random number generators,
    and restores them on exit.

    :param seed: the seed.
    :param state: (optional) a CommonRngState.
    """
    old_state = CommonRngState.get_state()

    set_random_seed(
        seed=seed,
        state=state,
    )

    yield

    old_state.set_state()
