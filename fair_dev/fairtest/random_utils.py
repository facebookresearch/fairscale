import contextlib
from typing import Iterator, Union

import torch


@contextlib.contextmanager
def with_random_seed(seed_or_state: Union[int | torch.ByteTensor]) -> Iterator:
    """
    Context manager which resets the global random number generator,
    and restores it on exit.

    :param seed_or_state: the seed (an int) or .get_rng_state() (a ByteTensor)
    """
    old_state = torch.get_rng_state()

    if isinstance(seed_or_state, torch.Tensor):
        torch.set_rng_state(seed_or_state)
    else:
        torch.manual_seed(seed_or_state)

    yield

    torch.set_rng_state(old_state)
