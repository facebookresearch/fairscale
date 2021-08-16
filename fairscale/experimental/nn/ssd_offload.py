import io

import numpy as np
import torch

DEFAULT_CHUNK_SIZE = 1024 * 1024


def _get_num_chunks(t: torch.Tensor, chunk_size_bytes: int = DEFAULT_CHUNK_SIZE) -> int:
    size_in_bytes = t.nelement() * t.element_size()
    num_chunks = (size_in_bytes + (chunk_size_bytes - 1)) // chunk_size_bytes
    return num_chunks


def _tensor_to_bytes_chunks(t: torch.Tensor, chunk_idx: int, chunk_size_bytes: int = DEFAULT_CHUNK_SIZE) -> bytes:
    size_in_bytes = t.nelement() * t.element_size()
    assert chunk_idx < _get_num_chunks(t, chunk_size_bytes)
    t_np = t.detach().numpy().view(np.uint8).reshape(-1)
    chunk_start = chunk_idx * chunk_size_bytes
    chunk_end = min(size_in_bytes, chunk_start + chunk_size_bytes)
    return t_np[chunk_start:chunk_end].tobytes()


def write(t: torch.Tensor, filename: str) -> None:
    num_chunks = _get_num_chunks(t)
    with open(filename, "wb") as f:
        for i in range(num_chunks):
            f.write(_tensor_to_bytes_chunks(t, i))


def read(t: torch.Tensor, filename: str) -> None:
    size_in_bytes = t.nelement() * t.element_size()
    chunk_size_bytes = DEFAULT_CHUNK_SIZE
    num_chunks = _get_num_chunks(t)
    t_np = t.detach().numpy()
    t_mv = memoryview(t_np.view(dtype=np.uint8).reshape(-1))
    fixed_mv = t_mv[0:chunk_size_bytes]
    with io.open(filename, "rb") as f:
        for i in range(num_chunks):
            chunk_start = i * chunk_size_bytes
            chunk_end = min(size_in_bytes, chunk_start + chunk_size_bytes)
            data_read = f.readinto(t_mv[chunk_start:chunk_end])
            assert data_read == chunk_end - chunk_start
