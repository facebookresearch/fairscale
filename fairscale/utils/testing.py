# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# We're not responsible for pytest decorators
# mypy: disallow_untyped_decorators = False

"""
Collection of some testing utilities for the Fairscale library. Please complement as you see fit, but refrain from ad-hoc test utils
within the different feature sets and relative imports.
"""

import functools
import inspect
import logging
import multiprocessing
import os
import random
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy
import pytest
import torch
import torch.distributed as dist
from torch.distributed import rpc
import torch.multiprocessing as mp
import torch.nn as nn

from fairscale.nn.model_parallel import destroy_model_parallel, initialize_model_parallel
from fairscale.nn.model_parallel.random import model_parallel_cuda_manual_seed

skip_if_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 1, reason="CUDA required"
)

skip_if_single_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="multiple GPUs required"
)

skip_if_py38 = pytest.mark.skipif(
    sys.version_info.major == 3 and sys.version_info.minor == 8, reason="Python3.8 is skipped"
)

_, filename_mpi = tempfile.mkstemp()


class IdentityLayer(torch.nn.Module):
    def __init__(self, size: int, scale: float = 1.0) -> None:
        super(IdentityLayer, self).__init__()
        self.weight = torch.nn.Parameter(scale * torch.randn(size))

    def forward(self, *_: Any, **__: Any) -> Any:
        return self.weight


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)


def torch_version() -> Tuple[int, ...]:
    numbering = torch.__version__.split("+")[0].split(".")[:3]

    # Catch torch version if run against internal pre-releases, like `1.8.0a0fb`,
    if not numbering[2].isnumeric():
        # Two options here:
        # - either skip this version (minor number check is not relevant)
        # - or check that our codebase is not broken by this ongoing development.

        # Assuming that we're interested in the second usecase more than the first,
        # return the pre-release or dev numbering
        logging.warning(f"Pytorch pre-relase version {torch.__version__} - assuming intent to test it")
        numbering[2] = "0"

    return tuple(int(n) for n in numbering)


def dist_init(rank: int, world_size: int, filename: str, filename_rpc: str = "") -> bool:
    """
    Initialize torch distributed, based on a temporary file shared across ranks, which makes it possible for unrelated
    tests to be run concurrently.

    Return false if not enough GPUs present in the system.

    .. warning: This limits the usecase to all ranks being on the same node
    """

    try:
        torch.distributed.rpc.shutdown()
    except Exception:
        pass

    print(f"dist init r={rank}, world={world_size}")

    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    url = "file://" + filename
    url_rpc = "file://" + filename_rpc

    if torch_version() >= (1, 6, 0):
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl" and torch.cuda.device_count() < world_size:
            logging.warning("Requested world size cannot be reached on this machine, not enough GPUs")
            return False

        torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size, init_method=url)

        rpc.init_rpc(
            f"Test{rank}",
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method=url_rpc),
        )

    else:
        if world_size > 1:
            # TensorPipe is not available in Torch 1.5
            rpc.init_rpc(
                name=f"Test{rank}",
                rank=rank,
                world_size=world_size,
                rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(init_method=url_rpc),
            )
        elif torch.cuda.is_available():
            torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method=url)
        else:
            return False

    if torch.cuda.is_available() and torch.cuda.device_count():
        torch.cuda.set_device(rank % torch.cuda.device_count())

    return True


def get_worker_map() -> Dict[Any, Any]:
    return {rank: f"Test{rank}" for rank in range(dist.get_world_size())}


def get_world_sizes() -> List[int]:
    limit = torch.cuda.device_count()
    return [x for x in [1, 2, 4, 8] if x <= limit]


def spawn_for_all_world_sizes(test_func: Callable, world_sizes: List[int] = get_world_sizes(), args: Any = []) -> None:

    for world_size in world_sizes:
        _, filename = tempfile.mkstemp()
        _, filename_rpc = tempfile.mkstemp()

        # (lefaudeux) Let mp handle the process joining, join=False and handling context has been unstable in the past
        mp.spawn(test_func, args=(world_size, filename, filename_rpc, *args), nprocs=world_size, join=True)


def worker_process(
    rank: int, world_size: int, filename: str, filename_rpc: str, func: Callable, args: Any, error_queue: Any
) -> None:
    """Main function for unit tests launced with torch_spawn"""

    if not dist_init(rank, world_size, filename, filename_rpc):
        logging.warning("failed initializing torch distributed")
        teardown()
        return

    kwargs = {}
    if "OMPI_COMM_WORLD_RANK" not in os.environ:
        kwargs["pipeline_backend"] = "gloo"

    initialize_model_parallel(1, world_size, **kwargs)

    try:
        func(*args)
        teardown()
    except BaseException as e:
        logging.warning(f" Rank {rank}: {e}")

        # Make sure that the group is properly destroyed, even for tests which check for exceptions being raised
        teardown()

        # If the function raises 'Skipped', this indicates pytest.skip(), so
        # forward it to parent so we can call pytest.skip() there
        if e.__class__.__name__ == "Skipped":
            error_queue.put(str(e))
            return

        raise e


def teardown() -> None:
    destroy_model_parallel()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    try:
        # torch 1.5 hangs on shutdown if waiting for all processes
        torch.distributed.rpc.shutdown(graceful=False)
    except Exception:
        pass


def torch_spawn(world_sizes: Optional[List[int]] = None) -> Callable:
    if world_sizes is None:
        world_sizes = get_world_sizes()

    def prepare_test(func: Callable) -> Callable:
        """Function called with the test function as the argument. Generates a
        replacement which serves as the actual test function."""

        name = func.__name__
        parameters = inspect.signature(func).parameters

        if name.startswith("test"):
            raise ValueError(
                f"Tests marked with @torch_spawn (i.e. '{name}') should not have names beginning in 'test' as they will"
                " be picked up by pytest without running the spawn wrapper"
            )

        @functools.wraps(func)
        def replacement(*args: Any, **kwargs: Any) -> None:
            assert args == tuple()
            assert world_sizes is not None  # mypy crutch

            args = tuple(
                kwargs[p] for p in parameters if p != "rank"
            )  # converting named parameters to positional parameters to pass to `spawn`

            error_queue = multiprocessing.get_context("spawn").SimpleQueue()
            if "OMPI_COMM_WORLD_RANK" in os.environ:
                global filename_mpi

                os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
                os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
                torch.distributed.init_process_group("mpi", init_method=f"file://{filename_mpi}")

                world_size = torch.distributed.get_world_size()
                destroy_model_parallel()
                initialize_model_parallel(1, world_size)
                torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())
                if world_size in world_sizes:
                    try:
                        func(*args)
                        teardown()
                    except BaseException as e:
                        teardown()
                        import traceback

                        print(f"{traceback.format_exc()}")
                        raise e
                else:
                    pytest.skip("Requested world size doesn't match current world size")
            else:
                spawn_for_all_world_sizes(worker_process, world_sizes, (func, args, error_queue))

            if not error_queue.empty():
                msg = error_queue.get()
                pytest.skip(msg)

        # Register a function with the same name, prefixed with "test_" in the
        # calling module, so it will be picked up by pytest
        current_frame = inspect.currentframe()
        assert current_frame is not None
        caller_module = inspect.getmodule(current_frame.f_back)
        setattr(caller_module, f"test_{name}", replacement)

        return func

    return prepare_test


class _Block(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)  # type: ignore
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim),)

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        x = inputs[0]
        attn_mask = torch.full((len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)

        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class GPT2(nn.Module):
    """
    GPT2 pytorch implementation, for testing purposes in the image-GPT context
    Credits: https://github.com/teddykoker/image-gpt"""

    def __init__(
        self, embed_dim: int, num_heads: int, num_layers: int, num_positions: int, num_vocab: int, num_classes: int
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        self.token_embeddings = nn.Embedding(num_vocab, embed_dim)
        self.position_embeddings = nn.Embedding(num_positions, embed_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(_Block(embed_dim, num_heads))

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vocab, bias=False)
        self.clf_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, classify=False) -> Any:  # type: ignore
        """
        Expect input as shape [sequence len, batch]
        If classify, return classification logits
        """
        length, batch = x.shape

        h = self.token_embeddings(x)

        # prepend sos token
        sos = torch.ones(1, batch, self.embed_dim, device=x.device) * self.sos
        h = torch.cat([sos, h[:-1, :, :]], dim=0)

        # add positional embeddings
        positions = torch.arange(length, device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)

        # transformer
        for layer in self.layers:
            h = layer(h)

        h = self.ln_f(h)

        logits = self.head(h)

        if not classify:
            # return logits
            return logits

        h = torch.mean(h, dim=0)  # average pool over sequence
        # return classification logits and generative logits
        return self.clf_head(h), logits


def objects_are_equal(a: Any, b: Any, raise_exception: bool = False) -> bool:
    """
    Test that two objects are equal. Tensors are compared to ensure matching
    size, dtype, device and values.
    """
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        for k in a.keys():
            if not objects_are_equal(a[k], b[k], raise_exception):
                return False
        return True
    elif isinstance(a, (list, tuple, set)):
        if len(a) != len(b):
            return False
        return all(objects_are_equal(x, y, raise_exception) for x, y in zip(a, b))
    elif torch.is_tensor(a):
        try:
            torch.testing.assert_allclose(a, b)
            # assert_allclose doesn't strictly test shape, dtype and device
            shape_dtype_device_match = a.size() == b.size() and a.dtype == b.dtype and a.device == b.device
            assert shape_dtype_device_match
            return True
        except AssertionError as e:
            if raise_exception:
                raise e
            else:
                return False
    else:
        return a == b
