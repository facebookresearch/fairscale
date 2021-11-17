# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Distributed Gossip Wrapper

:description: Multi-Threaded Gossip Model Wrapper; designed for efficient
              multi-peer training.
"""

from enum import Enum
import functools
import logging
import os
import sys
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast

import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.nn.modules import Module

from .gossiper import Gossiper, PushPull, PushSum
from .graph_manager import GraphManager
from .graph_manager import NPeerDynamicDirectedExponentialGraph as NPDDEGraph
from .mixing_manager import MixingManager, UniformMixing
from .utils import (
    MultiProcessAdapter,
    communicate,
    create_process_group,
    flatten_tensors,
    group_by_dtype,
    make_logger,
    unflatten_tensors,
)
from .utils.cuda_metering import EventRecorder, create_event_recorder

HEARTBEAT_TIMEOUT = 300  # maximum time to wait for message (seconds)
BROADCAST_BUCKET_SIZE = 10 * 1024 * 1024


class SlowMoBaseAlgorithm(str, Enum):
    LOCALSGD = "localsgd"
    SGP = "sgp"


class SlowMoDistributedDataParallel(Module):
    """Wraps an arbitrary :class:`nn.Module <torch.nn.Module>` module and allows
    it to be run on multiple GPUs (distributed) in a data parallel setting.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. The module is replicated on each machine and each device, and
    each such replica handles a portion of the input. After the optimizer update,
    it synchronizes the parameters on the different nodes using SlowMo
    (https://arxiv.org/abs/1910.00643).

    Please make sure to read the documentation for slowmo_memory_efficient parameter as
    it contains a non-trivial trick in order to optimize our implementation.

    Please refer to the documentation of ``torch.nn.parallel.DistributedDataParallel``
    for other useful tips for using this container.

    Parameters:
        module (Module):
            module to be parallelized
        nprocs_per_node (int):
            Number of processes per node (one per GPU). This needs to be specified for optimal accuracy and speed.
            Syncing across GPUs in a node is extremely fast, which we utilize for performance optimization
        broadcast_buffers (bool):
            Flag that enables syncing (broadcasting) buffers (example - batchnorm buffers) of the module at beginning
            of the ``forward`` function. Setting it to False would result in better performance due to less
            communication on the network but might result in a reduced accuracy (default: ``True``)
        slowmo_base_algorithm (SlowMoBaseAlgorithm):
            The base algorithm to be used for approximately averaging the different parameters across nodes.  The base
            algorithm is responsible for increasing the efficiency of this module. The base algorithm, combined with
            SlowMo, results in significant speedups without accuracy loss. Either Stochastic Gradient Push
            (SlowMoBaseAlgorithm.SGP) (https://arxiv.org/abs/1811.10792) or LocalSGD (SlowMoBaseAlgorithm.LOCALSGD)
            (https://arxiv.org/abs/1808.07217) can be used here (default: SlowMoBaseAlgorithm.LOCALSGD)
    SlowMo Parameters:
        slowmo_momentum (float):
            This specifies the value of slowmo momentum to be used (read https://arxiv.org/abs/1910.00643 for more
            details). This parameter might need to be tuned and the optimal value varies according to the use case and
            the number of nodes being run on. The optimal value typically increases with the number of nodes. On
            training transfomers on the WMT 16 En-De dataset, we have found the optimal values to be 0 for less than 4
            nodes, 0.2 for 4 nodes, 0.5 for 8 nodes and 0.6 for 16 nodes (default: 0.5)
        slowmo_memory_efficient (bool):
            If enabled, use a memory efficient implementation of SlowMo. The basic implementation of SlowMo occupies
            extra memory equal to double the memory occupied by the model parameters. The memory efficient
            implementation shards that memory across a certain number of shards which is specified as a parameter
            below.
            In addition, slowmo_memory_efficient leads to extra communication with throughput equivalent to an
            allreduce, and performs an allreduce as a side-effect. In order to optimize the implementation, we skip
            the typical allreduce when slowmo_base_algorithm is localsgd and the localsgd step and slowmo step occur
            on the same iteration. Also, we skip the gossip step when slowmo_base_algorithm is sgp. We can skip these
            because the memory-efficient slowmo step does an allreduce as a side effect. Due to this skipping, when
            slowmo_base_algorithm is localsgd, we recommend setting slowmo_frequency to be a multiple of
            localsgd_frequency.
            We recommend setting this parameter to True when slowmo_base_algorithm is localsgd. In case of sgp, there
            is a tradeoff between extra memory usage which is double the memory occupied by the parameters, and extra
            time spent which is half the time taken up by an allreduce every slowmo_frequency iterations and we
            suggest setting it to False (default: True)
        slowmo_frequency (int):
            This specifies how often (number of iterations) slow momentum is to be performed. We recommend keeping
            slowmo_frequency as a multiple of localsgd_frequency. Please look at the documentation of
            slowmo_memory_efficient for the reasoning (default: 48)
        slowmo_lr (float):
            This specifies the value of slowmo learning rate to be used (read https://arxiv.org/abs/1910.00643 for
            more details). We do not recommend changing this (default: 1.0)
        slowmo_num_shards (int):
            The number of shards between which slow momentum parameters are distributed. This is only used when
            memory_efficient is set to True.
            The number of shards should scale with the number of parameters in the model. Increasing the number of
            shards decreases the memory used per node for storing the slow momentum parameters. However, if the shard
            size per node is too small, it results in a communication overhead (default: 32)
    LocalSGD Parameters:
        localsgd_frequency (int):
            LocalSGD typically averages the parameters once every few iterations. This parameter specifices the
            frequency of averaging.  We recommend keeping slowmo_frequency as a multiple of localsgd_frequency. Please
            look at the documentation of slowmo_memory_efficient for the reasoning (default: 3)
    SGP Parameters:
        graph (Optional[GraphManager):
            Graph to be used for gossip communication. This is used to specify the interaction graph between the
            different nodes (default: None)
        mixing (Optional[MixingManager]):
            Mixing manager to be used for gossip communication. This is used to specify weights given to outgoing and
            incoming messages (default: None)
        push_sum (bool):
            Whether to use PushSum or PushPull gossip (default: True)
        overlap (bool):
            Whether to use the overlap form of SGP. This feature is currently disabled until further testing is done
            for its use (default: False)
        synch_freq (int):
            How often (number of iterations) to synchronize for overlap SGP. A value of 0 means to synchronize overlap
            SGP every iteration (default: 0)
        use_streams (bool):
            Whether to use CUDA streams to speed up SGP overlap (default: True)
        slowmo_sgp_average_params (bool):
            Whether to completely average the parameters when slowmo is done instead of a partial averaging that
            happens every iteration (default: False)
    Debugging Parameters:
        verbose (bool):
            Prints various logs which are useful for debugging (default: False)
        profile_mode (bool):
            Prints the time taken by different parts of the code, which can help in finding bottlenecks (default: False)
    Parameters for Advanced Users:
        process_rank (Optional[int]):
            Rank of the current process in the process group (default: None)
        process_world_size (Optional[int]):
            Size of the process group (default: None)
        global_group (Optional[torch.distributed.ProcessGroup]):
            Global process group initialized by init_process_group (default: None)
        master_group (Optional[torch.distributed.ProcessGroup]):
            Process group which only contains the master GPUs of each node (default: None)
        local_node_group (Optional[torch.distributed.ProcessGroup]):
            Process group which only contains the GPUs local to the current node (default: None)
        comm_device: (Optional[torch.device]):
            The torch.device on which torch tensors are to be placed before communication (default: None)

    Example:
        >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
        >>> net = fairscale.data_parallel.SlowMoDistributedDataParallel(model, nprocs_per_node=8)
        >>> loss = criterion(net(inputs), targets)
        >>> loss.backward()
        >>> optimizer.step()
        >>> net.perform_slowmo(optimizer)
    """

    def __init__(
        self,
        module: torch.nn.Module,
        nprocs_per_node: int,
        broadcast_buffers: bool = True,
        slowmo_base_algorithm: SlowMoBaseAlgorithm = SlowMoBaseAlgorithm.LOCALSGD,
        # SlowMo Args
        slowmo_momentum: float = 0.5,
        slowmo_memory_efficient: bool = True,
        slowmo_frequency: int = 48,
        slowmo_lr: float = 1.0,
        slowmo_num_shards: int = 32,
        # LocalSGD Args
        localsgd_frequency: int = 3,
        # SGP Args
        graph: Optional[GraphManager] = None,
        mixing: Optional[MixingManager] = None,
        push_sum: bool = True,
        overlap: bool = False,
        synch_freq: int = 0,
        use_streams: bool = True,
        slowmo_sgp_average_params: bool = False,
        # Debugging Args
        verbose: bool = False,
        profile_mode: bool = False,
        # Args for advanced users (these are automatically handled otherwise)
        process_rank: Optional[int] = None,
        process_world_size: Optional[int] = None,
        global_group: Optional[torch.distributed.ProcessGroup] = None,
        master_group: Optional[torch.distributed.ProcessGroup] = None,
        local_node_group: Optional[torch.distributed.ProcessGroup] = None,
        comm_device: Optional[torch.device] = None,
    ) -> None:
        super(SlowMoDistributedDataParallel, self).__init__()

        # NCCL_BLOCKING_WAIT causes issues with using multiple process groups
        assert os.environ.get("NCCL_BLOCKING_WAIT", "0") == "0"

        assert nprocs_per_node >= 1
        self.nprocs_per_node = nprocs_per_node

        if process_world_size is None or process_rank is None:
            assert dist.is_initialized()
            process_rank = dist.get_rank()
            process_world_size = dist.get_world_size()
        assert process_world_size is not None and process_rank is not None
        self.process_rank = process_rank
        self.process_world_size = process_world_size

        self._initialize_logger(verbose, self.process_rank)

        # The logical prefix in the following variables denotes the variable value if nprocs_per_node processes
        # were treated as one process and then the following variables were calculated for the resulting process
        # group. This is how they are being treated for optimization purposes because intra-node communication is
        # very efficient with NVLink.
        logical_rank, logical_world_size = self._maybe_create_process_groups(
            self.process_rank, self.process_world_size, nprocs_per_node, global_group, master_group, local_node_group
        )
        self.logical_rank = logical_rank
        self.logical_world_size = logical_world_size

        self.module = module
        self.broadcast_buffers = broadcast_buffers
        first_param_dtype = next(self.module.parameters()).dtype

        # prepare local intra-node all-reduce objects
        self.broadcast_bucket_size = BROADCAST_BUCKET_SIZE  # bytes
        self.module_buffers = list(self.module.buffers())

        # choose communication device based on backend
        if comm_device is None:
            cpu_comm = dist.get_backend() == "gloo"
            comm_device = torch.device("cpu") if cpu_comm else torch.device("cuda")
        self._cpu_comm = comm_device.type == "cpu"

        # distributed backend config
        self.dist_config = {
            "verbose": verbose,
            "comm_device": comm_device,
            "logical_rank": logical_rank,
            "process_rank": self.process_rank,
            "logical_world_size": logical_world_size,
            "cpu_comm": self._cpu_comm,
        }
        self.profile_mode = profile_mode
        self.num_updates = 0
        self.portion_start: Optional[int] = None

        # slowmo being set to False is equivalent to slowmo_lr being set to 1 and slowmo_momentum being set to 0
        # This condition is ensuring the values are safe to use even when slowmo is disabled
        self.slowmo = slowmo_lr != 1 or slowmo_momentum != 0

        self.slowmo_lr = slowmo_lr if self.slowmo else 1
        self.slowmo_momentum = slowmo_momentum if self.slowmo else 0

        self.slowmo_frequency = slowmo_frequency
        self.slowmo_sgp_average_params = slowmo_sgp_average_params

        self.localsgd = slowmo_base_algorithm == SlowMoBaseAlgorithm.LOCALSGD
        self.sgp = slowmo_base_algorithm == SlowMoBaseAlgorithm.SGP

        self.localsgd_frequency = localsgd_frequency
        self.ef1: Optional[List[torch.Tensor]] = None
        self.global_momentum_buffers_initialized = False

        if self.master_group is None:
            assert self.localsgd or self.sgp
            self.localsgd = self.sgp = False
            self.logger.warning("Disabling LocalSGD and SGP since a local allreduce will suffice")

        if self.slowmo and not self.localsgd and not self.sgp:
            self.logger.warning("SlowMo is being used without LocalSGD and SGP")

        self.slowmo_memory_efficient = slowmo_memory_efficient
        self.slowmo_num_shards = min(self.process_world_size, slowmo_num_shards) if self.slowmo_memory_efficient else 1
        self.is_current_node_a_slowmo_shard = (
            self.process_rank < self.slowmo_num_shards if self.slowmo_memory_efficient else True
        )

        self.nprocs_per_node_device = torch.tensor([self.nprocs_per_node], device=comm_device, dtype=first_param_dtype)

        if self.sgp:
            self._sgp_init(
                module=module,
                first_param_dtype=first_param_dtype,
                logical_rank=logical_rank,
                logical_world_size=logical_world_size,
                comm_device=comm_device,
                graph=graph,
                mixing=mixing,
                push_sum=push_sum,
                overlap=overlap,
                synch_freq=synch_freq,
                use_streams=use_streams,
                slowmo_sgp_average_params=slowmo_sgp_average_params,
            )

        # register ps/grad-reduction hooks
        self._register_hooks()

        self.logger.debug("Initialization of SlowMoDistributedDataParallel complete")

    def _initialize_logger(self, verbose: bool, process_rank: int) -> None:
        """Initializes the logger"""
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)

        # Only create an adapter if debug logging is enabled to avoid additional overhead
        if self.logger.isEnabledFor(logging.DEBUG):
            # Set custom adapter on top of logger
            self.logger = cast(logging.Logger, MultiProcessAdapter(self.logger, {"process_num": process_rank}))

    def _maybe_create_process_groups(
        self,
        process_rank: int,
        process_world_size: int,
        nprocs_per_node: int,
        global_group: Optional[torch.distributed.ProcessGroup],
        master_group: Optional[torch.distributed.ProcessGroup],
        local_node_group: Optional[torch.distributed.ProcessGroup],
    ) -> Tuple[int, int]:
        """Creates the process groups required for the SlowMo implementation"""

        self.local_rank = process_rank % self.nprocs_per_node
        assert (
            process_world_size % self.nprocs_per_node == 0
        )  # total world size must be a multiple of `nprocs_per_node`
        logical_world_size = process_world_size // self.nprocs_per_node
        logical_rank = process_rank // self.nprocs_per_node

        self._maybe_initialize_global_group(global_group, process_world_size)
        self._maybe_initialize_local_node_group(local_node_group, process_rank, logical_world_size)
        self._maybe_initialize_master_group(master_group, process_rank, process_world_size, nprocs_per_node)

        self.logger.debug("Initialization of all process groups complete")
        return logical_rank, logical_world_size

    def _maybe_initialize_global_group(
        self, global_group: Optional[torch.distributed.ProcessGroup], process_world_size: int
    ) -> None:
        if global_group is None:
            all_processes = list(range(process_world_size))
            self.global_group = create_process_group(all_processes)
            self.logger.debug("Initialization of global group complete")
        else:
            self.global_group = global_group
        self.logger.debug("Global group set")
        self.process_group = self.global_group

    def _maybe_initialize_master_group(
        self,
        master_group: Optional[torch.distributed.ProcessGroup],
        process_rank: int,
        process_world_size: int,
        nprocs_per_node: int,
    ) -> None:
        if master_group is not None:
            self.master_group: Optional[torch.distributed.ProcessGroup] = master_group
            return

        if self.nprocs_per_node > 1:
            self.logger.debug("Initializing master process group")
            master_nodes = [i for i in range(process_world_size) if i % nprocs_per_node == 0]
            self.master_group = create_process_group(master_nodes) if len(master_nodes) > 1 else None
            if self.master_group is not None and process_rank in master_nodes:
                self.logger.debug("Initialization of master group complete")
        else:
            self.master_group = self.global_group

    def _maybe_initialize_local_node_group(
        self, local_node_group: Optional[torch.distributed.ProcessGroup], process_rank: int, logical_world_size: int
    ) -> None:
        if self.nprocs_per_node == 1:
            self.local_node_group = None
            return

        if local_node_group is not None:
            self.local_node_group = local_node_group
            return

        self.logger.debug("Initializing local process groups")
        for node in range(logical_world_size):
            node_processes_ranks = list(
                range(
                    node * self.nprocs_per_node,
                    (node + 1) * self.nprocs_per_node,
                )
            )
            # Process group to communicate between processes on this machine
            new_local_group = create_process_group(node_processes_ranks)
            if process_rank in node_processes_ranks:
                self.local_node_group = new_local_group
        assert self.local_node_group is not None
        self.logger.debug("Initialization of local groups complete")

    def forward(self, *inputs: Any, **kwargs: Any) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward pass performed in parallel across all devices on node"""
        return self.module(*inputs, **kwargs)

    def _sync_params(self) -> None:
        """Synchronize parameters across devices (intra-node)"""
        if self.local_node_group is None:
            return

        # intra-node parameter sync
        params = cast(List[torch.Tensor], list(self.module.parameters()))
        communication_op = functools.partial(
            dist.broadcast,
            src=self.logical_rank * self.nprocs_per_node,
            group=self.local_node_group,
        )
        communicate(params, communication_op)
        self.logger.debug("Intra-node param sync complete")

    def _sync_buffers(self) -> None:
        """Synchronize buffers across nodes"""
        # module buffer sync
        if self.broadcast_buffers and len(self.module_buffers) > 0:
            # Synchronize buffers across processes.
            # The process with rank 0 is considered the authoritative copy.
            self._distributed_broadcast_coalesced(self.process_group, self.module_buffers, self.broadcast_bucket_size)
        self.logger.debug("Intra-node buffer sync complete")

    def _distributed_broadcast_coalesced(
        self, process_group: torch.distributed.ProcessGroup, tensors: List[torch.Tensor], buffer_size: int
    ) -> None:
        dist._broadcast_coalesced(process_group, tensors, buffer_size)

    def _create_event_recorder(self, event_name: str) -> EventRecorder:
        """Creates an cuda event recorder which helps in profiling"""
        return create_event_recorder(event_name, dummy=not self.profile_mode)

    def _fp16_fp32_iterator(
        self, optimizer: torch.optim.Optimizer, fp32_params: Optional[torch.Tensor]
    ) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterator for those fp16 parameters which have a fp32 copy"""
        # Handle apex fp16 optimizer
        if hasattr(optimizer, "_amp_stash") and hasattr(optimizer._amp_stash, "fp16_groups"):
            for p_fp16_group, p_fp32_group in zip(
                optimizer._amp_stash.fp16_groups,
                optimizer._amp_stash.fp32_from_fp16_groups,
            ):
                for p_fp16, p_fp32 in zip(p_fp16_group, p_fp32_group):
                    yield p_fp16, p_fp32

        # Handle fairseq fp16 optimizer
        elif fp32_params is not None:
            if isinstance(fp32_params, dict):
                fp32_params_list = list(fp32_params.values())
                assert len(fp32_params_list) == 1
                fp32_params = fp32_params_list[0]

            if isinstance(fp32_params, list):
                for p, fp32_param in zip(self.parameters(), fp32_params):
                    yield p.view(-1), fp32_param
            else:
                offset = 0
                for p in self.parameters():
                    yield p.view(-1), fp32_params[offset : offset + p.numel()]
                    offset += p.numel()

    def _should_perform_slowmo(self) -> bool:
        return self.slowmo and (self.num_updates + 1) % self.slowmo_frequency == 0

    def _should_perform_localsgd(self) -> bool:
        return self.localsgd and (self.num_updates + 1) % self.localsgd_frequency == 0

    def _skip_averaging_memory_efficient_slowmo(self) -> bool:
        return self.slowmo_memory_efficient and self._should_perform_slowmo()

    def _should_perform_sgp_common(self) -> bool:
        return self.sgp and not self.overlap and not self._skip_averaging_memory_efficient_slowmo()

    def _should_perform_sgp(self) -> bool:
        return self._should_perform_sgp_common() and not self.overlap

    def _should_perform_sgp_overlap(self) -> bool:
        return self._should_perform_sgp_common() and self.overlap

    def _should_use_error_feedback(self, fp16_fp32_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> bool:
        return bool(fp16_fp32_list) and (self._should_perform_sgp() or self._should_allreduce_params())

    def _should_allreduce_params(self) -> bool:
        # We do not all-reduce parameters with local SGD if a slow momentum step is
        # performed, since this step contains a reduce operation already. Note that this
        # also means there is no error feedback correction in that case: it is not needed
        # since communication within the slow momentum step happens in fp32.
        return (self.sgp and self._should_perform_slowmo() and self.slowmo_sgp_average_params) or (
            self._should_perform_localsgd() and not self._skip_averaging_memory_efficient_slowmo()
        )

    def _maybe_pre_communicate_error_feedback(self, fp16_fp32_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        ef_rec = self._create_event_recorder("Error feedback")
        if self._should_use_error_feedback(fp16_fp32_list):
            with torch.no_grad():
                for p_fp16, p_fp32 in fp16_fp32_list:
                    if self._should_allreduce_params():
                        # This division and multiplication with the same number is done
                        # to ensure that we do not lose bits of information when we divide
                        # before the all_reduce. In order to preserve these bits in an
                        # error feedback (https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1050.5040&rep=rep1&type=pdf)
                        # like manner, we are forcing the bits to be lost
                        # initially, and storing the lost information in error feedback
                        p_fp16.div_(self.logical_world_size)
                        p_fp16.mul_(self.logical_world_size)
                    p_fp32 -= p_fp16.float()

                if self.ef1 is not None:
                    for idx, (_, p_fp32) in enumerate(fp16_fp32_list):
                        p_fp32 += self.ef1[idx]
                        p_fp32.div_(2)
        ef_rec.stop()
        self.logger.debug("Error feedback completed")

    def _maybe_post_communicate_error_feedback(self, fp16_fp32_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        ef_unroll_rec = self._create_event_recorder("Sync and error feedback unroll rec")
        if self._should_use_error_feedback(fp16_fp32_list):
            # Error Feedback Reversal
            with torch.no_grad():
                for p, p_fp32 in fp16_fp32_list:
                    p_fp32 += p.float()
        ef_unroll_rec.stop()
        self.logger.debug("Error feedback unroll completed")

    def _maybe_perform_sgp(self) -> None:
        sgp_rec = self._create_event_recorder("SGP")
        if self._should_perform_sgp():
            if not self._should_allreduce_params():
                self._sgp_transfer_params()
                self._sgp_query_gossip_queue()
                torch.cuda.synchronize()
            self.logger.debug("SGP completed")
        sgp_rec.stop()

    def _maybe_allreduce(self) -> None:
        localsgd_rec = self._create_event_recorder("Localsgd communication time")
        if self._should_allreduce_params():
            communication_op = functools.partial(dist.all_reduce, group=self.master_group)
            params = cast(List[torch.Tensor], list(self.parameters()))
            with torch.no_grad():
                for p in params:
                    p.div_(self.logical_world_size)
            self.logger.debug("Params normalized before localsgd step")

            # Commenting this out as it may cause an overhead. Can be uncommented if needed
            # synch_rec = self._create_event_recorder("Synchronization time for localsgd")
            # dist.barrier()
            # synch_rec.stop()
            # self.logger.debug("Barrier completed before localsgd step")

            communicate(params, communication_op, self.logger)
            torch.cuda.synchronize()
            self.logger.debug("Allreduce completed")
        localsgd_rec.stop()

    def _maybe_sync_locally(self) -> None:
        if self._should_perform_sgp() or self._should_allreduce_params():
            self._sync_params()
            torch.cuda.synchronize()

    def _maybe_perform_slowmo(self, optimizer: torch.optim.Optimizer) -> None:
        slowmo_rec = self._create_event_recorder("Slowmo")
        if self._should_perform_slowmo():
            self._global_momentum_step(optimizer)
        slowmo_rec.stop()
        self.logger.debug("Global momentum step completed")

    def _maybe_copy_back_fp32_parameters(self, fp16_fp32_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        ef_copy_rec = self._create_event_recorder("Error feedback copy back")
        if (
            self._should_perform_sgp() or self._should_allreduce_params() or self._should_perform_slowmo()
        ) and fp16_fp32_list:
            with torch.no_grad():
                for idx, (p_fp16, p_fp32) in enumerate(fp16_fp32_list):
                    p_fp16.copy_(p_fp32)
        ef_copy_rec.stop()
        self.logger.debug("Error feedback copy-back completed")

    def _maybe_sgp_overlap_pre_communicate_error_feedback(
        self, fp16_fp32_list: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        if self._should_perform_sgp_overlap() and fp16_fp32_list:
            # Initialize error feedback for SGP-overlap
            if self.ef1 is None:
                self.ef1 = [p_fp32.clone().detach_() for _, p_fp32 in fp16_fp32_list]

            with torch.no_grad():
                assert self.ef1 is not None
                for ef1, (p_fp16, p_fp32) in zip(self.ef1, fp16_fp32_list):
                    ef1.copy_(p_fp32 - p_fp16.float())

    def perform_slowmo(self, optimizer: torch.optim.Optimizer, fp32_params: Optional[torch.Tensor] = None) -> None:
        """This is to be called after optimizer.step(). It performs the approximate averaging using
        the base algorithm (SGP/ LocalSGD) and the slow momentum step. Since LocalSGD and the slow
        momentum step are not performed every iteration, it only performs those when needed.

        It is recommended to call ``model.zero_grad(set_to_none=True)`` just before calling this function. This
        is because ``model.zero_grad(set_to_none=True)`` frees up the memory occupied by the gradients, some of which
        may be reused by this function.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer being used for training the model
            fp32_params (Optional[torch.Tensor]): To be used when performing fp16 training. Needs to be
                        set to the fp16 copy of the parameters (default: None)
        """
        # Done here in case the global momentum buffers have not been initialized by the caller.
        # In an ideal implementation, this would be called by the caller. We do it here instead of
        # waiting for it to happen in the global_momentum step function so that we store a copy of
        # the version of the parameters at iteration 0 and can use them for a slow momentum step later.
        if not self.global_momentum_buffers_initialized:
            self._init_global_momentum_buffers(optimizer)

        fp16_fp32_list = list(self._fp16_fp32_iterator(optimizer, fp32_params))
        self.logger.debug("Created a list of fp16 and fp32 corresponding parameters")

        self.logger.debug(
            "Booleans set. Values - self._should_perform_slowmo()=%r, self._should_perform_localsgd()=%r, self._should_allreduce_params()=%r",
            self._should_perform_slowmo(),
            self._should_perform_localsgd(),
            self._should_allreduce_params(),
        )
        self.logger.debug("Step number(0-indexed)=%d", self.num_updates)

        if (
            self.num_updates == 0
            and fp32_params is None
            and not hasattr(optimizer, "_amp_stash")
            and any(p.dtype == torch.float16 for p in self.parameters())
        ):
            self.logger.warning("WARNING: please set fp32_params in perform_slowmo() in order to avoid accuracy loss")

        self._maybe_pre_communicate_error_feedback(fp16_fp32_list)
        self._maybe_perform_sgp()
        self._maybe_allreduce()
        self._maybe_sync_locally()
        self._maybe_post_communicate_error_feedback(fp16_fp32_list)
        self._maybe_perform_slowmo(optimizer)
        self._maybe_copy_back_fp32_parameters(fp16_fp32_list)
        self._maybe_sgp_overlap_pre_communicate_error_feedback(fp16_fp32_list)

        self.num_updates += 1

    def _init_global_momentum_buffers(self, optimizer: torch.optim.Optimizer) -> None:
        """Initializes the slow momentum buffers"""
        self.global_momentum_buffers_initialized = True

        if not self.slowmo:
            return

        total_elements = 0
        params_dtype = None
        for group in optimizer.param_groups:
            for p in group["params"]:
                total_elements += p.numel()

                # Assert that all parameters have the same device and dtype
                if params_dtype is None:
                    params_dtype, params_device = p.dtype, p.device
                # Check that dtype is fp32 since slow mometum is to be performed in fp32
                assert p.dtype == params_dtype == torch.float32
                assert p.device == params_device

        self.world_portion_length = (total_elements + self.slowmo_num_shards - 1) // self.slowmo_num_shards

        if not self.is_current_node_a_slowmo_shard:
            return

        self.portion_start = self.process_rank * self.world_portion_length if self.slowmo_memory_efficient else 0
        self.portion_end = (
            min((self.process_rank + 1) * self.world_portion_length, total_elements)
            if self.slowmo_memory_efficient
            else total_elements
        )

        self.old_params = torch.empty(self.world_portion_length, dtype=params_dtype).to(params_device).detach()

        # copy params to old_params to initialize old_params
        offset = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                numel = p.numel()

                if offset + numel > self.portion_start and offset < self.portion_end:

                    # start and end for each
                    overall_start = max(self.portion_start, offset)
                    overall_end = min(self.portion_end, offset + numel)

                    p_start = overall_start - offset
                    p_end = overall_end - offset

                    buffer_start = overall_start - self.portion_start
                    buffer_end = overall_end - self.portion_start

                    # let's see size of p and split based on that
                    current_p = p.view(-1)[p_start:p_end]
                    current_p_old = self.old_params[buffer_start:buffer_end]

                    current_p_old.copy_(current_p)

                offset += numel

        self.global_momentum_buffer = torch.zeros_like(self.old_params).detach()

    def _distributed_comm(self, optimizer: torch.optim.Optimizer, mode: str) -> None:
        """Performs the communication needed for the efficient SlowMo implementation"""
        offset = 0
        slowmo_comm_lists: List[List[torch.Tensor]] = [[] for _ in range(self.slowmo_num_shards)]
        with torch.no_grad():
            for group in optimizer.param_groups:
                # aggregate different parts of p in required node
                for p in group["params"]:
                    numel = p.numel()

                    # gather has a reduce operation so division by world size is needed
                    if mode == "gather":
                        p /= self.process_world_size

                    current_start = offset
                    while current_start < offset + numel:
                        main_node = current_start // self.world_portion_length

                        main_node_end = (main_node + 1) * self.world_portion_length
                        current_end = min(offset + numel, main_node_end)

                        p_start = current_start - offset
                        p_end = current_end - offset

                        slowmo_comm_lists[main_node].append(p.view(-1)[p_start:p_end])

                        current_start = current_end
                    offset += numel

            for slowmo_rank, slowmo_comm_list in enumerate(slowmo_comm_lists):
                if mode == "gather":
                    communication_op = functools.partial(dist.reduce, dst=slowmo_rank)
                elif mode == "scatter":
                    communication_op = functools.partial(dist.broadcast, src=slowmo_rank)
                communicate(slowmo_comm_list, communication_op)

    def _global_momentum_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Performs the slow momentum step"""
        if not self.slowmo:
            return

        if not self.global_momentum_buffers_initialized:
            self._init_global_momentum_buffers(optimizer)

        if self.slowmo_memory_efficient:
            self._distributed_comm(optimizer, mode="gather")

        if self.is_current_node_a_slowmo_shard:
            self._perform_local_optimization(optimizer)

        if self.slowmo_memory_efficient:
            self._distributed_comm(optimizer, mode="scatter")

    def _perform_local_optimization(self, optimizer: torch.optim.Optimizer) -> None:
        """Performs the slow momentum on the local shard"""
        assert self.portion_start is not None

        with torch.no_grad():
            offset = 0
            for group in optimizer.param_groups:
                # perform local slowmo for p
                for p in group["params"]:
                    numel = p.numel()

                    if offset + numel > self.portion_start and offset < self.portion_end:

                        # start and end for each
                        overall_start = max(self.portion_start, offset)
                        overall_end = min(self.portion_end, offset + numel)

                        p_start = overall_start - offset
                        p_end = overall_end - offset

                        buffer_start = overall_start - self.portion_start
                        buffer_end = overall_end - self.portion_start

                        # let's see size of p and split based on that
                        current_p = p.view(-1)[p_start:p_end]
                        current_p_gmb = self.global_momentum_buffer[buffer_start:buffer_end]
                        current_p_old = self.old_params[buffer_start:buffer_end]

                        current_p_gmb.mul_(self.slowmo_momentum).sub_(current_p, alpha=1 / group["lr"]).add_(
                            current_p_old, alpha=1 / group["lr"]
                        )
                        current_p_old.add_(current_p_gmb, alpha=-group["lr"] * self.slowmo_lr)  # type: ignore
                        current_p.copy_(current_p_old)

                    offset += numel

    def _register_hooks(self) -> None:
        """
        Registers push-sum de-bias/bias hooks in pre-forward/post-backward
        passes in all leaf modules
        """
        self.register_forward_pre_hook(self.__make_forward_pre_hook())
        self.register_backward_hook(self.__make_backward_hook())

    def __make_backward_hook(self) -> Callable[..., None]:
        self.logger.debug("making backward hook")

        def hook(*unused: Any) -> None:
            # reduce gradients across devices on a single machine
            if self.local_node_group is not None:
                grads = []
                for p in self.module.parameters():
                    if not p.requires_grad or p.grad is None:
                        continue
                    p.grad.div_(self.nprocs_per_node)
                    grads.append(p.grad)
                self.logger.debug("Gradients ready for syncing")

                communication_op = functools.partial(dist.all_reduce, group=self.local_node_group)
                communicate(grads, communication_op, self.logger)
                self.logger.debug("Gradient sync during backward pass in local_group complete")

            if self.sgp:
                # convert model back to ps-numerator
                self._sgp_ps_numerator()

                # gossip during training (not inference)
                if self.gossip_enable and self.overlap and not self._skip_averaging_memory_efficient_slowmo():
                    self._sgp_query_gossip_queue()

        def queue_hook(*unused: Any) -> None:
            Variable._execution_engine.queue_callback(hook)

        return queue_hook

    def __make_forward_pre_hook(self) -> Callable[..., None]:
        self.logger.debug("making forward pre-hook")

        def hook(*unused: Any) -> None:
            """Query gossip queue and de-bias during forward pass"""
            # sync buffers before the forward pass
            self._sync_buffers()

            # gossip during training (not inference)
            if self.sgp:
                if self.gossip_enable and self.overlap and not self._skip_averaging_memory_efficient_slowmo():
                    self._sgp_transfer_params()

                # convert model to de-biased estimate
                self._sgp_unbias()

        return hook

    # SGP related functions

    def _sgp_init(
        self,
        module: torch.nn.Module,
        first_param_dtype: torch.dtype,
        logical_rank: int,
        logical_world_size: int,
        comm_device: Optional[torch.device] = None,
        graph: Optional[GraphManager] = None,
        mixing: Optional[MixingManager] = None,
        push_sum: bool = True,
        overlap: bool = False,
        synch_freq: int = 0,
        use_streams: bool = True,
        slowmo_sgp_average_params: bool = False,
    ) -> None:
        """Perform initialization for Stochastic Gradient Push base algorithm"""

        if graph is None:
            graph = NPDDEGraph(logical_rank, logical_world_size, self.nprocs_per_node, self.local_rank)

        if mixing is None:
            mixing = UniformMixing(graph, comm_device)

        self.dist_config.update({"graph": graph, "mixing": mixing, "push_sum": push_sum})

        self.overlap = overlap
        assert not self.overlap  # currently disabled, see docstring

        self.synch_freq = synch_freq
        self.asynch = synch_freq > 0

        # push-sum weight=1.0 ==> distributed averaging
        self.ps_weight = torch.ones(1, device=comm_device, dtype=first_param_dtype)
        self.is_sgp_ps_numerator = False
        self.gossip_enable = True
        self.gossiping = False
        self.params_mixed = True
        self.gossip_ps_factor = torch.zeros(1, device=comm_device, dtype=first_param_dtype)
        self.gossip_ps_weight = self.ps_weight.clone()
        self.gossip_params = []
        self.gossip_device_buffer = []
        for p in module.parameters():
            cp = cast(torch.nn.Parameter, p.clone().detach_())
            cp = cast(torch.nn.Parameter, cp.cpu().pin_memory() if self._cpu_comm else cp.cuda())
            self.gossip_params.append(cp)
            self.gossip_device_buffer.append(cp)

        # prepare gossip process control objects
        self.gossip_lock = threading.Lock()
        self.gossip_flag = threading.Event()
        self.train_flag = threading.Event()

        if cast(torch.device, self.dist_config["comm_device"]).type != "cpu" and use_streams:
            self.gossip_stream = torch.cuda.Stream()
        else:
            self.gossip_stream = torch.cuda.current_stream()

        if self.process_rank % self.nprocs_per_node == 0:
            self.gossip_thread = threading.Thread(
                target=SlowMoDistributedDataParallel._sgp_gossip_target,
                args=(
                    self.dist_config,
                    self.gossip_flag,
                    self.train_flag,
                    self.gossip_lock,
                    self.gossip_params,
                    self.gossip_device_buffer,
                    self.gossip_ps_weight,
                    self.gossip_ps_factor,
                    self.gossip_stream,
                ),
            )
            self.gossip_thread.daemon = True
            self.gossip_thread.name = "Gossip-Thread"
            self.gossip_thread.start()
        else:
            self.gossip_flag.set()

        # wait for thread to complete initialization
        self.gossip_flag.wait()
        self.gossip_flag.clear()

        # lazy mixing avoids additional bias/de-bias steps
        self.lazy_mixing = not self.asynch and cast(MixingManager, self.dist_config["mixing"]).is_regular()
        self.lazy_ps_factor = self.gossip_ps_factor.clone()
        self.logger.debug("lazy mixing: %r", self.lazy_mixing)

    def state_dict(self) -> Dict[str, Union[torch.Tensor, bool]]:  # type: ignore
        state_dict = super(SlowMoDistributedDataParallel, self).state_dict()
        if self.sgp:
            state_dict["ps_weight"] = self.ps_weight.cpu()
            state_dict["is_sgp_ps_numerator"] = self.is_sgp_ps_numerator  # type: ignore
        return state_dict  # type: ignore

    def load_state_dict(self, state_dict: Dict[str, Union[torch.Tensor, bool]]) -> None:  # type: ignore
        if self.sgp:
            assert isinstance(state_dict, dict)
            self.ps_weight = cast(torch.Tensor, state_dict.pop("ps_weight")).to(
                device=cast(torch.device, self.dist_config["comm_device"])
            )
            self.is_sgp_ps_numerator = cast(bool, state_dict.pop("is_sgp_ps_numerator"))

        super(SlowMoDistributedDataParallel, self).load_state_dict(cast(Dict[str, torch.Tensor], state_dict))

    def _sgp_ps_numerator(self) -> None:
        """Convert model params to ps-numerator"""
        if not self.is_sgp_ps_numerator:
            if not self.lazy_mixing:
                ps_weight = self.ps_weight
                with torch.no_grad():
                    for p in self.module.parameters():
                        p.mul_(cast(torch.Tensor, ps_weight.type(p.dtype)))
            self.is_sgp_ps_numerator = True

    def _sgp_unbias(self) -> None:
        """Convert model params to de-biased estimate"""
        if self.is_sgp_ps_numerator:
            if not self.lazy_mixing:
                ps_weight = self.ps_weight
                with torch.no_grad():
                    for p in self.module.parameters():
                        p.div_(cast(torch.Tensor, ps_weight.type(p.dtype)))  # type: ignore
            self.is_sgp_ps_numerator = False

    def train(self, mode: bool = True) -> "SlowMoDistributedDataParallel":
        super(SlowMoDistributedDataParallel, self).train(mode)
        if self.sgp:
            self.gossip_enable = True
        return self

    def eval(self) -> "SlowMoDistributedDataParallel":
        super(SlowMoDistributedDataParallel, self).eval()
        if self.sgp:
            self.gossip_enable = False
            self._sgp_query_gossip_queue(non_blocking=self.asynch)
        return self

    def _sgp_query_gossip_queue(self, non_blocking: bool = False) -> bool:
        """Check gossip-queue for push-sum residuals and update model"""
        if not self.gossip_enable:
            return False

        self.logger.debug("querying gossip queue")

        # no gossip happening right now so just return
        if not self.gossiping:
            if self.process_rank % self.nprocs_per_node == 0:
                self.logger.warning("not gossiping right now")
            return False

        if not non_blocking and not self.gossip_flag.wait(timeout=HEARTBEAT_TIMEOUT):
            raise RuntimeError("Gossip flag timeout")
            sys.exit()  # HEARTBEAT monitor

        # query gossip thread
        if self.gossip_flag.is_set():
            self.logger.debug("received gossip flag")

            # atomic gossip was interrupted so try again
            if self.gossip_ps_weight[0] == -1:
                self.gossip_flag.clear()
                self.params_mixed = True
                self.gossiping = False
                self._sgp_transfer_params(mix=False)
                return False

            self.lazy_ps_factor.copy_(self.gossip_ps_factor)

            # convert model-params to ps numerators b4 adding residuals
            self._sgp_ps_numerator()

            # add residuals
            self.ps_weight += self.gossip_ps_weight
            if self.lazy_mixing:
                self.ps_weight *= self.lazy_ps_factor
            with torch.no_grad():
                for p, r in zip(self.module.parameters(), self.gossip_device_buffer):
                    p.add_(r)  # type: ignore
                    if self.lazy_mixing:
                        p.mul_(cast(torch.Tensor, self.lazy_ps_factor.type(p.dtype)))

            # update flags
            self.logger.debug("updated ps-weight %f", self.ps_weight)
            self.logger.debug("updated model params")
            self.gossip_flag.clear()
            self.params_mixed = True
            self.gossiping = False
            return True

        return False

    def _sgp_transfer_params(self, mix: bool = True) -> bool:
        """Transfers COPY of model parameters to gossip queue"""
        if not self.gossip_enable or self.process_rank % self.nprocs_per_node != 0:
            return False

        self.logger.debug("transferring model params")

        # don't transfer new params if old params haven't been mixed yet
        if not self.params_mixed:
            self.logger.warning("params not mixed")
            return False

        # using lazy mixing ==> mix on query not transfer
        mix = mix and not self.lazy_mixing

        # Transfer ps-numerators to gossip-process:
        # --
        self._sgp_ps_numerator()
        if mix:
            self.ps_weight *= self.gossip_ps_factor
        self.gossip_ps_weight.copy_(self.ps_weight)
        # --
        # params gpu-gpu copy (fast)
        # --
        with torch.no_grad():
            for p, gossip_device_buffer_elem in zip(self.module.parameters(), self.gossip_device_buffer):
                if mix:
                    p.mul_(cast(torch.Tensor, self.gossip_ps_factor.type(p.dtype)))
                gossip_device_buffer_elem.copy_(p)
        # --
        # buffer to gossip-thread copy (potentially slow, but asynchronous)
        # --
        self.gossip_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.gossip_stream):
            for b, gp in zip(self.gossip_device_buffer, self.gossip_params):
                gp.copy_(b, non_blocking=True)

        # --

        # update flags
        self.logger.debug("transferred model params")
        self.params_mixed = False
        self.gossiping = True
        self.train_flag.set()
        return True

    @staticmethod
    def _sgp_gossip_into_receive_buffer(
        send_buffer: List[torch.Tensor],
        gossiper: Gossiper,
        receive_buffer: List[torch.Tensor],
        gossip_ps_weight: torch.Tensor,
        gossip_lock: threading.Lock,
        dist_config: Dict[Any, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # flatten parameters before sending
        out_msg = flatten_tensors(send_buffer)

        # send and receive parameters
        with gossip_lock:
            in_msg, ps_weight = gossiper.mix(out_msg, gossip_ps_weight)
            ps_factor = gossiper.mixing_weights["lo"]

        # unflatten parameters
        with torch.no_grad():
            for r, g in zip(unflatten_tensors(in_msg, send_buffer), receive_buffer):
                if dist_config["cpu_comm"]:
                    g.copy_(r, non_blocking=True)
                else:
                    g.copy_(r)

        return ps_weight, ps_factor

    @staticmethod
    def _sgp_gossip_target(
        dist_config: Dict[Any, Any],
        gossip_flag: threading.Event,
        train_flag: threading.Event,
        gossip_lock: threading.Lock,
        gossip_params: List[torch.Tensor],
        gossip_device_buffer: List[torch.Tensor],
        gossip_ps_weight: torch.Tensor,
        gossip_ps_factor: torch.Tensor,
        gossip_stream: torch.cuda.Stream,
    ) -> None:
        """Gossip thread, which performs push-sum on model params"""
        logger = make_logger(dist_config["logical_rank"], dist_config["verbose"])

        gossip_params_by_dtype = group_by_dtype(gossip_params)
        gossip_device_buffer_by_dtype = group_by_dtype(gossip_device_buffer)

        gossipers = {}
        # init gossip instance
        gossiper_class = PushSum if dist_config["push_sum"] else PushPull
        for dtype in gossip_params_by_dtype:
            gossipers[dtype] = gossiper_class(
                flatten_tensors(gossip_params_by_dtype[dtype]),
                device=cast(torch.device, dist_config["comm_device"]),
                graph=cast(GraphManager, dist_config["graph"]),
                mixing=cast(MixingManager, dist_config["mixing"]),
                rank=dist_config["process_rank"],
                world_size=dist_config["logical_world_size"],
                logger=logger,
            )

        dist_config["gossipers"] = gossipers
        gossip_ps_factor.copy_(gossipers[list(gossipers)[0]].mixing_weights["lo"])
        gossip_flag.set()

        # gossip loop
        while True:
            train_flag.wait()
            logger.debug("received train-flag")
            try:
                with torch.cuda.stream(gossip_stream):
                    for dtype in gossip_params_by_dtype:
                        (ps_weight, ps_factor,) = SlowMoDistributedDataParallel._sgp_gossip_into_receive_buffer(
                            gossip_params_by_dtype[dtype],
                            gossipers[dtype],
                            gossip_device_buffer_by_dtype[dtype],
                            gossip_ps_weight,
                            gossip_lock,
                            dist_config,
                        )
                    gossip_ps_weight.copy_(ps_weight)
                    gossip_ps_factor.copy_(ps_factor)
            except RuntimeError as e:
                logger.warning("received runtime error {}".format(e))
                for gossiper in gossipers.values():
                    gossiper.clean_msg_buffers_()
                gossip_ps_weight.fill_(-1)
            finally:
                # Make sure all queued operations are complete
                gossip_stream.synchronize()
                # give main thread go-ahead to read our gossip buffer
                train_flag.clear()
                gossip_flag.set()
