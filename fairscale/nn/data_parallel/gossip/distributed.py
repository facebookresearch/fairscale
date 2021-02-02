# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

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

if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group

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


class SlowMoAlgorithms(Enum):
    LOCALSGD = 1
    SGP = 2
    NONE = 3


class GossipDataParallel(Module):
    """ Distributed Gossip model wrapper """

    def __init__(
        self,
        module: torch.nn.Module,
        nprocs_per_node: int = 1,  # To be specified as the number of GPUs in a node
        broadcast_buffers: bool = True,  # Copy from Pytorch DDP
        # SlowMo parameters
        slowmo: bool = True,
        # Might need to be tuned depending on the use case
        slowmo_momentum: float = 0.5,
        # Most probably don't need to be changed
        slowmo_frequency: int = 48,
        slowmo_lr: float = 1.0,
        # Only to be tuned if slow momentum is becoming a bottleneck
        slowmo_world_size: int = 32,
        # SlowMo algorithm parameters
        slowmo_base_algorithm: SlowMoAlgorithms = SlowMoAlgorithms.LOCALSGD,
        # LocalSGD
        localsgd_frequency: int = 48,
        # SGP
        graph: Optional[GraphManager] = None,
        mixing: Optional[MixingManager] = None,
        push_sum: bool = True,
        overlap: bool = False,
        synch_freq: int = 0,
        use_streams: bool = True,
        slowmo_sgp_average_params: bool = False,
        # For debugging
        verbose: bool = False,
        profile_mode: bool = False,
        # Most probably not needed. Process groups can be specified only if they are to be re-used to save memory
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        global_group: Optional[torch.distributed.ProcessGroup] = None,
        master_group: Optional[torch.distributed.ProcessGroup] = None,
        local_node_group: Optional[torch.distributed.ProcessGroup] = None,
        comm_device: Optional[torch.device] = None,
    ) -> None:
        super(GossipDataParallel, self).__init__()

        # NCCL_BLOCKING_WAIT causes issues with using multiple process groups
        assert os.environ.get("NCCL_BLOCKING_WAIT", "0") == "0"

        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)

        self.nprocs_per_node = nprocs_per_node

        if world_size is None or rank is None:
            assert dist.is_initialized()
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        self.process_rank = rank

        # Only create an adapter if debug logging is enabled to avoid additional overhead
        if self.logger.isEnabledFor(logging.DEBUG):
            # Set custom adapter on top of logger
            self.logger = cast(logging.Logger, MultiProcessAdapter(self.logger, {"process_num": self.process_rank}))

        # TODO: move initialization of process groups to a helper function
        if global_group is None:
            # Using private member to avoid creating another process group
            self.global_group = _get_default_group()
        else:
            self.global_group = global_group
        self.logger.debug("Global group set")

        self.process_group = self.global_group
        if self.nprocs_per_node > 1:
            self.local_rank = self.process_rank % self.nprocs_per_node
            world_size //= nprocs_per_node
            rank //= nprocs_per_node
            if local_node_group is None:
                self.logger.debug("Initializing local process groups")
                for node in range(world_size):
                    node_processes_ranks = list(range(node * self.nprocs_per_node, (node + 1) * self.nprocs_per_node,))
                    # Process group to communicate between processes on this
                    # machine
                    new_local_group = create_process_group(node_processes_ranks)
                    if self.process_rank in node_processes_ranks:
                        self.local_node_group = new_local_group
                self.logger.debug("Initialization of local groups complete")
            else:
                self.local_node_group = local_node_group
            if master_group is None:
                self.logger.debug("Maybe initializing master process group")
                master_nodes = [i for i in range(dist.get_world_size()) if i % nprocs_per_node == 0]
                self.master_group = create_process_group(master_nodes) if len(master_nodes) > 1 else None
                if self.master_group is not None and self.process_rank in master_nodes:
                    self.logger.debug("Initialization of master group complete")
            else:
                self.master_group = master_group
        else:
            self.local_rank = 0
            if master_group is None:
                self.master_group = self.global_group
            else:
                self.master_group = master_group
        self.logger.debug("Initialization of all process groups complete")

        # put model on output device
        self.module = module
        self.broadcast_buffers = broadcast_buffers
        first_param_dtype = next(self.module.parameters()).dtype

        # prepare local intra-node all-reduce objects
        self.broadcast_bucket_size = 10 * 1024 * 1024  # bytes
        self.nccl_reduce_bucket_size = 256 * 1024 * 1024  # bytes  # TODO: Maybe not needed
        self.module_buffers = list(self.module.buffers())

        # choose communication device based on backend
        if comm_device is None:
            cpu_comm = True if dist.get_backend() == "gloo" else False
            comm_device = torch.device("cpu") if cpu_comm else torch.device("cuda")
        self.__cpu_comm = comm_device.type == "cpu"

        # distributed backend config
        self.dist_config = {
            "verbose": verbose,
            "comm_device": comm_device,
            "rank": rank,
            "process_rank": self.process_rank,
            "world_size": world_size,
            "cpu_comm": self.__cpu_comm,
        }
        self.profile_mode = profile_mode
        self.num_updates = 0
        self.portion_start: Optional[int] = None

        self.slowmo = False if slowmo_lr == 1 and slowmo_momentum == 0 else slowmo

        # slowmo being set to False is equivalent to slowmo_lr being set to 1 and slowmo_momentum being set to 0
        # This condition is ensuring the values are safe to use even when slowmo is disabled
        self.slowmo_lr = slowmo_lr if self.slowmo else 1
        self.slowmo_momentum = slowmo_momentum if self.slowmo else 0

        self.slowmo_frequency = slowmo_frequency
        self.slowmo_sgp_average_params = slowmo_sgp_average_params
        self.slowmo_base_algorithm = slowmo_base_algorithm
        self.localsgd = slowmo_base_algorithm == SlowMoAlgorithms.LOCALSGD
        self.sgp = slowmo_base_algorithm == SlowMoAlgorithms.SGP

        self.localsgd_frequency = localsgd_frequency
        self.ef1: Optional[List[torch.Tensor]] = None
        self.global_momentum_buffers_initialized = False

        if self.master_group is None:
            if self.localsgd or self.sgp:
                self.localsgd = self.sgp = False
                self.logger.warning("Disabling LocalSGD and SGP since a local allreduce will suffice")

        if self.slowmo and not self.localsgd and not self.sgp:
            self.logger.warning("SlowMo is being used without LocalSGD and SGP")

        self.slowmo_world_size = min(dist.get_world_size(), slowmo_world_size)
        self.is_computing_slowmo = self.process_rank < self.slowmo_world_size

        self.nprocs_per_node_device = torch.tensor([self.nprocs_per_node], device=comm_device, dtype=first_param_dtype)

        if self.sgp:
            if graph is None:
                graph = NPDDEGraph(rank, world_size, self.nprocs_per_node, self.local_rank)

            if mixing is None:
                mixing = UniformMixing(graph, comm_device)

            self.dist_config.update({"graph": graph, "mixing": mixing, "push_sum": push_sum})

            self.overlap = overlap
            # TODO: fix error feedback implementation for SGP-overlap
            assert self.overlap is False

            self.synch_freq = synch_freq
            self.asynch = synch_freq > 0

            # push-sum weight=1.0 ==> distributed averaging
            self.ps_weight = torch.ones(1, device=comm_device, dtype=first_param_dtype)
            self.is_ps_numerator = False
            self.gossip_enable = True
            self.gossiping = False
            self.params_mixed = True
            self.gossip_ps_factor = torch.zeros(1, device=comm_device, dtype=first_param_dtype)
            self.gossip_ps_weight = self.ps_weight.clone()
            self.gossip_params = []
            self.gossip_device_buffer = []
            for p in module.parameters():
                cp = cast(torch.nn.Parameter, p.clone().detach_())
                cp = cast(torch.nn.Parameter, cp.cpu().pin_memory() if self.__cpu_comm else cp.cuda())
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
                    target=GossipDataParallel._gossip_target,
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

        # register ps/grad-reduction hooks
        self.__register_hooks()

        self.logger.debug("Initialization of GossipDataParallel complete")

    def state_dict(self) -> Dict[str, Union[torch.Tensor, bool]]:  # type: ignore
        state_dict = super(GossipDataParallel, self).state_dict()
        if self.sgp:
            state_dict["ps_weight"] = self.ps_weight.cpu()
            state_dict["is_ps_numerator"] = self.is_ps_numerator  # type: ignore
        return state_dict  # type: ignore

    def load_state_dict(self, state_dict: Dict[str, Union[torch.Tensor, bool]]) -> None:  # type: ignore
        if self.sgp:
            assert isinstance(state_dict, dict)
            self.ps_weight = cast(torch.Tensor, state_dict.pop("ps_weight")).to(
                device=cast(torch.device, self.dist_config["comm_device"])
            )
            self.is_ps_numerator = cast(bool, state_dict.pop("is_ps_numerator"))
            super(GossipDataParallel, self).load_state_dict(cast(Dict[str, torch.Tensor], state_dict))
        else:
            super(GossipDataParallel, self).load_state_dict(cast(Dict[str, torch.Tensor], state_dict))

    def forward(self, *inputs: Any, **kwargs: Any) -> Union[torch.Tensor, List[torch.Tensor]]:
        """ Forward pass performed in parallel across all devices on node """
        # scatter inputs onto devices
        self.module
        # print(inputs)
        # print(*inputs)
        # print(*inputs[0])
        return self.module(*inputs, **kwargs)  # type: ignore

    def _sync_params(self) -> None:
        """ Synchronize parameters across devices (intra-node) """
        if self.nprocs_per_node <= 1 or self.local_node_group is None:
            return

        # intra-node parameter sync
        params = cast(List[torch.Tensor], list(self.module.parameters()))
        communication_op = functools.partial(
            dist.broadcast,
            src=(cast(int, self.dist_config["rank"]) * self.nprocs_per_node),
            group=self.local_node_group,
        )
        communicate(params, communication_op)
        self.logger.debug("Intra-node param sync complete")

    def _sync_buffers(self) -> None:
        """ Made to make it same as the new DDP implementation. """
        """ Synchronize buffers across devices (intra-node) """
        # module buffer sync
        if self.broadcast_buffers and len(self.module_buffers) > 0:
            # Synchronize buffers across processes.
            # The process with rank 0 is considered the authoritative copy.
            self._distributed_broadcast_coalesced(self.module_buffers, self.broadcast_bucket_size)
        self.logger.debug("Intra-node buffer sync complete")

    def _distributed_broadcast_coalesced(self, tensors: List[torch.Tensor], buffer_size: int) -> None:
        dist._broadcast_coalesced(self.process_group, tensors, buffer_size)

    def ps_numerator(self) -> None:
        """ Convert model params to ps-numerator """
        if not self.is_ps_numerator:
            ps_weight = self.ps_weight
            if not self.lazy_mixing:
                with torch.no_grad():
                    for p in self.module.parameters():
                        p.mul_(cast(torch.Tensor, ps_weight.type(p.dtype)))
            self.is_ps_numerator = True

    def unbias(self) -> None:
        """ Convert moel params to de-biased estimate """
        if self.is_ps_numerator:
            ps_weight = self.ps_weight
            if not self.lazy_mixing:
                with torch.no_grad():
                    for p in self.module.parameters():
                        p.div_(cast(torch.Tensor, ps_weight.type(p.dtype)))  # type: ignore
            self.is_ps_numerator = False

    def train(self, mode: bool = True) -> "GossipDataParallel":
        super(GossipDataParallel, self).train(mode)
        if self.sgp:
            self.gossip_enable = True
        return self

    def eval(self) -> "GossipDataParallel":
        super(GossipDataParallel, self).eval()
        if self.sgp:
            self.gossip_enable = False
        if self.sgp:
            self._query_gossip_queue(non_blocking=self.asynch)
        return self

    def block(self) -> None:
        self.logger.debug("blocking")
        dist.barrier()

    def _query_gossip_queue(self, non_blocking: bool = False) -> bool:
        """ Check gossip-queue for push-sum residuals and update model """
        if not self.gossip_enable:
            return False

        self.logger.debug("querying gossip queue")

        # no gossip happening right now so just return
        if not self.gossiping:
            if self.process_rank % self.nprocs_per_node == 0:
                self.logger.warning("not gossiping right now")
            return False

        if not non_blocking:
            if not self.gossip_flag.wait(timeout=HEARTBEAT_TIMEOUT):
                raise NameError("Gossip flag timeout")
                sys.exit()  # HEARTBEAT monitor

        # query gossip thread
        if self.gossip_flag.is_set():
            self.logger.debug("received gossip flag")

            # atomic gossip was interrupted so try again
            if self.gossip_ps_weight[0] == -1:
                self.gossip_flag.clear()
                self.params_mixed = True
                self.gossiping = False
                self.transfer_params(mix=False)
                return False

            self.lazy_ps_factor.copy_(self.gossip_ps_factor)

            # convert model-params to ps numerators b4 adding residuals
            self.ps_numerator()

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

    def create_event_recorder(self, event_name: str) -> EventRecorder:
        return create_event_recorder(event_name, dummy=not self.profile_mode)

    def fp16_fp32_iterator(
        self, optimizer: torch.optim.Optimizer, fp32_params: Optional[torch.Tensor]
    ) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterator for those fp16 parameters which have a fp32 copy"""
        if hasattr(optimizer, "_amp_stash") and hasattr(optimizer._amp_stash, "fp16_groups"):  # type: ignore
            for p_fp16_group, p_fp32_group in zip(
                optimizer._amp_stash.fp16_groups, optimizer._amp_stash.fp32_from_fp16_groups,  # type: ignore
            ):
                for p_fp16, p_fp32 in zip(p_fp16_group, p_fp32_group):
                    yield p_fp16, p_fp32
        elif fp32_params is not None:
            offset = 0
            for p in self.parameters():
                numel = p.numel()
                yield p.view(-1), fp32_params[offset : offset + numel]
                offset += numel

    def should_perform_slowmo(self) -> bool:
        return self.slowmo and (self.num_updates + 1) % self.slowmo_frequency == 0

    def should_perform_localsgd(self) -> bool:
        return self.localsgd and (self.num_updates + 1) % self.localsgd_frequency == 0

    def should_allreduce_params(self) -> bool:
        # We do not all-reduce parameters with local SGD if a slow momentum step is
        # performed, since this step contains a reduce operation already. Note that this
        # also means there is no error feedback correction in that case: it is not needed
        # since communication within the slow momentum step happens in fp32.
        return (self.sgp and self.should_perform_slowmo() and self.slowmo_sgp_average_params) or (
            self.should_perform_localsgd() and not self.should_perform_slowmo()
        )

    def maybe_pre_communicate_error_feedback(self, fp16_fp32_list) -> None:
        ef_rec = self.create_event_recorder("Error feedback")
        if self.sgp or self.should_allreduce_params():
            with torch.no_grad():
                for p_fp16, p_fp32 in fp16_fp32_list:
                    if self.should_allreduce_params():
                        # This division and multiplication with the same number is done
                        # to ensure that we do not lose bits of information when we divide
                        # before the all_reduce. In order to preserve these bits in an
                        # error feedback like manner, we are forcing the bits to be lost
                        # initially, and storing the lost information in error feedback
                        p_fp16.div_(dist.get_world_size(self.master_group))
                        p_fp16.mul_(dist.get_world_size(self.master_group))
                    p_fp32 -= p_fp16.float()

            if self.ef1 is not None:
                with torch.no_grad():
                    for idx, (_, p_fp32) in enumerate(fp16_fp32_list):
                        p_fp32 += self.ef1[idx]
                        p_fp32.div_(2)
        ef_rec.stop()
        self.logger.debug("Error feedback completed")

    def maybe_post_communicate_error_feedback(self, fp16_fp32_list) -> None:
        ef_unroll_rec = self.create_event_recorder("Sync and error feedback unroll rec")
        if self.sgp or self.should_allreduce_params():
            # Error Feedback Reversal
            with torch.no_grad():
                for p, p_fp32 in fp16_fp32_list:
                    p_fp32 += p.float()
        ef_unroll_rec.stop()
        self.logger.debug("Error feedback unroll completed")

    def maybe_perform_sgp(self) -> None:
        if self.sgp and not self.overlap:
            sgp_rec = self.create_event_recorder("SGP")
            if not self.should_allreduce_params():
                self.transfer_params()
                self._query_gossip_queue()
                torch.cuda.synchronize()
            sgp_rec.stop()
            self.logger.debug("SGP completed")

    def maybe_allreduce(self) -> None:
        localsgd_rec = self.create_event_recorder("Localsgd communication time")
        if self.should_allreduce_params():
            communication_op = functools.partial(dist.all_reduce, group=self.master_group)
            params = cast(List[torch.Tensor], list(self.parameters()))
            with torch.no_grad():
                for p in params:
                    p.div_(cast(int, self.dist_config["world_size"]))
            self.logger.debug("Params normalized before localsgd step")

            # Commenting this out as it may cause an overhead. Can be uncommented if needed
            # synch_rec = self.create_event_recorder("Synchronization time for localsgd")
            # dist.barrier()
            # synch_rec.stop()
            # self.logger.debug("Barrier completed before localsgd step")

            communicate(params, communication_op, self.logger)
            torch.cuda.synchronize()
            self.logger.debug("Allreduce completed")
        localsgd_rec.stop()

    def maybe_sync_locally(self) -> None:
        if self.sgp or self.should_allreduce_params():
            self._sync_params()
            torch.cuda.synchronize()

    def maybe_perform_slowmo(self, optimizer) -> None:
        slowmo_rec = self.create_event_recorder("Slowmo")
        if self.should_perform_slowmo():
            self.global_momentum_step(optimizer)
        slowmo_rec.stop()
        self.logger.debug("Global momentum step completed")

    def maybe_copy_back_fp32_parameters(self, fp16_fp32_list) -> None:
        ef_copy_rec = self.create_event_recorder("Error feedback copy back")
        if (self.sgp or self.should_allreduce_params() or self.should_perform_slowmo()) and fp16_fp32_list:
            with torch.no_grad():
                for idx, (p_fp16, p_fp32) in enumerate(fp16_fp32_list):
                    p_fp16.copy_(p_fp32)
        ef_copy_rec.stop()
        self.logger.debug("Error feedback copy-back completed")

    def sgp_overlap_pre_communicate_error_feedback(self, fp16_fp32_list) -> None:
        if self.sgp and self.overlap and fp16_fp32_list:
            # Initialize error feedback for SGP-overlap
            if self.ef1 is None:
                self.ef1 = []
                for _, p_fp32 in fp16_fp32_list:
                    self.ef1.append(p_fp32.clone().detach_())

            with torch.no_grad():
                for idx, (p_fp16, p_fp32) in enumerate(fp16_fp32_list):
                    assert self.ef1 is not None
                    self.ef1[idx].copy_(p_fp32 - p_fp16.float())

    def perform_additional_optimizer_actions(
        self, optimizer: torch.optim.Optimizer, fp32_params: Optional[torch.Tensor] = None
    ) -> None:
        """Perform additional steps needed for SGP/LocalSGD/SlowMo"""
        # Done here in case the global momentum buffers has not been initialized by the caller.
        # In an ideal implementation, this would be called by the caller. We do it here instead of
        # waiting for it to happen in the global_momentum step function so that we store a copy of
        # the version of the parameters at iteration 0 and can use them for a slow momentum step later
        if not self.global_momentum_buffers_initialized:
            self.init_global_momentum_buffers(optimizer)

        fp16_fp32_list = list(self.fp16_fp32_iterator(optimizer, fp32_params))
        self.logger.debug("Created a list of fp16 and fp32 corresponding parameters")

        self.logger.debug(
            "Booleans set. Values - self.should_perform_slowmo()=%r, self.should_perform_localsgd()=%r, self.should_allreduce_params()=%r",
            self.should_perform_slowmo(),
            self.should_perform_localsgd(),
            self.should_allreduce_params(),
        )
        self.logger.debug("Step number(0-indexed)=%d", self.num_updates)

        if (
            self.num_updates == 0
            and fp32_params is None
            and not hasattr(optimizer, "_amp_stash")
            and any(p.dtype == torch.float16 for p in self.parameters())
        ):
            self.logger.warning(
                "WARNING: please set fp32_params in "
                "perform_additional_optimizer_actions in order to "
                "avoid accuracy loss"
            )

        self.maybe_pre_communicate_error_feedback(fp16_fp32_list)
        self.maybe_perform_sgp()
        self.maybe_allreduce()
        self.maybe_sync_locally()
        self.maybe_post_communicate_error_feedback(fp16_fp32_list)
        self.maybe_perform_slowmo(optimizer)
        self.maybe_copy_back_fp32_parameters(fp16_fp32_list)
        if self.sgp and self.overlap:
            self.sgp_overlap_pre_communicate_error_feedback(fp16_fp32_list)

        self.num_updates += 1

    def transfer_params(self, mix: bool = True) -> bool:
        """ Transfers COPY of model parameters to gossip queue """
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
        self.ps_numerator()
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
    def _gossip_into_receive_buffer(
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
            in_msg, ps_weight = gossiper.mix(out_msg, gossip_ps_weight, residual=True)
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
    def _gossip_target(
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
        """ Gossip thread, which performs push-sum on model params """
        logger = make_logger(cast(int, dist_config["rank"]), cast(bool, dist_config["verbose"]))

        gossip_params_by_dtype = group_by_dtype(gossip_params)
        gossip_device_buffer_by_dtype = group_by_dtype(gossip_device_buffer)

        gossipers = {}
        # init gossip instance
        gossiper_class = PushSum if cast(bool, dist_config["push_sum"]) else PushPull
        for dtype in gossip_params_by_dtype:
            gossipers[dtype] = gossiper_class(
                flatten_tensors(gossip_params_by_dtype[dtype]),
                device=cast(torch.device, dist_config["comm_device"]),
                graph=cast(GraphManager, dist_config["graph"]),
                mixing=cast(MixingManager, dist_config["mixing"]),
                rank=cast(int, dist_config["process_rank"]),
                world_size=cast(int, dist_config["world_size"]),
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
                        (ps_weight, ps_factor,) = GossipDataParallel._gossip_into_receive_buffer(
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

    def init_global_momentum_buffers(self, optimizer: torch.optim.Optimizer) -> None:
        if not self.slowmo:
            return

        self.global_momentum_buffers_initialized = True

        total_elements = 0
        params_dtype = None
        for group in optimizer.param_groups:
            for p in group["params"]:
                numel = p.numel()
                total_elements += numel

                # Assert that all parameters have the same device and dtype
                if params_dtype is None:
                    params_dtype, params_device = p.dtype, p.device
                # Check that dtype is fp32 since slow mometum is to be performed in fp32
                assert p.dtype == params_dtype == torch.float32
                assert p.device == params_device

        self.world_portion_length = (total_elements + self.slowmo_world_size - 1) // self.slowmo_world_size

        rank = dist.get_rank()
        if not self.is_computing_slowmo:
            return

        self.portion_start = rank * self.world_portion_length
        self.portion_end = min((rank + 1) * self.world_portion_length, total_elements)

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

    def distributed_comm(self, optimizer: torch.optim.Optimizer, mode: str) -> None:
        offset = 0
        slowmo_comm_lists: List[List[torch.Tensor]] = [[] for _ in range(self.slowmo_world_size)]
        with torch.no_grad():
            for group in optimizer.param_groups:
                # aggregate different parts of p in required node
                for p in group["params"]:
                    numel = p.numel()

                    # gather has a reduce operation so division by world size is needed
                    if mode == "gather":
                        p /= dist.get_world_size()

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

    def global_momentum_step(self, optimizer: torch.optim.Optimizer) -> None:
        if not self.slowmo:
            return

        if not self.global_momentum_buffers_initialized:
            self.init_global_momentum_buffers(optimizer)

        # actual global_momentum_step
        self.distributed_comm(optimizer, mode="gather")

        if self.is_computing_slowmo:
            self.perform_local_optimization(optimizer)

        self.distributed_comm(optimizer, mode="scatter")

    def perform_local_optimization(self, optimizer: torch.optim.Optimizer) -> None:
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

    def __register_hooks(self) -> None:
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
            if self.nprocs_per_node > 1 and self.local_node_group is not None:
                grads = []
                for p in self.module.parameters():
                    if not p.requires_grad or p.grad is None:
                        continue
                    p.grad.div_(dist.get_world_size(self.local_node_group))
                    grads.append(p.grad)
                self.logger.debug("Gradients ready for syncing")

                communication_op = functools.partial(dist.all_reduce, group=self.local_node_group)
                communicate(grads, communication_op, self.logger)
                self.logger.debug("Gradient sync during backward pass in local_group complete")

            if self.sgp:
                # convert model back to ps-numerator
                self.ps_numerator()

                # gossip during training (not inference)
                if self.gossip_enable:
                    if self.overlap:
                        self._query_gossip_queue()

        def queue_hook(*unused: Any) -> None:
            Variable._execution_engine.queue_callback(hook)

        return queue_hook

    def __make_forward_pre_hook(self) -> Callable[..., None]:
        self.logger.debug("making forward pre-hook")

        def hook(*unused: Any) -> None:
            """ Query gossip queue and de-bias during forward pass """
            # sync buffers before the forward pass
            self._sync_buffers()

            # gossip during training (not inference)
            if self.sgp:
                if self.gossip_enable:
                    if self.overlap:
                        self.transfer_params()

                # convert model to de-biased estimate
                self.unbias()

        return hook
