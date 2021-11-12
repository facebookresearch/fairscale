# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from fairscale.nn import FullyShardedDataParallel


class TraceForwardEvent(NamedTuple):
    """
    Complementary trace event collected during the forward pass
    to trace the memory increase and the memory taken by activations
    """

    memory_diff: int
    memory_activations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_diff": self.memory_diff,
            "memory_activations": self.memory_activations,
        }

    @classmethod
    def from_dict(cls, serialized: Dict[str, Any]) -> "TraceForwardEvent":
        return TraceForwardEvent(
            memory_diff=serialized["memory_diff"],
            memory_activations=serialized["memory_activations"],
        )


class TraceBackwardEvent(NamedTuple):
    """
    Complementary trace event collected during the forward pass
    to trace the memory taken by activations
    """

    memory_activations: int

    def to_dict(self) -> Dict[str, Any]:
        return {"memory_activations": self.memory_activations}

    @classmethod
    def from_dict(cls, serialized: Dict[str, Any]) -> "TraceBackwardEvent":
        return TraceBackwardEvent(memory_activations=serialized["memory_activations"])


class LayerMemoryTrace(NamedTuple):
    """
    Trace event providing the current memory usage at a point
    occuring during the forward or backward

        module_name: name of the module under processing
        module_params: size of the module parameters
        allocated: state of the PyTorch allocated memory
        reserved: state of the PyTorch reserved memory
        is_forward: whether the trace was collected during forward
        all_gathered: memory gathered since last event by FSDP
        cumul_all_gathered: total amount of memory currently gathered by FSDP
        event: additional information on the trace
    """

    module_name: str
    module_params: int
    allocated: int
    reserved: int
    is_forward: bool
    all_gathered: int
    cumul_all_gathered: int
    event: Union[TraceForwardEvent, TraceBackwardEvent]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_name": self.module_name,
            "module_params": self.module_params,
            "allocated": self.allocated,
            "reserved": self.reserved,
            "is_forward": self.is_forward,
            "all_gathered": self.all_gathered,
            "cumul_all_gathered": self.cumul_all_gathered,
            "event": self.event.to_dict(),
        }

    @classmethod
    def from_dict(cls, serialized: Dict[str, Any]) -> "LayerMemoryTrace":
        if serialized["is_forward"]:
            event: Union[TraceForwardEvent, TraceBackwardEvent] = TraceForwardEvent.from_dict(serialized["event"])
        else:
            event = TraceBackwardEvent.from_dict(serialized["event"])
        return LayerMemoryTrace(
            module_name=serialized["module_name"],
            module_params=serialized["module_params"],
            allocated=serialized["allocated"],
            reserved=serialized["reserved"],
            is_forward=serialized["is_forward"],
            all_gathered=serialized["all_gathered"],
            cumul_all_gathered=serialized["cumul_all_gathered"],
            event=event,
        )


@dataclass
class LayerwiseMemoryTrackerSummary:
    """
    Summary of the memory allocation during forward/backward
    - max_memory_allocated: the peak of memory allocated
    - max_memory_cached: the peak of memory cached by PyTorch
    - total_activation_allocations: cumulative count of activations allocations
    - total_forward_allocations: cumulative count of forward pass allocations
    - top_forward_activation_producers: layers that allocated the most activations
    """

    max_memory_allocated: int
    max_memory_cached: int
    total_activation_allocations: int
    total_forward_allocations: int
    top_forward_activation_producers: List[LayerMemoryTrace]


class ProcessGroupTrackingEvent(Enum):
    """
    Types of events that can be tracked in the process group:
    - allgather: will track calls to ProcessGroup.allgather
    """

    allgather = auto()


class ProcessGroupTracker:
    """
    To be used as a wrapper around a ProcessGroup to track
    the calls to specific ProcessGroup function such as
    "allgather" calls.

    The tracker will send a notification to the listener
    when such calls occur.

    Best used in conjunction with LayerwiseMemoryTracker:

        ```
        # wrap the group used for FSDP
        group = ProcessGroupTracker(group)

        # use this group when creating FSDP blocks
        model = FullyShardedDataParallel(model, process_group=group),

        # monitor the model as before
        tracker = LayerwiseMemoryTracker()
        tracker.monitor(model)

        # the detailed traces will now contain information
        # about the amount of all gathered data
        tracker.memory_traces
        ```
    """

    def __init__(self, group: Any, listener: Optional[Callable] = None):
        self.group = group
        self.listener = listener

    def __getattr__(self, item: str) -> Any:
        # Forward: for functions not traces
        if item == "allgather":
            # For PyTorch 1.8 and below
            return self._build_wrapper(fct=self.group.allgather)
        elif item == "_allgather_base":
            # For PyTorch 1.9 and above
            return self._build_wrapper(fct=getattr(self.group, item))
        return getattr(self.group, item)

    def _build_wrapper(self, fct: Callable) -> Callable:
        def wrapper(
            output_tensors: Union[torch.Tensor, Sequence[torch.Tensor]],
            input_tensors: Union[torch.Tensor, Sequence[torch.Tensor]],
            *args: list,
            **kwargs: dict,
        ) -> Any:
            if self.listener is not None:
                self.listener(ProcessGroupTrackingEvent.allgather, output_tensors, input_tensors)
            return fct(output_tensors, input_tensors, *args, **kwargs)

        return wrapper


class LayerwiseMemoryTracker:
    """
    Observe a module to get the graph of the memory consumption during
    the forward and backward, layer by layer, with:
    - a breakdown of the memory used (activations memory estimation)
    - additional details such as amount of data exchanged with all gather

    Requires the model to be on a CUDA device to track its memory

    Example usage (no FSDP):

        ```
        # create your model
        model = models.resnet50().cuda()

        # monitor the model
        tracker = LayerwiseMemoryTracker()
        tracker.monitor(model)

        # Do a forward/backward
        criterion(model(input), target).backward()

        # show the plots
        tracker.show_plots()

        # get the detailed traces
        tracker.memory_traces

        # print a summary
        print(tracker.summary)
        ```

    Advanced usage (for FSDP):

        ```
        # wrap the group used for FSDP
        group = ProcessGroupTracker(group)

        # use this group when creating FSDP blocks
        model = FullyShardedDataParallel(model, process_group=group),

        # monitor the model as before
        tracker = LayerwiseMemoryTracker()
        tracker.monitor(model)

        # the detailed traces will now contain information
        # about the amount of all gathered data
        tracker.memory_traces
        ```
    """

    def __init__(self) -> None:
        self.memory_traces: List[LayerMemoryTrace] = []
        self._hooks: List[RemovableHandle] = []
        self._previous_module_name: Optional[str] = None
        self._last_all_gather_memory = 0
        self._cumul_all_gather_memory: List[int] = []
        self._memory_pre_forward = 0
        self._traced_module_names: Set[str] = set()

    def monitor(self, model: nn.Module) -> None:
        """
        Install hooks on the model to track its memory usage
        """
        for name, m in model.named_modules():
            h1 = m.register_forward_pre_hook(self._create_pre_forward_hook(name))
            h2 = m.register_forward_hook(self._create_post_forward_hook(name))
            h3 = m.register_backward_hook(self._create_backward_hook(name))
            self._hooks.extend([h1, h2, h3])
            if isinstance(m, FullyShardedDataParallel):
                if isinstance(m.process_group, ProcessGroupTracker):
                    m.process_group.listener = self._handle_process_group_call
        torch.cuda.empty_cache()

    def clear_traces(self) -> None:
        """
        Clear all the traces: new traces will be written on a clean slate
        """
        self.memory_traces.clear()

    def stop(self) -> None:
        """
        Stop any form of tracking (removes the hooks used to monitor the model)
        """
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._previous_module_name = None
        self._memory_pre_forward = 0
        self._last_all_gather_memory = 0
        self._cumul_all_gather_memory.clear()

    @property
    def forward_traces(self) -> List[LayerMemoryTrace]:
        """
        Get the part of the traces which corresponds to the forward pass
        """
        return [t for t in self.memory_traces if t.is_forward]

    @property
    def backward_traces(self) -> List[LayerMemoryTrace]:
        """
        Get the part of the traces which corresponds to the backward pass
        """
        return [t for t in self.memory_traces if not t.is_forward]

    @property
    def max_memory_allocated(self) -> int:
        """
        Peak memory allocated during the forward/backward pass
        """
        return max(t.allocated for t in self.memory_traces)

    @property
    def max_memory_cached(self) -> int:
        """
        Peak memory cached during the forward/backward pass
        """
        return max(t.reserved for t in self.memory_traces)

    @property
    def summary(self) -> LayerwiseMemoryTrackerSummary:
        """
        A quick summary of interesting statistics on the memory usage
        during the forward/backward pass
        """
        total_diff = sum(t.event.memory_diff for t in self.forward_traces)  # type: ignore
        total_act = sum(t.event.memory_activations for t in self.forward_traces)
        top_act_producers = self.top_forward_activation_producers(top=10)
        return LayerwiseMemoryTrackerSummary(
            max_memory_allocated=self.max_memory_allocated,
            max_memory_cached=self.max_memory_cached,
            total_activation_allocations=total_act,
            total_forward_allocations=total_diff,
            top_forward_activation_producers=top_act_producers,
        )

    def top_forward_activation_producers(self, top: int = 10) -> List[LayerMemoryTrace]:
        """
        What are the top activation producers during the forward pass
        """
        return sorted(self.forward_traces, key=lambda a: a.event.memory_activations, reverse=True)[:top]

    def show_plots(self, figsize: Tuple[int, int] = (16, 20), capture: bool = False) -> Optional[Any]:
        """
        Show useful memory plots. Use "capture=True" to return an image
        rather than displaying the plots.
        """
        return compare_memory_traces_in_plot({"run": self.memory_traces}, figsize=figsize, capture=capture)

    def save_traces(self, path: str) -> None:
        """
        Save the traces in a JSON file
        """
        import json

        with open(path, "w") as f:
            json_traces = [t.to_dict() for t in self.memory_traces]
            json.dump({"traces": json_traces}, f)

    @classmethod
    def load(cls, path: str) -> "LayerwiseMemoryTracker":
        import json

        out = cls()
        with open(path, "r") as f:
            traces = json.load(f)["traces"]
        out.memory_traces = [LayerMemoryTrace.from_dict(t) for t in traces]
        return out

    def _create_pre_forward_hook(self, name: str) -> Callable:
        def _pre_forward_hook(module: nn.Module, inputs: Any) -> None:
            torch.cuda.synchronize()
            allocated, reserved = self._capture_memory()
            self._previous_module_name = name
            self._memory_pre_forward = allocated
            if isinstance(module, FullyShardedDataParallel):
                self._cumul_all_gather_memory.append(0)

        return _pre_forward_hook

    def _handle_process_group_call(self, event: ProcessGroupTrackingEvent, *args: Sequence[Any]) -> None:
        torch.cuda.synchronize()
        if event == ProcessGroupTrackingEvent.allgather:
            outputs, inputs = args
            output_size = self._get_module_output_size(outputs)
            self._last_all_gather_memory += output_size
            if self._cumul_all_gather_memory:
                self._cumul_all_gather_memory[-1] += output_size

    def _create_post_forward_hook(self, name: str) -> Callable:
        def _post_forward_hook(
            module: nn.Module, inputs: Sequence[torch.Tensor], outputs: Sequence[torch.Tensor]
        ) -> None:
            torch.cuda.synchronize()
            if isinstance(module, FullyShardedDataParallel):
                self._cumul_all_gather_memory.pop()

            # Only if it is a leaf module
            if name == self._previous_module_name:
                allocated, reserved = self._capture_memory()
                self._traced_module_names.add(name)

                # Get the memory allocated for output activations
                ys = self._filter_allocated_output(inputs, outputs)
                activations = sum(self._get_module_output_size(y) for y in ys)

                # Compute the memory diff + memory taken by the activations
                self.memory_traces.append(
                    LayerMemoryTrace(
                        module_name=name,
                        module_params=self._get_parameter_size(module),
                        allocated=allocated,
                        reserved=reserved,
                        is_forward=True,
                        all_gathered=self._last_all_gather_memory,
                        cumul_all_gathered=sum(self._cumul_all_gather_memory),
                        event=TraceForwardEvent(
                            memory_diff=allocated - self._memory_pre_forward,
                            memory_activations=activations,
                        ),
                    )
                )
                self._last_all_gather_memory = 0

            # Clean previous forward call values
            self._previous_module_name = None
            self._memory_pre_forward = 0

        return _post_forward_hook

    def _create_backward_hook(self, name: str) -> Callable:
        def _backward_hook(module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor) -> None:
            torch.cuda.synchronize()
            if name not in self._traced_module_names:
                return

            ys = self._filter_allocated_output(grad_input, grad_output)
            memory = sum(self._get_module_output_size(y) for y in ys)
            allocated, reserved = self._capture_memory()
            self.memory_traces.append(
                LayerMemoryTrace(
                    module_name=name,
                    module_params=self._get_parameter_size(module),
                    allocated=allocated,
                    reserved=reserved,
                    is_forward=False,
                    all_gathered=self._last_all_gather_memory,
                    cumul_all_gathered=0,
                    event=TraceBackwardEvent(memory_activations=memory),
                )
            )

            # Cleaning accumulated values since last call
            self._last_all_gather_memory = 0

        return _backward_hook

    @staticmethod
    def _capture_memory() -> Tuple[int, int]:
        torch.cuda.synchronize()
        allocated_mb = torch.cuda.memory_allocated()
        reserved_mb = torch.cuda.memory_reserved()  # type: ignore
        return allocated_mb, reserved_mb

    @classmethod
    def _get_parameter_size(cls, module: nn.Module) -> int:
        return sum(p.numel() * cls._get_dtype_size(p) for p in module.parameters())

    @classmethod
    def _get_module_output_size(cls, xs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> int:
        """
        Return the minimum memory requirement to store the tensors
        provided as parameters
        """
        if isinstance(xs, torch.Tensor):
            x = xs
            p = cls._get_dtype_size(x)
            for d in x.shape:
                p *= d
            return p
        elif isinstance(xs, tuple) or isinstance(xs, list):
            return sum(cls._get_module_output_size(x) for x in xs)
        return 0

    @classmethod
    def _get_dtype_size(cls, x: torch.Tensor) -> int:
        return 2 if x.dtype == torch.float16 else 4

    @classmethod
    def _filter_allocated_output(
        cls, inputs: Union[torch.Tensor, Sequence[torch.Tensor]], outputs: Union[torch.Tensor, Sequence[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Only return the outputs that are allocated and not views, reshape
        or stride of the inputs
        """
        xs = cls._collect_tensors(inputs)
        ys = cls._collect_tensors(outputs)
        return [y for y in ys if all(not cls._is_same_storage(x, y) for x in xs)]

    @staticmethod
    def _is_same_storage(x: torch.Tensor, y: torch.Tensor) -> bool:
        """
        Indicate if x and y share the same storage, meaning that one of them
        is a view, reshape or stride of the other or from a common tensor
        """
        return x.storage().data_ptr() == y.storage().data_ptr()  # type: ignore

    @staticmethod
    def _collect_tensors(module_io_tensors: Union[torch.Tensor, Sequence[torch.Tensor]]) -> List[torch.Tensor]:
        """
        Extract the tensors out of the provided input or output of a nn.Module
        """
        tensors = []
        to_visit = [module_io_tensors]
        while to_visit:
            x = to_visit.pop()
            if isinstance(x, torch.Tensor):
                tensors.append(x)
            elif isinstance(x, tuple) or isinstance(x, list):
                to_visit.extend(module_io_tensors)
        return tensors


def find_best_reset_points(activation_sizes: List[int], num_checkpoints: int) -> Tuple[int, List[int]]:
    """
    Assuming constant memory requirement from the model, its gradients
    and the associated optimizer state (realistic for small models
    or models that are sharded enough to be considered small), this
    function computes the ideal placement for the checkpoints by
    returning the limits at which we should reset memory.
    """
    n = len(activation_sizes)

    @lru_cache(maxsize=None)
    def visit(pos: int, remaining: int) -> Tuple[int, List[int]]:
        if pos == n:
            return 0, []
        if remaining == 0:
            return sum(activation_sizes[pos:]), []

        min_val = float("inf")
        allocation = []

        current_chunk = 0
        for curr_pos in range(pos, n):
            current_chunk += activation_sizes[curr_pos]
            sub_result, sub_alloc = visit(curr_pos + 1, remaining - 1)
            result = max(current_chunk, sub_result)
            if result < min_val:
                min_val = result
                allocation = list(sub_alloc)
                allocation.append(curr_pos + 1)

        return int(min_val), allocation

    best_score, best_allocation = visit(0, num_checkpoints)
    return best_score, best_allocation[::-1]


@dataclass
class SuggestedCheckpoints:
    max_memory: int
    split_modules: List[str]
    all_modules: List[str]


def suggest_checkpoint_location(
    traces: List[LayerMemoryTrace], num_checkpoints: int, num_skipped_layers: int = 0
) -> SuggestedCheckpoints:
    """
    Given a trace of a model, collected with or without checkpoint,
    return the best places to insert a reset of activation memory.

    The names of the returned modules are the boundaries of the
    suggested checkpoint_wrapper wrappings
    """

    # From the traces, extract how much activation memory
    # is generated during the forward pass, layer by layer
    visited = set()
    modules, allocations = [], []
    for t in traces:
        if t.is_forward:
            name = t.module_name
            memory = t.event.memory_activations
            if name not in visited:
                visited.add(name)
                modules.append(name)
                allocations.append(memory)

    # To skip some layers where we do not want activations
    if num_skipped_layers:
        modules = modules[num_skipped_layers:]
        allocations = allocations[num_skipped_layers:]

    # Compute the best positions to reset the memory
    max_memory, reset_indices = find_best_reset_points(allocations, num_checkpoints=num_checkpoints)

    # Then map it back to module names
    return SuggestedCheckpoints(
        max_memory=max_memory,
        split_modules=[modules[i] for i in reset_indices],
        all_modules=modules,
    )


def _assert_visualisation_library_installed() -> None:
    try:
        import PIL  # NOQA
        import matplotlib  # NOQA
    except ImportError:
        install_matplotlib = "pip install matplotlib"
        install_pil = "pip install Pillow"
        error_message = "Visualizing memory plots requires matplotlib and Pillow installed"
        assert False, f"{error_message}: {install_matplotlib}, {install_pil}"


def compare_memory_traces_in_plot(
    memory_traces_by_job: Dict[str, List[LayerMemoryTrace]],
    figsize: Tuple[int, int] = (16, 20),
    capture: bool = False,
) -> Optional[Any]:
    """
    Create a plot of the memory allocation over time during the forward/backward
    passes, with a breakdown of the memory used for activation VS parameters
    """
    _assert_visualisation_library_installed()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, ncols=2, nrows=3)
    graph_creator = _MemoryGraphCreator()

    ax[0, 0].set_title("memory allocated")
    for job_name, memory_traces in memory_traces_by_job.items():
        graph_creator.allocated_memory_curve(ax[0, 0], job_name, memory_traces)
    if len(memory_traces_by_job) > 1:
        ax[0, 0].legend()

    ax[0, 1].set_title("memory reserved")
    for job_name, memory_traces in memory_traces_by_job.items():
        graph_creator.reserved_memory_curve(ax[0, 1], job_name, memory_traces)
    if len(memory_traces_by_job) > 1:
        ax[0, 1].legend()

    ax[1, 0].set_title("activation allocations")
    for job_name, memory_traces in memory_traces_by_job.items():
        graph_creator.activation_allocations(ax[1, 0], job_name, memory_traces)
    if len(memory_traces_by_job) > 1:
        ax[1, 0].legend()

    ax[1, 1].set_title("cumulative forward activations")
    for job_name, memory_traces in memory_traces_by_job.items():
        graph_creator.cumulative_activations(ax[1, 1], job_name, memory_traces)
    if len(memory_traces_by_job) > 1:
        ax[1, 1].legend()

    ax[2, 0].set_title("all gathered memory")
    for job_name, memory_traces in memory_traces_by_job.items():
        graph_creator.all_gathered_memory(ax[2, 0], job_name, memory_traces)
    if len(memory_traces_by_job) > 1:
        ax[2, 0].legend()

    ax[2, 1].set_title("parameter memory")
    for job_name, memory_traces in memory_traces_by_job.items():
        graph_creator.module_parameters(ax[2, 1], job_name, memory_traces)
    if len(memory_traces_by_job) > 1:
        ax[2, 1].legend()

    if not capture:
        plt.show()
        return None
    else:
        return matplotlib_figure_to_image(fig)


class _MemoryGraphCreator:
    """
    Helper class to create graphs to display memory
    """

    def __init__(self) -> None:
        import matplotlib

        self.font = {
            "family": matplotlib.rcParams["font.family"],
            "weight": "normal",
            "size": 12,
        }

    def allocated_memory_curve(self, ax: Any, job_name: str, memory_traces: List[LayerMemoryTrace]) -> None:
        allocated_memory = [t.allocated for t in memory_traces]
        x, y_forward, y_backward = self._split_forward_backward(memory_traces, allocated_memory)
        ax.plot(x, y_forward, x, y_backward, label=job_name)

        max_index = np.argmax(allocated_memory)
        max_trace = memory_traces[max_index]
        max_module = ".".join([n for n in max_trace.module_name.split(".") if not n.startswith("_")])
        max_phase = "fwd" if max_trace.is_forward else "bwd"
        ax.set_ylim([None, max_trace.allocated * 1.1])
        x_text, y_text = max(0, max_index * 0.8), max_trace.allocated * 1.04  # type: ignore
        ax.text(x_text, y_text, f"{max_module} ({max_phase})", fontdict=self.font)
        self._y_axis_in_gigabytes(ax)

    def reserved_memory_curve(self, ax: Any, job_name: str, memory_traces: List[LayerMemoryTrace]) -> None:
        reserved_memory = [t.reserved for t in memory_traces]
        x, y_forward, y_backward = self._split_forward_backward(memory_traces, reserved_memory)
        ax.plot(x, y_forward, x, y_backward, label=job_name)
        self._y_axis_in_gigabytes(ax)

    def activation_allocations(self, ax: Any, job_name: str, memory_traces: List[LayerMemoryTrace]) -> None:
        event_allocations = [t.event.memory_activations for t in memory_traces]
        x, y_forward, y_backward = self._split_forward_backward(memory_traces, event_allocations)
        ax.plot(x, y_forward, x, y_backward, label=job_name)
        self._y_axis_in_gigabytes(ax)

    def cumulative_activations(self, ax: Any, job_name: str, memory_traces: List[LayerMemoryTrace]) -> None:
        event_allocations = [t.event.memory_activations for t in memory_traces]
        x, y_forward, y_backward = self._split_forward_backward(memory_traces, event_allocations)
        cumulative_forward_activations = np.cumsum(y_forward)
        ax.plot(x, cumulative_forward_activations, label=job_name)
        self._y_axis_in_gigabytes(ax)

    def all_gathered_memory(self, ax: Any, job_name: str, memory_traces: List[LayerMemoryTrace]) -> None:
        # Plot the all_gathered and cumulative all_gathered memory
        gathered_memory = [t.all_gathered for t in memory_traces]
        cumul_gathered_memory = [t.cumul_all_gathered for t in memory_traces]
        x, y_forward, y_backward = self._split_forward_backward(memory_traces, gathered_memory)
        ax.plot(x, y_forward, x, y_backward, label=job_name)
        ax.plot(x, cumul_gathered_memory, label=job_name)
        self._y_axis_in_gigabytes(ax)

        # Adding the name of the layer with max cumulative all_gathered memory
        max_index = np.argmax(cumul_gathered_memory)
        max_trace = memory_traces[max_index]
        max_module = ".".join([n for n in max_trace.module_name.split(".") if not n.startswith("_")])
        ax.set_ylim([None, max_trace.cumul_all_gathered * 1.1])
        x_text, y_text = max(0, max_index * 0.8), max_trace.cumul_all_gathered * 1.04  # type: ignore
        ax.text(x_text, y_text, f"{max_module} (fwd)", fontdict=self.font)

    def module_parameters(self, ax: Any, job_name: str, memory_traces: List[LayerMemoryTrace]) -> None:
        module_parameters = [t.module_params for t in memory_traces]
        x, y_forward, y_backward = self._split_forward_backward(memory_traces, module_parameters)
        ax.plot(x, y_forward, x, y_backward, label=job_name)
        self._y_axis_in_gigabytes(ax)

    @staticmethod
    def _y_axis_in_gigabytes(ax: Any) -> None:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(9, 9))

    @classmethod
    def _split_forward_backward(
        cls, memory_traces: List[LayerMemoryTrace], values: List[Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_values = np.array(list(range(len(memory_traces))))
        mask_forwards, mask_backwards = cls._mask_forward_backward(memory_traces)
        return (
            x_values,
            np.ma.masked_where(mask_backwards, values),  # type: ignore
            np.ma.masked_where(mask_forwards, values),  # type: ignore
        )

    @classmethod
    def _mask_forward_backward(cls, memory_traces: List[LayerMemoryTrace]) -> Tuple[np.ndarray, np.ndarray]:
        mask_forwards = np.array([t.is_forward for t in memory_traces])
        return mask_forwards, ~mask_forwards


@contextmanager
def null_context() -> Iterator[None]:
    yield


def matplotlib_figure_to_image(fig: Any) -> Any:
    """
    Convert a matplotlib figure to an image in RGB format, for instance
    to save it on disk
    """
    import io

    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
