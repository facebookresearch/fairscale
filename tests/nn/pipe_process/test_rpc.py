import copy
import os

import pytest
import torch
from torch import nn
from torch.distributed import rpc

from fairscale.fair_dev.testing.testing import get_worker_map, torch_spawn
from fairscale.internal import torch_version
from fairscale.nn.model_parallel.initialize import get_pipeline_parallel_group
from fairscale.nn.pipe import PipeRPCWrapper


def init_rpc():
    os.environ["MASTER_PORT"] = "10639"
    init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    rpc.init_rpc(
        f"Test{torch.distributed.get_rank()}",
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
        backend=rpc.BackendType.TENSORPIPE,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method=init_method),
    )


@torch_spawn([2])
@pytest.mark.skipif("OMPI_COMM_WORLD_RANK" not in os.environ, reason="mpi required")
def basic_rpc():
    init_rpc()
    if torch.distributed.get_rank() != 0:
        rpc.shutdown()
        torch.distributed.barrier()
        return

    model = [nn.Linear(10, 10), nn.ReLU()]
    pipe = PipeRPCWrapper(model, [1, 1], input_device=torch.cuda.current_device(), worker_map=get_worker_map())

    pipe.foreach_worker(register_optimizer, include_self=True)

    inputs = torch.rand(10).cuda()
    output = pipe(inputs)
    loss = output.mean()
    loss.backward()

    pipe.foreach_worker(step_optimizer, include_self=True)

    pipe.eval()

    rpc.shutdown()
    torch.distributed.barrier()


def register_optimizer(ctx, model):
    if len(list(model.parameters())) > 0:
        model.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        model.optimizer = None


def step_optimizer(ctx, model):
    if model.optimizer:
        model.optimizer.step()


def check_pipe_against_reference(balance, model_constructor, checkpoint="except_last", custom_inputs=None):
    model = model_constructor()
    reference_model = model_constructor()
    for src, dst in zip(model, reference_model):
        dst.load_state_dict(copy.deepcopy(src.state_dict()))

    reference_model = nn.Sequential(*reference_model).cuda()

    pipe = PipeRPCWrapper(
        model,
        balance,
        input_device=torch.cuda.current_device(),
        worker_map=get_worker_map(),
        checkpoint=checkpoint,
    )

    pipe.foreach_worker(register_optimizer, include_self=True)
    register_optimizer(None, reference_model)

    inputs = torch.rand(10).cuda()
    target = torch.rand(10).cuda()
    cloned = inputs.clone()
    output = pipe(inputs)
    ref_out = reference_model(inputs)

    assert torch.equal(ref_out.cpu(), output.cpu())

    for out in output, ref_out:
        target = target.to(out.device)
        loss = nn.MSELoss()(out, target)
        loss.backward()

    pipe.foreach_worker(step_optimizer, include_self=True)
    step_optimizer(None, reference_model.cuda())

    pipe.eval()
    reference_model.eval()

    final_output = pipe(inputs)
    final_ref = reference_model(inputs.cuda())

    assert torch.equal(final_output.cpu(), final_ref.cpu())


@torch_spawn([3])
@pytest.mark.skipif("OMPI_COMM_WORLD_RANK" not in os.environ, reason="mpi required")
def rpc_optimizer():

    init_rpc()
    if torch.distributed.get_rank() != 0:
        rpc.shutdown()
        torch.distributed.barrier()
        return

    def model_with_reuse():
        reused_1 = nn.Linear(10, 10)
        return [reused_1, nn.ReLU(), reused_1, nn.ReLU(), reused_1, nn.ReLU()]

    check_pipe_against_reference(
        [2, 2, 2],
        lambda: [nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10), nn.ReLU()],
    )
    check_pipe_against_reference([2, 1, 1], model_with_reuse)

    rpc.shutdown()
    torch.distributed.barrier()


@torch_spawn([6])
@pytest.mark.skipif("OMPI_COMM_WORLD_RANK" not in os.environ, reason="mpi required")
def rpc_megatron_reuse():

    from fairscale.nn.model_parallel import layers
    from fairscale.nn.model_parallel.initialize import destroy_model_parallel, initialize_model_parallel

    def make_model_simple():
        return [
            layers.ColumnParallelLinear(10, 10),
            nn.ReLU(),
            layers.RowParallelLinear(10, 10),
            nn.ReLU(),
            layers.ColumnParallelLinear(10, 10),
            nn.ReLU(),
            layers.RowParallelLinear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
        ]

    def make_model_with_reuse():
        column = layers.ColumnParallelLinear(10, 10)
        row = layers.RowParallelLinear(10, 10)
        return [
            column,
            nn.ReLU(),
            row,
            nn.ReLU(),
            column,
            nn.ReLU(),
            row,
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
        ]

    destroy_model_parallel()
    torch.distributed.destroy_process_group()
    torch.distributed.init_process_group("gloo", rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
    initialize_model_parallel(2, 3, model_parallel_backend="nccl", pipeline_backend="mpi")

    init_rpc()
    if get_pipeline_parallel_group().rank() != 0:
        rpc.shutdown()
        torch.distributed.barrier()
        return

    check_pipe_against_reference([4, 4, 2], make_model_simple, "always")
    check_pipe_against_reference([4, 2, 2], make_model_with_reuse)

    rpc.shutdown()
    torch.distributed.barrier()


@torch_spawn([3])
@pytest.mark.skipif("OMPI_COMM_WORLD_RANK" not in os.environ, reason="mpi required")
def rpc_reuse_in_final_stage():

    # 'reused' and 'reused2' are located on stage 2, so the backward pass for
    # the final stage will need to first send gradients to stage 2, then receive
    # gradients from stage 2. This tests custom logic to handle reuse of layers
    # in the final stage of the pipeline.

    reused = nn.Linear(10, 10)
    reused2 = nn.Linear(10, 10)
    model = [
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        reused2,
        nn.ReLU(),
        reused,
        nn.ReLU(),
        reused,
        reused2,
        nn.ReLU(),
        reused,
        nn.ReLU(),
    ]
    balance = [2, 3, 4]

    init_rpc()

    if torch.distributed.get_rank() != 0:
        rpc.shutdown()
        torch.distributed.barrier()
        return

    pipe = PipeRPCWrapper(model, balance, worker_map=get_worker_map())

    inputs = torch.rand(10).cuda()
    target = torch.rand(10).cuda()
    output = pipe(inputs)
    nn.MSELoss()(output, target).backward()
    output = pipe(inputs)
    nn.MSELoss()(output, target).backward()
    rpc.shutdown()
    torch.distributed.barrier()


@torch_spawn([3])
@pytest.mark.skipif("OMPI_COMM_WORLD_RANK" not in os.environ, reason="mpi required")
def rpc_multiple_tensors():
    class FuseTwo(nn.Module):
        def forward(self, left, right):
            return left + right

    class SplitTwo(nn.Module):
        def forward(self, inputs):
            return (inputs, 2 * inputs)


@torch_spawn([2])
@pytest.mark.skipif("OMPI_COMM_WORLD_RANK" in os.environ, reason="no mpi")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
# TODO(msb) Fix this
@pytest.mark.skipif(torch_version() >= (1, 8, 0), reason="disabled for torch 1.8.0")
def construct_only_rank_zero():
    model = [nn.Linear(10, 10), nn.ReLU()]
    if torch.distributed.get_rank() == 0:
        PipeRPCWrapper(model, [1, 1], worker_map=get_worker_map())
        rpc.shutdown()
    else:
        # Must enter rpc loop to complte PipeRPCWrapper constructor above
        rpc.shutdown()

        with pytest.raises(AssertionError):
            PipeRPCWrapper(model, [1, 1], worker_map=get_worker_map())
