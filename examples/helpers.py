import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def dist_init(rank, world_size):
    backend = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO  # type: ignore
    print(f"Using backend: {backend}")
    dist.init_process_group(backend=backend, init_method="tcp://localhost:29501", rank=rank, world_size=world_size)


def getModel():
    return nn.Sequential(torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 5))


def getData(n_batches=1):
    return [(torch.randn(20, 10), torch.randint(0, 2, size=(20, 1)).squeeze()) for i in range(n_batches)]


def getLossFun():
    return F.nll_loss
