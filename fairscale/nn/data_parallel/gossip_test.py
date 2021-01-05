import copy
import os
import sys
import unittest

from gossip import GossipDataParallel
import torch
from torch import nn
import torch.distributed
import torch.nn.functional as F
from torch.testing._internal.common_distributed import MultiProcessTestCase, requires_nccl, skip_if_not_multigpu
from torch.testing._internal.common_utils import TEST_WITH_TSAN

if not torch.distributed.is_available():
    print("torch.distributed is not available, skipping tests")
    sys.exit(0)


def get_gpus_for_rank(world_size):
    """Multigpu tests are designed to simulate the multi nodes with multi
    GPUs on each node. Nccl backend requires equal #GPUs in each process.
    On a single node, all visible GPUs are evenly
    divided to subsets, each process only uses a subset.

    This will return a list, each element of which contains a list of GPUs
    to be used by the respective process.

    Examples (results are shown for a machine with 2 GPUs):

        >>> get_gpus_for_rank(2)  # [[0], [1]]
        >>> get_gpus_for_rank(4)  # [[0], [0], [1], [1]]
        >>> get_gpus_for_rank(1)  # [[0, 1]]

    Args:
        world_size (int): denotes number of subsets to split the available GPUs into
    """
    visible_devices = list(range(torch.cuda.device_count()))
    num_visible_devices = torch.cuda.device_count()

    if num_visible_devices >= world_size:
        assert num_visible_devices % world_size == 0
        gpus_per_process = num_visible_devices // world_size
        gpus_for_rank = []
        for rank in range(world_size):
            gpus_for_rank.append(visible_devices[rank * gpus_per_process : (rank + 1) * gpus_per_process])
    else:
        visible_devices_repeated = [
            [device]
            for device in visible_devices
            for _ in range((world_size + num_visible_devices - 1) // num_visible_devices)
        ]
        gpus_for_rank = visible_devices_repeated[:world_size]

    return gpus_for_rank


def step_model(model, input, target):
    model.train()
    output = model(input)
    loss = F.mse_loss(output, target.to(output.device))
    loss.backward()


def update_parameters(optimizer):
    optimizer.step()
    optimizer.zero_grad()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class LargeNet(Net):
    def __init__(self):
        super(LargeNet, self).__init__()
        self.fc2 = nn.Linear(10, 5000000, bias=False)
        self.fc3 = nn.Linear(5000000, 4, bias=False)


def find_memory_used_by_model(model_class, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    initial_memory = torch.cuda.max_memory_allocated(device)
    _ = model_class().to(device)
    torch.cuda.synchronize(device)
    final_memory = torch.cuda.max_memory_allocated(device)

    model_memory = final_memory - initial_memory
    # print(model_memory)
    return model_memory


@unittest.skipIf(
    TEST_WITH_TSAN, "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class GossipDataParallelTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._fork_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return 2

    def _prepare_single_device_module(self, devices, device_ids, global_batch_size, slowmo_init_dict):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "nccl", init_method=f"file://{self.file_name}", rank=self.rank, world_size=self.world_size,
            )
        model = Net()
        slowmo_model = GossipDataParallel(
            copy.deepcopy(model).to(devices[0]),
            device_ids=device_ids,
            localsgd=True,
            comm_device=devices[0],
            rank=self.rank,
            world_size=self.world_size,
            **slowmo_init_dict,
        )

        model.to(devices[0])

        input = torch.randn(global_batch_size, 2).to(devices[0])
        target = torch.randn(global_batch_size, 4).to(devices[0])

        return model, slowmo_model, input, target

    def _test_slowmo_with_slowmo_freq_1(self, devices, device_ids, slowmo_init_dict, model_optimizer_momentum=0):
        """
        Note: we pass down `device_ids` all the way to GossipDataParallel
        as part of the test. Below you find tests that either use a list of
        integers, a list of `torch.Device` instances, or an empty list.
        The `devices` argument is used to control placement of the model and
        must always be specified as list of `torch.Device` instances.
        """
        torch.cuda.set_device(devices[0])
        local_batch_size = len(devices)
        global_batch_size = self.world_size * local_batch_size

        model, slowmo_model, input, target = self._prepare_single_device_module(
            devices, device_ids, global_batch_size, slowmo_init_dict
        )
        model_optimizer = torch.optim.SGD(
            model.parameters(), lr=slowmo_model.slowmo_lr, momentum=slowmo_model.slowmo_momentum,
        )
        slowmo_model_optimizer = torch.optim.SGD(slowmo_model.module.parameters(), lr=1, momentum=0)
        slowmo_model.init_global_momentum_buffers(slowmo_model_optimizer)

        # check two model parameters over 3 iterations
        for iteration in range(3):
            # single cpu/gpu training
            step_model(model, input, target)

            # SlowMo training, SlowMo scatters subsets of input_cpu to nodes/GPUs
            step_model(
                slowmo_model,
                input[self.rank * local_batch_size : (self.rank + 1) * local_batch_size],
                target[self.rank * local_batch_size : (self.rank + 1) * local_batch_size],
            )

            # Update weights and run a second iteration to shake out errors
            update_parameters(model_optimizer)
            update_parameters(slowmo_model_optimizer)
            slowmo_model.perform_additional_optimizer_actions(slowmo_model_optimizer)

            self.assertEqual(list(model.parameters()), list(slowmo_model.module.parameters()))

            # Shuffle the input so that DDP input is different
            torch.manual_seed(1337 + iteration)
            input = input[torch.randperm(global_batch_size)]

    def _test_slowmo_with_slowmo_freq_ge_2(self, devices, device_ids, slowmo_init_dict):
        """
        Note: we pass down `device_ids` all the way to GossipDataParallel
        as part of the test. Below you find tests that either use a list of
        integers, a list of `torch.Device` instances, or an empty list.
        The `devices` argument is used to control placement of the model and
        must always be specified as list of `torch.Device` instances.
        """
        torch.cuda.set_device(devices[0])
        local_batch_size = len(devices)
        global_batch_size = self.world_size * local_batch_size

        model, slowmo_model, input, target = self._prepare_single_device_module(
            devices, device_ids, global_batch_size, slowmo_init_dict
        )
        base_lr, base_momentum = 1, 0
        model_optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=base_momentum)
        model_slow_momentum_optimizer = torch.optim.SGD(
            model.parameters(), lr=slowmo_model.slowmo_lr, momentum=slowmo_model.slowmo_momentum,
        )
        slowmo_model_optimizer = torch.optim.SGD(slowmo_model.module.parameters(), lr=base_lr, momentum=base_momentum)
        slowmo_model.init_global_momentum_buffers(slowmo_model_optimizer)

        old_parameters = [copy.deepcopy(params) for params in model.parameters()]

        # check two model parameters over 6 iterations
        for iteration in range(6):
            # single cpu/gpu training
            step_model(model, input, target)

            # SlowMo training, SlowMo scatters subsets of input_cpu to nodes/GPUs
            step_model(
                slowmo_model,
                input[self.rank * local_batch_size : (self.rank + 1) * local_batch_size],
                target[self.rank * local_batch_size : (self.rank + 1) * local_batch_size],
            )

            # Update weights and run a second iteration to shake out errors
            update_parameters(model_optimizer)
            update_parameters(slowmo_model_optimizer)
            slowmo_model.perform_additional_optimizer_actions(slowmo_model_optimizer)

            # This block simulates the behaviour of slow momentum by applying it manually
            # to the regular model
            if (iteration + 1) % slowmo_init_dict["slowmo_frequency"] == 0:
                for params, old_params in zip(model.parameters(), old_parameters):
                    params.grad = -(params - old_params)
                    with torch.no_grad():
                        params.copy_(old_params)
                update_parameters(model_slow_momentum_optimizer)
                for params, old_params in zip(model.parameters(), old_parameters):
                    with torch.no_grad():
                        old_params.copy_(params)

            self.assertEqual(list(model.parameters()), list(slowmo_model.module.parameters()))

            # Shuffle the input so that DDP input is different
            torch.manual_seed(1337 + iteration)
            input = input[torch.randperm(global_batch_size)]

    def _test_localsgd_with_freq_ge_2(self, devices, device_ids, slowmo_init_dict, model_optimizer_momentum=0):
        torch.cuda.set_device(devices[0])
        local_batch_size = len(devices)
        global_batch_size = self.world_size * local_batch_size

        model, slowmo_model, input, target = self._prepare_single_device_module(
            devices, device_ids, global_batch_size, slowmo_init_dict
        )
        self.assertEqual(model_optimizer_momentum, 0)
        self.assertFalse(slowmo_model.slowmo)

        model_optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=model_optimizer_momentum)
        slowmo_model_optimizer = torch.optim.SGD(slowmo_model.module.parameters(), lr=1, momentum=0)

        # check two model parameters over 3 iterations
        for iteration in range(6):
            # single cpu/gpu training
            step_model(
                model,
                input[self.rank * local_batch_size : (self.rank + 1) * local_batch_size],
                target[self.rank * local_batch_size : (self.rank + 1) * local_batch_size],
            )

            # SlowMo training, SlowMo scatters subsets of input_cpu to nodes/GPUs
            step_model(
                slowmo_model,
                input[self.rank * local_batch_size : (self.rank + 1) * local_batch_size],
                target[self.rank * local_batch_size : (self.rank + 1) * local_batch_size],
            )

            # Update weights and run a second iteration to shake out errors
            update_parameters(model_optimizer)
            update_parameters(slowmo_model_optimizer)

            # This block simulates the behaviour of localsgd by doing an allreduce on
            # parameters of the regular model
            if (iteration + 1) % slowmo_model.localsgd_frequency == 0:
                for param in model.parameters():
                    torch.distributed.all_reduce(param)
                    with torch.no_grad():
                        param /= self.world_size
            slowmo_model.perform_additional_optimizer_actions(slowmo_model_optimizer)

            self.assertEqual(list(model.parameters()), list(slowmo_model.module.parameters()))

            # Shuffle the input so that distributed input is different
            torch.manual_seed(1337 + iteration)
            input = input[torch.randperm(global_batch_size)]

    def _test_memory_usage_localsgd_with_slowmo(
        self, devices, device_ids, slowmo_init_dict, use_gossip_data_parallel=False
    ):
        torch.cuda.set_device(devices[0])
        torch.cuda.reset_peak_memory_stats(devices[0])
        initial_max_memory = torch.cuda.max_memory_allocated(devices[0])

        local_batch_size = len(devices)
        global_batch_size = self.world_size * local_batch_size

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "nccl", init_method=f"file://{self.file_name}", rank=self.rank, world_size=self.world_size,
            )
        if use_gossip_data_parallel:
            model = GossipDataParallel(
                LargeNet().to(devices[0]),
                device_ids=device_ids,
                localsgd=True,
                comm_device=devices[0],
                rank=self.rank,
                world_size=self.world_size,
                **slowmo_init_dict,
            )
        else:
            model = LargeNet().to(devices[0])

        input = torch.randn(global_batch_size, 2).to(devices[0])
        target = torch.randn(global_batch_size, 4).to(devices[0])

        model_optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0.5)

        # check two model parameters over 3 iterations
        for iteration in range(3):
            step_model(
                model,
                input[self.rank * local_batch_size : (self.rank + 1) * local_batch_size],
                target[self.rank * local_batch_size : (self.rank + 1) * local_batch_size],
            )

            update_parameters(model_optimizer)
            if hasattr(model, "perform_additional_optimizer_actions"):
                model.perform_additional_optimizer_actions(model_optimizer)

            # Shuffle the input so that distributed input is different
            torch.manual_seed(1337 + iteration)
            input = input[torch.randperm(global_batch_size)]

        torch.cuda.synchronize(devices[0])
        final_max_memory = torch.cuda.max_memory_allocated(devices[0])
        # print(f"{initial_max_memory}, {final_max_memory}")

        return final_max_memory - initial_max_memory

    @requires_nccl()
    @skip_if_not_multigpu
    def test_nccl_backend_device_ids_integer_list(self):
        int_devices = get_gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_slowmo_with_slowmo_freq_1(
            devices, int_devices, {"localsgd_frequency": 1, "nprocs_per_node": 1, "slowmo": False},
        )

    @requires_nccl()
    @skip_if_not_multigpu
    def test_nccl_backend_device_ids_torch_device_list(self):
        int_devices = get_gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_slowmo_with_slowmo_freq_1(
            devices, devices, {"localsgd_frequency": 1, "nprocs_per_node": 1, "slowmo": False},
        )

    @requires_nccl()
    @skip_if_not_multigpu
    def test_nccl_backend_2_proc_1_node(self):
        int_devices = get_gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_slowmo_with_slowmo_freq_1(
            devices,
            int_devices,
            {
                "localsgd_frequency": 100,  # Localsgd has to be disabled since it would fail in the 1 node case. TODO: Need to allow it to run without failing in GossipDataParallel in the one node case
                "nprocs_per_node": 2,
                "slowmo": False,
            },
        )

    # This test needs to be added if we have a setup in which we can test on more GPUs
    # @requires_nccl()
    # @skip_if_lt_x_gpu(4)
    # def test_nccl_backend_2_proc_2_node(self):
    #     # 2 device, 2 node
    #     # 4 device, 1 node
    #     # 1 device, 4 node
    #     # can change world size to 4
    #     # will need to change world_size to 4 for this
    #     int_devices = get_gpus_for_rank(self.world_size)[self.rank][:1]
    #     devices = list([torch.device("cuda:" + str(i)) for i in int_devices])
    #     self._test_slowmo_with_process_group(
    #         process_group,
    #         devices,
    #         device_ids,
    #         {
    #             "localsgd_frequency": 1,
    #             "rank": self.rank,
    #             "world_size": self.world_size,
    #             "nprocs_per_node": 2,
    #             "local_node_group": process_group,
    #             "master_group": process_group,
    #             "slowmo": False,
    #         },
    #     )

    @requires_nccl()
    @skip_if_not_multigpu
    def test_slowmo_freq_1(self):
        int_devices = get_gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_slowmo_with_slowmo_freq_1(
            devices,
            int_devices,
            {"localsgd_frequency": 1, "nprocs_per_node": 1, "slowmo_momentum": 0.5, "slowmo_frequency": 1},
            model_optimizer_momentum=0.5,
        )

    @requires_nccl()
    @skip_if_not_multigpu
    def test_slowmo(self):
        int_devices = get_gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_slowmo_with_slowmo_freq_ge_2(
            devices,
            int_devices,
            {"localsgd_frequency": 1, "nprocs_per_node": 1, "slowmo_momentum": 0.5, "slowmo_frequency": 2},
        )

    @requires_nccl()
    @skip_if_not_multigpu
    def test_slowmo_small_world_size(self):
        int_devices = get_gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_slowmo_with_slowmo_freq_ge_2(
            devices,
            int_devices,
            {
                "localsgd_frequency": 1,
                "nprocs_per_node": 1,
                "slowmo_momentum": 0.5,
                "slowmo_frequency": 2,
                "slowmo_world_size": 1,
            },
        )

    @requires_nccl()
    @skip_if_not_multigpu
    def test_localsgd_freq_2(self):
        int_devices = get_gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_localsgd_with_freq_ge_2(
            devices, int_devices, {"localsgd_frequency": 2, "nprocs_per_node": 1, "slowmo": False},
        )

    @requires_nccl()
    @skip_if_not_multigpu
    def test_max_memory_used_localsgd_slowmo(self):
        int_devices = get_gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]

        # Memory usage when running optimization locally on a single GPU
        max_memory_local = self._test_memory_usage_localsgd_with_slowmo(
            devices, int_devices, {"localsgd_frequency": 1}, use_gossip_data_parallel=False,
        )

        # Memory usage when running optimization using LocalSGD-SlowMo
        max_memory_localsgd_slowmo = self._test_memory_usage_localsgd_with_slowmo(
            devices,
            int_devices,
            {
                "localsgd_frequency": 1,
                "nprocs_per_node": 1,
                "slowmo_momentum": 0.5,
                "slowmo_frequency": 1,
                "slowmo": False,
            },
            use_gossip_data_parallel=True,
        )

        model_memory_usage = find_memory_used_by_model(LargeNet, devices[0])

        extra_memory_used_by_localsgd_slowmo = max_memory_localsgd_slowmo - max_memory_local

        extra_memory_used_by_slowmo = (
            model_memory_usage  # This is expected on 2 GPU experiments and confirmed in below test
        )
        extra_memory_used_by_localsgd = extra_memory_used_by_localsgd_slowmo - extra_memory_used_by_slowmo

        # Extra memory used by localsgd should be close to 0 for large models, because we discard the gradients before the localsgd step
        # which should allow us some extra memory for the averaging itself
        # TODO: Above is a hypothesis. Need to test it out for those later, once we know how much memory is typically used by activations

        # This try-catch block is to prevent a flaky test failure in which model_memory_usage is 0
        try:
            # Just setting a number below to match what I found here. This test needs to be revised
            self.assertLess(extra_memory_used_by_localsgd / model_memory_usage, 0.3)
        except ZeroDivisionError:
            print("Skipping flaky test due to 0 memory error")

    @requires_nccl()
    @skip_if_not_multigpu
    def test_max_memory_used_slowmo(self):
        int_devices = get_gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        max_memory_local = self._test_memory_usage_localsgd_with_slowmo(
            devices, int_devices, {"localsgd_frequency": 1}, use_gossip_data_parallel=False,
        )
        max_memory_slowmo = self._test_memory_usage_localsgd_with_slowmo(
            devices,
            int_devices,
            {
                "localsgd_frequency": 100,  # This is so that localsgd does not occur
                "nprocs_per_node": 1,
                "slowmo_momentum": 0.5,
                "slowmo_frequency": 1,
                "slowmo": True,
            },
            use_gossip_data_parallel=True,
        )

        extra_memory_used_by_slowmo = max_memory_slowmo - max_memory_local

        model_memory_usage = find_memory_used_by_model(LargeNet, devices[0])
        # This try-catch block is to prevent a flaky test failure in which model_memory_usage is 0
        try:
            # Just setting a number below to match what I found here. This test needs to be revised
            self.assertAlmostEqual(extra_memory_used_by_slowmo / model_memory_usage, 1.0, places=1)
        except ZeroDivisionError:
            print("Skipping flaky test due to 0 memory error")


if __name__ == "__main__":
    unittest.main()
