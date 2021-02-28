import torch.nn as nn
from fairscale.nn import FullyShardedDataParallel as FSDP
import unittest


class TestAutoWrap(unittest.TestCase):
    def setUp(self) -> None:
        class TestDistributedGroup:
            def size(self):
                return 1

            @property
            def world_size(self):
                return 1

            def rank(self):
                return 0

        self.process_group = TestDistributedGroup()

    def test_wrap(self):
        with FSDP.enable_wrap(flatten_parameters=False, process_group=self.process_group):
            layer = FSDP.wrap(nn.Linear(5, 5))
        assert isinstance(layer, FSDP)
        assert layer.flatten_parameters is False

    def test_wrap_disabled_outside_context(self):
        layer = FSDP.wrap(nn.Linear(5, 5))
        assert isinstance(layer, nn.Linear)

    def test_wrap_override_defaults(self):
        with FSDP.enable_wrap(flatten_parameters=False, process_group=self.process_group):
            layer = FSDP.wrap(nn.Linear(5, 5), flatten_parameters=True)
        assert isinstance(layer, FSDP)
        assert layer.flatten_parameters

    def test_auto_wrap(self):
        """
        Test to ensure with auto wrap, we wrap child modules correctly based on the bucket size.
        ``nn.Linear(5, 5)`` does not exceed the bucket size, but combined they do.
        """
        with FSDP.enable_wrap(process_group=self.process_group, flatten_parameters=False):
            sequential = nn.Sequential(
                nn.Linear(5, 5), nn.Linear(5, 5), nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5))
            )
            model = FSDP.auto_wrap(sequential, bucket_size=40)
        assert isinstance(model, FSDP)
        assert isinstance(model.module[0], nn.Linear)
        assert isinstance(model.module[1], nn.Linear)
        assert isinstance(model.module[2], FSDP)
        assert isinstance(model.module[2].module[0], nn.Linear)
        assert isinstance(model.module[2].module[1], nn.Linear)
