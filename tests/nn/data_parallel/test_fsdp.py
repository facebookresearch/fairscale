# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import functools
import itertools
from math import inf
import pickle
import sys
from typing import Dict
import unittest
from unittest import mock

from parameterized import parameterized
import torch
from torch import nn

from fairscale.nn.data_parallel import FullyShardedDataParallel
from fairscale.nn.misc.checkpoint_activations import checkpoint_wrapper
from fairscale.utils.testing import (
    DeviceAndTypeCheckModule,
    DummyProcessGroup,
    dist_init,
    get_cycles_per_ms,
    objects_are_equal,
    spawn_for_all_world_sizes,
)

# How to use remote-pdb: https://gist.github.com/sshleifer/9d43351957179c13606e015b072927d4
# All helper functions called by spawn must be either @classmethod, @staticmethod


class DistributedTest(unittest.TestCase):
    def setUp(self):
        major, minor = torch.__version__.split(".")[:2]
        major, minor = int(major), int(minor)
        if major < 1 or (major == 1 and minor < 6):
            raise unittest.SkipTest("Need pytorch version >= 1.6 due to autocast")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available, skipping test")
        if sys.platform == "win32":
            raise unittest.SkipTest("NCCL doesn't support Windows, skipping test")
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("distributed tests require 2+ GPUs, skipping")

    @staticmethod
    def _train_for_several_steps(model, num_steps, autocast, lr=0.01, norm_type=None):
        model_device = next(model.parameters()).device
        # use SGD with momentum instead of Adam, since Adam is scale invariant
        # and this makes it bad for tests
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        for _ in range(num_steps):
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=autocast):
                # Inputs always cuda regardless of move_grads_cpu, or model.device
                input = model.module.get_input(torch.device("cuda"))
                output = model(*input)
                loss = model.module.get_loss(input, output).to(model_device)
            assert loss.dtype == torch.float32
            model.module.run_backward(loss)
            if norm_type is not None:
                clip_norm = 0.3
                if isinstance(model, FullyShardedDataParallel):
                    model.clip_grad_norm_(clip_norm, norm_type)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, norm_type)
            optim.step()
        if hasattr(model, "assert_idle"):
            model.assert_idle()
        return loss.detach()

    @staticmethod
    def get_wrapped_model(group, cuda_first=False, config={}, **model_kwargs) -> FullyShardedDataParallel:
        if cuda_first:
            model = FullyShardedDataParallel(TransformerWithSharedParams(group, **model_kwargs).cuda(), group, **config)
        else:
            model = FullyShardedDataParallel(TransformerWithSharedParams(group, **model_kwargs), group, **config).cuda()
        return model


class TestMixedPrecision(DistributedTest):
    def test_all_fp32(self):
        self._spawn_test_case(
            {"mixed_precision": False},
            False,  # autocast enabled
            torch.float32,  # expected_input_dtype
            torch.float32,  # expected_param_dtype
            torch.float32,  # expected_loss_dtype
            torch.float32,  # expected_reduce_dtype
        )

    def test_mixed_precision(self):
        self._spawn_test_case(
            {"mixed_precision": True},
            False,  # autocast enabled
            torch.float16,  # expected_input_dtype
            torch.float16,  # expected_param_dtype
            torch.float16,  # expected_loss_dtype
            torch.float16,  # expected_reduce_dtype
        )

    def test_mixed_precision_autocast(self):
        """If autocast enabled, loss should be fp32."""
        self._spawn_test_case(
            {"mixed_precision": True},
            True,  # autocast enabled
            torch.float16,  # expected_input_dtype
            torch.float16,  # expected_param_dtype
            torch.float32,  # expected_loss_dtype
            torch.float16,  # expected_reduce_dtype
        )

    def test_mixed_precision_autocast_fp32_compute(self):
        self._spawn_test_case(
            {"mixed_precision": True, "compute_dtype": torch.float32},
            True,  # autocast enabled
            torch.float16,  # expected_input_dtype
            torch.float32,  # expected_param_dtype
            torch.float32,  # expected_loss_dtype
            torch.float32,  # expected_reduce_dtype
        )

    def test_fp32_reduce_scatter(self):
        self._spawn_test_case(
            {"mixed_precision": True, "fp32_reduce_scatter": True},
            False,  # autocast enabled
            torch.float16,  # expected_input_dtype
            torch.float16,  # expected_param_dtype
            torch.float16,  # expected_loss_dtype
            torch.float32,  # expected_reduce_dtype
        )

    def test_fp32_reduce_scatter_autocast(self):
        self._spawn_test_case(
            {"mixed_precision": True, "fp32_reduce_scatter": True},
            True,  # autocast enabled
            torch.float16,  # expected_input_dtype
            torch.float16,  # expected_param_dtype
            torch.float32,  # expected_loss_dtype
            torch.float32,  # expected_reduce_dtype
        )

    def _spawn_test_case(self, cfg, autocast_enabled, in_dtype, p_dtype, loss_dtype, reduce_dtype, world_size=2):
        """Call test_dtypes inside of torch.multiprocessing.spawn"""
        fn = functools.partial(self._test_dtypes, cfg, autocast_enabled, in_dtype, p_dtype, loss_dtype, reduce_dtype)
        spawn_and_init(fn, world_sizes=[world_size])

    @staticmethod
    def _test_dtypes(cfg: Dict, autocast, in_dtype, p_dtype, loss_dtype, reduce_dtype, rank, group):
        # Patch torch.distributed.reduce_scatter to check the dtype of the reduction
        orig_reduce_scatter = torch.distributed.reduce_scatter

        model = DeviceAndTypeCheckModule(
            expected_input_dtype=in_dtype, expected_param_dtype=p_dtype, expected_loss_dtype=loss_dtype,
        )

        def _reduce_scatter(output, input_list, **kwargs):
            for tensor in input_list:
                model._check("reduce_scatter.dtype", tensor.dtype, expected=reduce_dtype)
            return orig_reduce_scatter(output, input_list, **kwargs)

        with mock.patch("torch.distributed.reduce_scatter", new=_reduce_scatter):
            model = FullyShardedDataParallel(model, group, **cfg).cuda()
            device = next(model.parameters()).device
            x = torch.rand(2, 5).to(device)
            with torch.cuda.amp.autocast(enabled=autocast):
                loss = model(x)
            loss.backward()


keys = ["reshard_after_forward", "mixed_precision", "flatten_parameters"]
CONFIG_OPTIONS = [[dict(zip(keys, config))] for config in itertools.product([True, False], repeat=len(keys))]


def rename_test(testcase_func, param_num, param):
    return "%s_%s" % (testcase_func.__name__, parameterized.to_safe_name(str(param.args)),)


class TestComparisonToPyTorchDDP(DistributedTest):
    """
    Compare losses and parameter values after several updates when using
    PyTorch DDP vs. FullyShardedDataParallel.
    """

    @parameterized.expand(CONFIG_OPTIONS, name_func=rename_test)
    def test_transformer_parameterized(self, config):
        # Test every combination of these options:
        spawn_and_init(functools.partial(self._test_identical_outputs, TransformerWithSharedParams, config))

    def test_cpu_offload_and_cpu_grads(self):
        # We don't test the False condition because that requires the optimizer to internally do
        # the device transfer and PyTorch optimizers don't support this.
        config = {"mixed_precision": True, "cpu_offload": True, "move_grads_to_cpu": True}
        test_fn = functools.partial(
            self._test_identical_outputs, TransformerWithSharedParams, config, use_cuda=False, lr=0.01
        )
        spawn_and_init(test_fn)

    def test_cpu_offload_and_cuda_grads_breaks(self):
        # If grads are on gpu, but model and optimizer are on cpu, backward breaks.
        config = {"mixed_precision": True, "cpu_offload": True, "move_grads_to_cpu": False}
        with self.assertRaises(Exception):  # RuntimeError inside spawn
            test_fn = functools.partial(
                self._test_identical_outputs, TransformerWithSharedParams, config, use_cuda=False
            )
            spawn_and_init(test_fn)

    def test_delayed_optim_step(self):
        # We use a model with a long CUDA delay right before the optimizer step.
        # This tests our streams logic, and that we don't start the FP32 -> FP16
        # transfer until after the optimization step completes.
        config = {"mixed_precision": True}
        model_fn = functools.partial(NestedWrappedModuleWithDelay, delay_after_loss_ms=250)
        test_fn = functools.partial(self._test_identical_outputs, model_fn, config)
        spawn_and_init(test_fn)

    def test_delayed_reduce_scatter(self):
        # We insert a delay in the torch.distributed.reduce_scatter op, so that
        # the post_backward_stream takes much longer than the backward pass.
        # This tests that we properly block at the end of the backward pass for
        # the reductions to finish.
        config = {"mixed_precision": True}
        model_fn = functools.partial(NestedWrappedModuleWithDelay, delay_before_reduction_ms=250)
        test_fn = functools.partial(self._test_identical_outputs, model_fn, config)
        spawn_and_init(test_fn)

    @parameterized.expand([[{"checkpoint_act": False}], [{"checkpoint_act": True}]], name_func=rename_test)
    def test_mixture_of_experts(self, moe_config):
        fsdp_config = {"mixed_precision": True}
        test_fn = functools.partial(
            self._test_identical_outputs,
            functools.partial(MixtureOfExperts, **moe_config),
            fsdp_config,
            # MixtureOfExperts implements custom reduce logic, so the reference
            # behavior should use that logic instead of PyTorch DDP.
            ref_ddp_fn=self._dummy_ddp_fn,
            norm_type=None,
        )
        spawn_and_init(test_fn)

    def test_mixture_of_experts_grad_clip_breaks(self):
        config = {"mixed_precision": True}
        test_fn = functools.partial(
            self._test_identical_outputs, MixtureOfExperts, config, ref_ddp_fn=self._dummy_ddp_fn, norm_type=2,
        )
        with self.assertRaises(Exception):
            spawn_and_init(test_fn)

    @classmethod
    def _dummy_ddp_fn(self, model, group):
        return DummyDDP(model)

    @classmethod
    def _test_identical_outputs(
        cls, model_init_fn, config, rank, group, num_steps=2, use_cuda=True, lr=0.01, ref_ddp_fn=None, norm_type=2,
    ):
        if config["mixed_precision"]:
            autocast = True
            # Force the compute dtype to be torch.float32 so that we get
            # identical results as PyTorch DDP when using autocast. Note that
            # this will cause the all-gather to happen in FP32, which is slower
            # than necessary in most cases.
            config["compute_dtype"] = torch.float32
        else:
            autocast = False

        # Establish reference behavior with PyTorch DDP (+ optionally autocast).
        model = model_init_fn(group=group, wrapper_config=None).cuda()
        if ref_ddp_fn is None:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[rank], output_device=rank, process_group=group
            )
        else:
            model = ref_ddp_fn(model, group)
        ref_loss = cls._train_for_several_steps(model, num_steps, autocast, lr=lr, norm_type=norm_type)
        ref_state_dict = model.module.state_dict()

        # Confirm we get the same behavior using FullyShardedDataParallel.
        model = FullyShardedDataParallel(model_init_fn(group=group, wrapper_config=config), group, **config)
        if use_cuda:
            model = model.cuda()
        else:
            assert next(model.parameters()).device == torch.device("cpu")
        shard_loss = cls._train_for_several_steps(model, num_steps, autocast, lr=lr, norm_type=norm_type)
        shard_state_dict = model.state_dict()

        try:
            torch.testing.assert_allclose(ref_loss, shard_loss)
            assert objects_are_equal(ref_state_dict, shard_state_dict, raise_exception=True)
        except (AssertionError, RuntimeError) as e:
            raise Exception(f"FullyShardedDataParallel didn't match PyTorch DDP using config: {config}\n\n {e}")

    @parameterized.expand([[1], [inf]], name_func=rename_test)
    def test_clip_norm_transformer(self, norm_type):
        config = {"mixed_precision": True}
        test_fn = functools.partial(
            self._test_identical_outputs, TransformerWithSharedParams, config, norm_type=norm_type,
        )
        spawn_and_init(test_fn)


class TestParamInit(DistributedTest):
    def test_param_change_after_init(self):
        test_fn = functools.partial(self._test_param_change_after_init, config={"mixed_precision": True})
        spawn_and_init(test_fn)

    @classmethod
    def _test_param_change_after_init(self, rank, group, config):
        # Establish reference behavior.
        model = self.get_wrapped_model(group, cuda_first=False, config=config)
        model.eval()  # no dropout for this test
        input = model.module.get_input(torch.device("cuda"))
        ref_output = model(*input)

        # Change the weights in place.
        model = self.get_wrapped_model(group, cuda_first=False, config=config)
        model.eval()  # no dropout for this test
        first_param = next(model.parameters())
        nn.init.normal_(first_param.data)
        new_output = model(*input)

        assert not objects_are_equal(ref_output, new_output), "new_output did not reflect change to param after init"


class TestSerialization(DistributedTest):
    @parameterized.expand([[False, False], [True, False], [True, True]], name_func=rename_test)
    def test_pickle(self, mixed_precision, cpu_offload):
        """Ensure that wrapped modules can be pickled/unpickled."""
        config = {"mixed_precision": mixed_precision, "cpu_offload": cpu_offload}
        test_fn = functools.partial(self._test_pickle, config=config)
        spawn_and_init(test_fn, world_sizes=[2])

    @parameterized.expand([[False, False], [True, False], [True, True]], name_func=rename_test)
    def test_multiprocessing(self, mixed_precision, cpu_offload):
        """Ensure that wrapped modules can be sent via multiprocessing."""
        config = {"mixed_precision": mixed_precision, "cpu_offload": cpu_offload}
        test_fn = functools.partial(self._test_multiprocessing, config=config)
        spawn_and_init(test_fn, world_sizes=[2])

    @classmethod
    def _test_pickle(self, rank, group, config):
        model = self._get_model(group, config)
        model = pickle.loads(pickle.dumps(model))
        if not config["cpu_offload"]:
            model = model.cuda()
        self._one_step(model, group)

    @classmethod
    def _test_multiprocessing(self, rank, group, config):
        mp = torch.multiprocessing.Pool(1)
        dummy_group = DummyProcessGroup(rank=group.rank(), size=group.size())
        model = mp.apply(self._get_model, (dummy_group, config))
        if not config["cpu_offload"]:
            model = model.cuda()
        self._one_step(model, group)

    @classmethod
    def _get_model(self, group, config):
        with torch.no_grad():  # required for multiprocessing
            model = NestedWrappedModule(group, wrapper_config=config)
            return FullyShardedDataParallel(model, group, **config)

    @classmethod
    def _one_step(self, model, group):
        # reset the process group (required after unpickling)
        for m in model.modules():
            if isinstance(m, FullyShardedDataParallel):
                m.process_group = group
        optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        input = model.module.get_input(torch.device("cuda"))
        output = model(*input)
        loss = model.module.get_loss(input, output)
        model.module.run_backward(loss)
        optim.step()


class TestLocalStateDict(DistributedTest):
    @parameterized.expand([[True, True], [False, False]], name_func=rename_test)
    def test_load_local_state_dict(self, flatten_params, mixed_precision):
        test_fn = functools.partial(
            self._load_local_and_train, {"flatten_parameters": flatten_params, "mixed_precision": mixed_precision}
        )
        spawn_and_init(test_fn)

    @classmethod
    def _load_local_and_train(self, config, rank, group, d_model=16, d_vocab=23):
        """Check that local_state_dict can be saved and loaded for a given worker, and that training updates it"""
        model = self.get_wrapped_model(group, cuda_first=False, config=config, d_vocab=d_vocab, d_model=d_model)
        state_1 = model.local_state_dict()
        state_before_training = {k: v.cpu().clone() for k, v in state_1.items()}
        assert len(state_1) > 0
        model.load_local_state_dict(state_1)
        weight_key = "flat_param" if model.flatten_parameters else "embed_tokens.weight"

        state_1_weight = state_1[weight_key]
        assert state_1_weight.dtype == torch.float32, f"got dtype {state_1_weight.dtype} expected torch.float32"
        if not model.flatten_parameters:
            # The weight will be sharded since we access module.state_dict directly
            state_1_module_weight = model.module.state_dict()[weight_key]
            torch.testing.assert_allclose(state_1_weight, state_1_module_weight)
            torch.testing.assert_allclose(state_1_weight, model.module.embed_tokens.weight)
        self._train_for_several_steps(model, 1, model.mixed_precision)

        state_2 = model.local_state_dict()
        state_after_training = {k: v.cpu().clone() for k, v in state_2.items()}
        model.load_local_state_dict(state_2)

        assert state_1.keys() == state_2.keys()

        # Assert that parameters were updated since before training
        unchanged = []
        unwrapped_model = model.module.module if config["flatten_parameters"] else model.module
        buffers = {name for name, _ in unwrapped_model.named_buffers()}
        for k in state_1:
            if (state_before_training[k] == state_after_training[k]).all() and (k not in buffers):
                unchanged.append(k)
        if unchanged:
            raise AssertionError(f"params {unchanged} not changed after training")


class TestSaveLoadStateDict(DistributedTest):
    @parameterized.expand([[False], [True]], name_func=rename_test)
    def test_calling_state_dict_twice_mixed_precision(self, mixed_precision):
        test_fn = functools.partial(
            self._test_calling_state_dict_twice, {"flatten_parameters": False, "mixed_precision": mixed_precision}
        )
        spawn_and_init(test_fn)

    @classmethod
    def _test_calling_state_dict_twice(self, config, rank, group, **model_kwargs):
        ddp_model = self.get_wrapped_model(group, cuda_first=False, config=config, **model_kwargs)
        autocast = ddp_model.mixed_precision
        self._train_for_several_steps(ddp_model, 1, autocast)
        ddp_model.state_dict()
        ddp_model.state_dict()  # second call

    @parameterized.expand([[False], [True]], name_func=rename_test)
    def test_state_dict_after_forward_mixed_precision(self, mixed_precision):
        test_fn = functools.partial(
            self._test_module_state_dict, {"flatten_parameters": False, "mixed_precision": mixed_precision}
        )
        spawn_and_init(test_fn)

    @parameterized.expand([[False], [True]], name_func=rename_test)
    def test_state_dict_before_forward(self, mixed_precision):
        test_fn = functools.partial(
            self._test_state_dict_before_forward, {"flatten_parameters": False, "mixed_precision": mixed_precision}
        )
        spawn_and_init(test_fn)

    @classmethod
    def _test_state_dict_before_forward(cls, config, rank, group):
        ddp_model = cls.get_wrapped_model(group, cuda_first=False, config=config)
        sd = ddp_model.state_dict()
        expected_dtype = torch.float16 if ddp_model.mixed_precision else torch.float32
        wt = sd["embed_tokens.weight"]
        assert wt.dtype == expected_dtype, f"got dtype {wt.dtype} expected {expected_dtype}"
        cls._train_for_several_steps(ddp_model, 1, ddp_model.mixed_precision)

    @classmethod
    def _test_module_state_dict(cls, config, rank, group):
        ddp_model = cls.get_wrapped_model(group, cuda_first=False, config=config)
        autocast = ddp_model.mixed_precision
        cls._train_for_several_steps(ddp_model, 2, autocast)
        state_1 = ddp_model.state_dict()
        # You must make a new FullyShardedDataParallel instance to use module.load_state_dict
        unwrapped_model = TransformerWithSharedParams(group)
        unwrapped_model.load_state_dict(state_1)
        new_ddp_model = FullyShardedDataParallel(unwrapped_model, group, **config).cuda()
        cls._train_for_several_steps(new_ddp_model, 2, autocast)
        try:
            ddp_model.load_state_dict(new_ddp_model.state_dict())
            assert False, "ddp_model.load_state_dict(new_ddp_model.state_dict()) succeeded"
        except Exception:
            pass


class TestHooks(DistributedTest):
    # Feel free to modify these tests as the implementation changes.
    # They aspire to make sure that backward hooks are registered and used

    @parameterized.expand([[True], [False]])
    def test_output_backward_hooks(self, cuda_first):
        fn = functools.partial(self._test_output_backward_hooks, cuda_first=cuda_first)
        spawn_and_init(fn)

    def test_backward_hooks_after_save(self):
        fn = functools.partial(self._test_backward_hooks_after_save, cuda_first=False)
        spawn_and_init(fn)

    @classmethod
    def _test_backward_hooks_after_save(self, rank, group, cuda_first=False):
        model = self.get_wrapped_model(group, cuda_first=cuda_first)
        self._train_for_several_steps(model, 2, model.mixed_precision)
        state_1 = model.local_state_dict()
        model.load_local_state_dict(state_1)
        self._test_output_backward_hooks(rank, group, cuda_first=cuda_first, model=model)

    @classmethod
    def _test_output_backward_hooks(self, rank, group, cuda_first=False, model=None):
        if model is None:
            model = self.get_wrapped_model(group, cuda_first=cuda_first)
        optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optim.zero_grad()
        # Inputs always cuda regardless of move_grads_cpu, or model.device
        input = model.module.get_input(torch.device("cuda"))
        output = model(*input)
        assert len(output._backward_hooks) == 1  # this is pre-bwd hook
        loss = model.module.get_loss(input, output).cuda()
        loss.backward()
        assert len(output._backward_hooks) == 1  # It doesn't get removed
        optim.step()
        assert len(output._backward_hooks) == 1

    @parameterized.expand([[True], [False]])
    def test_register_functions_called(self, cuda_first):
        fn = functools.partial(self._test_register_functions_called, cuda_first=cuda_first)
        spawn_and_init(fn)

    @classmethod
    def _test_register_functions_called(self, rank, group, cuda_first=False):
        """Tests that _register_{pre|post}_backward_hooks called during forward."""
        model = self.get_wrapped_model(group, cuda_first=cuda_first)
        input = model.module.get_input(torch.device("cuda"))
        model._register_post_backward_hooks = mock.MagicMock(return_value=None)
        model._register_pre_backward_hooks = mock.MagicMock(return_value=None)
        assert not model._register_post_backward_hooks.called
        assert not model._register_pre_backward_hooks.called
        model(*input)
        assert model._register_post_backward_hooks.called
        assert model._register_pre_backward_hooks.called


class TestNoGrad(DistributedTest):
    @parameterized.expand(CONFIG_OPTIONS, name_func=rename_test)
    def test_transformer_parameterized(self, config):
        test_fn = functools.partial(self._test_transformer, config=config)
        spawn_and_init(test_fn)

    @classmethod
    def _test_transformer(self, rank, group, config):
        autocast = config["mixed_precision"]

        # Train model for a step
        model = self.get_wrapped_model(group, cuda_first=False, config=config)
        self._train_for_several_steps(model, 1, autocast)

        model.eval()  # no dropout for this test

        # Eval in standard mode (i.e., without no_grad)
        input = model.module.get_input(torch.device("cuda"))
        ref_output = model(*input)

        # Eval with no_grad and compare
        with torch.no_grad():
            no_grad_output = model(*input)

        assert objects_are_equal(ref_output, no_grad_output), "no_grad_output did not match ref_output"


class TestNoSync(DistributedTest):
    def test_transformer(self):
        fn = functools.partial(self._test_transformer, config={})
        spawn_and_init(fn)

    def test_transformer_no_flat_params(self):
        config = {"flatten_parameters": False}
        fn = functools.partial(self._test_transformer, config=config)
        spawn_and_init(fn)

    def test_nested_wrapper(self):
        fn = functools.partial(self._test_nested_wrapper, config={})
        spawn_and_init(fn)

    def test_no_sync_before_first_forward(self):
        group = DummyProcessGroup(rank=0, size=1)
        model = self.get_wrapped_model(group, config={})
        batch = model.module.get_input(torch.device("cuda"))
        with model.no_sync():
            output = model(*batch)
            loss = model.module.get_loss(batch, output)
            loss.backward()
        output = model(*batch)
        loss = model.module.get_loss(batch, output)
        loss.backward()

    @classmethod
    def _test_transformer(self, rank, group, config):
        model = self.get_wrapped_model(group, config=config)
        model.eval()  # turn off dropout for the test
        self._test_no_sync(model, batch_dim=1)

    @classmethod
    def _test_nested_wrapper(self, rank, group, config):
        model = NestedWrappedModule(group, config)
        model = FullyShardedDataParallel(model, group, **config).cuda()
        self._test_no_sync(model, batch_dim=0)

    @classmethod
    def _test_no_sync(self, model, batch_dim):
        # Generate two input batches. We'll test that we get the same grads if
        # we train on them sequentially while accumulating grads (with no_sync)
        # vs. concatenating the batches and training in one go.
        batch1 = model.module.get_input(torch.device("cuda"))
        assert isinstance(batch1, tuple)
        batch2 = tuple(
            # This randomly permutes the values in a multi-dim tensor.
            x.view(-1)[torch.randperm(x.numel())].view_as(x)
            for x in batch1
        )
        for x, y in zip(batch1, batch2):
            assert not torch.all(x == y)

        # Concat the batches along batch dimension.
        concat_batch = tuple(torch.cat((x, y), dim=batch_dim) for (x, y) in zip(batch1, batch2))

        # Establish reference behavior on the concat batch.
        model.zero_grad()
        output = model(*concat_batch)
        ref_loss = model.module.get_loss(concat_batch, output)
        ref_loss.backward()
        ref_grads = [p.grad.detach().clone() for p in model.parameters()]

        # Test that we get the same results by accumulating grads.
        model.zero_grad()
        with model.no_sync():  # accumulate gradients from the first batch
            output = model(*batch1)
            loss1 = model.module.get_loss(batch1, output)
            loss1.backward()
        output = model(*batch2)
        loss2 = model.module.get_loss(batch2, output)
        loss2.backward()
        accumulated_loss = loss1 + loss2
        accumulated_grads = [p.grad.detach().clone() for p in model.parameters()]

        torch.testing.assert_allclose(ref_loss, accumulated_loss)
        assert objects_are_equal(ref_grads, accumulated_grads, raise_exception=True)


class TransformerWithSharedParams(nn.Module):
    def __init__(self, group, *unused_args, d_vocab=23, d_model=16, **unused_kwargs):
        super().__init__()
        self.rank = group.rank()
        self.world_size = group.size()
        torch.manual_seed(0)  # keep everything deterministic
        assert d_vocab >= 12  # we use torch.arange(12) as input
        self.embed_tokens = nn.Embedding(d_vocab, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=8, dropout=0.1,
        )
        self.output_proj = nn.Linear(d_model, d_vocab)
        # share the embedding and output projection weights
        self.output_proj.weight = self.embed_tokens.weight
        self.register_buffer("vocab_bias", self.embed_tokens.weight.new_ones((d_model,)))
        self.register_buffer("long_buffer", torch.zeros_like(self.vocab_bias, dtype=torch.long))

    def get_input(self, device):
        torch.manual_seed(1 + self.rank)  # keep everything deterministic
        src = torch.arange(12, device=device).view(6, 2)  # T x B
        tgt = torch.arange(8, device=device).view(4, 2)  # T x B
        return (src, tgt)

    def forward(self, src_ids, tgt_ids):
        src = self.embed_tokens(src_ids)
        src = src + self.vocab_bias + self.long_buffer.type_as(src)
        tgt = self.embed_tokens(tgt_ids)
        x = self.transformer(src, tgt)
        return self.output_proj(x)

    def get_loss(self, input, output):
        _, tgt = input
        return nn.functional.cross_entropy(output.view(-1, output.size(-1)), tgt.view(-1), reduction="sum")

    def run_backward(self, loss):
        loss.backward()


class NestedWrappedModule(nn.Module):
    def __init__(self, group, wrapper_config):
        super().__init__()
        self.rank = group.rank()
        self.world_size = group.size()
        self.wrapper_config = wrapper_config

        def _maybe_wrap(layer):
            if wrapper_config is not None:
                return FullyShardedDataParallel(layer, group, **wrapper_config)
            return layer

        torch.manual_seed(0)  # keep everything deterministic
        self.module = nn.Sequential(
            nn.Linear(8, 4), _maybe_wrap(nn.Linear(4, 16)), _maybe_wrap(nn.Linear(16, 4)), nn.Linear(4, 8),
        )

    def get_input(self, device):
        torch.manual_seed(1 + self.rank)  # keep everything deterministic
        return (torch.rand(4, 8, device=device),)

    def forward(self, x):
        return self.module(x)

    def get_loss(self, input, output):
        loss = output.sum()
        return loss

    def run_backward(self, loss):
        loss.backward()


class DummyDDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class MixtureOfExperts(NestedWrappedModule):
    def __init__(self, group, wrapper_config, checkpoint_act=False):
        super().__init__(group, wrapper_config)
        self.group = group

        # "expert" params are different on each rank
        torch.manual_seed(42 + group.rank())
        expert = nn.Linear(16, 4)
        for p in expert.parameters():
            p.expert = True

        # everything else is shared
        torch.manual_seed(0)
        shared = nn.Linear(4, 16)

        if checkpoint_act:
            expert = checkpoint_wrapper(expert)
            shared = checkpoint_wrapper(shared)

        if wrapper_config is not None:
            # we create a process group of size 1 for the expert params
            expert_group = torch.distributed.new_group([group.rank()])
            expert = FullyShardedDataParallel(expert, expert_group, **wrapper_config)

            shared = FullyShardedDataParallel(shared, group, **wrapper_config)

        self.module = nn.Sequential(nn.Linear(8, 4), shared, expert, nn.Linear(4, 8))

    def run_backward(self, loss):
        loss.backward()

        # manually reduce gradients if not wrapped in FullyShardedDataParallel
        if self.wrapper_config is None:
            with torch.no_grad():
                for p in self.parameters():
                    if hasattr(p, "expert"):
                        continue  # these params don't need grad reduction
                    p.grad.data.div_(self.world_size)
                    torch.distributed.all_reduce(p.grad.data, group=self.group)


class ModuleWithDelay(nn.Module):
    def __init__(self, module, delay_after_loss_ms=0, delay_before_reduction_ms=0):
        super().__init__()
        self.delay_after_loss_ms = delay_after_loss_ms
        self.delay_before_reduction_ms = delay_before_reduction_ms
        self.module = module

    def get_input(self, device):
        return self.module.get_input(device)

    def forward(self, x):
        return self.module(x)

    def get_loss(self, input, output):
        loss = self.module.get_loss(input, output)
        if self.delay_after_loss_ms > 0:
            torch.cuda._sleep(int(self.delay_after_loss_ms * get_cycles_per_ms()))
        return loss

    def run_backward(self, loss):
        orig_reduce_scatter = torch.distributed.reduce_scatter

        def _delayed_reduce_scatter(*args, **kwargs):
            if self.delay_before_reduction_ms > 0:
                torch.cuda._sleep(int(self.delay_before_reduction_ms * get_cycles_per_ms()))
            return orig_reduce_scatter(*args, **kwargs)

        with mock.patch("torch.distributed.reduce_scatter", _delayed_reduce_scatter):
            self.module.run_backward(loss)


class NestedWrappedModuleWithDelay(ModuleWithDelay):
    def __init__(self, group, wrapper_config, **kwargs):
        super().__init__(NestedWrappedModule(group, wrapper_config), **kwargs)


def spawn_and_init(fn, args=None, **spawn_kwargs):
    if args is None:
        args = ()

    run_fn = functools.partial(init_and_run, fn, args)
    spawn_for_all_world_sizes(run_fn, **spawn_kwargs)


def init_and_run(fn, args, rank, world_size, filename, filename_rpc):
    dist_init(rank, world_size, filename, filename_rpc)
    group = torch.distributed.new_group()
    fn(rank, group, *args)


if __name__ == "__main__":
    unittest.main()
