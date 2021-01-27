# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Test FlattenParamsWrapper
"""

import unittest

import torch

from fairscale.nn import FlattenParamsWrapper
from fairscale.utils.testing import objects_are_equal


class TestFlattenParams(unittest.TestCase):
    def _get_transformer(self, seed=0):
        torch.manual_seed(seed)  # keep everything deterministic
        module = torch.nn.Transformer(
            d_model=32, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=128, dropout=0.1,
        )
        module.register_buffer("dummy_buffer", torch.tensor(1.0))
        return module

    def _get_shared_params_transformer(self, seed=0):
        module = self._get_transformer(seed=seed)
        # share the FFNs
        for enc_layer, dec_layer in zip(module.encoder.layers, module.decoder.layers):
            dec_layer.linear1.weight = enc_layer.linear1.weight
            dec_layer.linear2.weight = enc_layer.linear2.weight
        return module

    def _get_output(self, module):
        torch.manual_seed(1)  # keep everything deterministic
        device = next(module.parameters()).device
        dtype = next(module.parameters()).dtype
        src = torch.rand(20, 8, 32).to(device=device, dtype=dtype)  # T x B x C
        tgt = torch.rand(10, 8, 32).to(device=device, dtype=dtype)  # T x B x C
        return module(src, tgt)

    def _get_pnorm_after_step(self, module):
        optim = torch.optim.SGD(module.parameters(), lr=0.01)
        loss = self._get_output(module).sum()
        loss.backward()
        optim.step()
        return torch.norm(torch.stack([p.detach().norm() for p in module.parameters()]))

    def _test_num_params(self, module):
        ref_num_params = sum(p.numel() for p in module.parameters())

        flat_module = FlattenParamsWrapper(module)
        flat_num_params = sum(p.numel() for p in flat_module.parameters())

        assert ref_num_params == flat_num_params
        assert flat_num_params == flat_module.flat_param.numel()

    def _test_output(self, module):
        ref_output = self._get_output(module)

        flat_module = FlattenParamsWrapper(module)
        flat_output = self._get_output(flat_module)
        assert objects_are_equal(ref_output, flat_output)

    def test_partial_flattening(self):
        module = self._get_transformer()
        num_params = sum(p.numel() for p in module.parameters())

        params_to_flatten = list(module.encoder.layers[1].parameters()) + list(module.decoder.layers[0].parameters())
        num_params_to_flatten = sum(p.numel() for p in params_to_flatten)

        module = FlattenParamsWrapper(module, param_list=params_to_flatten)
        assert module.flat_param.numel() == num_params_to_flatten
        assert sum(p.numel() for p in module.parameters()) == num_params

        # flattened parameters are removed
        assert len(list(module.encoder.layers[1].parameters())) == 0
        assert len(list(module.decoder.layers[0].parameters())) == 0

        # non-flattened parameters remain
        assert len(list(module.encoder.layers[0].parameters())) > 0
        assert len(list(module.decoder.layers[1].parameters())) > 0

        # test that changing the module dtype works properly
        orig_dtype = params_to_flatten[0].dtype
        new_dtype = torch.float32 if orig_dtype == torch.float16 else torch.float16
        assert module.flat_param.dtype == orig_dtype
        assert all(p.dtype == orig_dtype for p in module.encoder.layers[0].parameters())
        module = module.to(dtype=new_dtype)
        assert module.flat_param.dtype == new_dtype
        assert all(p.dtype == new_dtype for p in module.encoder.layers[0].parameters())

    def test_num_params(self):
        module = self._get_transformer()
        self._test_num_params(module)

    def test_shared_params_num_params(self):
        module = self._get_shared_params_transformer()
        self._test_num_params(module)

    def test_output(self):
        module = self._get_transformer()
        self._test_output(module)

    def test_shared_params_output(self):
        module = self._get_shared_params_transformer()
        self._test_output(module)

    def test_shared_params_pnorm_after_step(self):
        # incorrect parameter sharing is likely to cause problems after an
        # optimization step
        module = self._get_shared_params_transformer()
        ref_pnorm_after_step = self._get_pnorm_after_step(module)

        module = self._get_shared_params_transformer()  # recreate
        flat_module = FlattenParamsWrapper(module)
        flat_pnorm_after_step = self._get_pnorm_after_step(flat_module)

        torch.testing.assert_allclose(ref_pnorm_after_step, flat_pnorm_after_step)

    def test_state_dict_equality(self):
        module = self._get_shared_params_transformer()
        ref_state_dict = module.state_dict()

        flat_module = FlattenParamsWrapper(module)
        flat_state_dict = flat_module.state_dict()

        assert objects_are_equal(ref_state_dict, flat_state_dict)

    def test_load_state_dict(self):
        module = self._get_shared_params_transformer()
        ref_state_dict = module.state_dict()
        ref_output = self._get_output(module)

        module = self._get_shared_params_transformer(seed=1234)
        flat_module = FlattenParamsWrapper(module)
        flat_module.load_state_dict(ref_state_dict)
        flat_output = self._get_output(flat_module)

        assert objects_are_equal(ref_output, flat_output)

    def test_flat_state_dict(self):
        flat_module = self._get_shared_params_transformer()
        flat_module = FlattenParamsWrapper(flat_module)
        ref_output = self._get_output(flat_module)

        flat_state_dict = flat_module.flat_state_dict()

        new_module = self._get_shared_params_transformer(seed=1234)
        new_module = FlattenParamsWrapper(new_module)
        new_module.load_state_dict(flat_state_dict)
        new_output = self._get_output(new_module)

        assert objects_are_equal(ref_output, new_output)


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestFlattenParamsCUDA(TestFlattenParams):
    def _get_transformer(self, seed=0):
        module = super()._get_transformer(seed=seed)
        return module.cuda()


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestFlattenParamsCUDAHalf(TestFlattenParams):
    def _get_transformer(self, seed=0):
        module = super()._get_transformer(seed=seed)
        return module.cuda().half()


if __name__ == "__main__":
    unittest.main()
