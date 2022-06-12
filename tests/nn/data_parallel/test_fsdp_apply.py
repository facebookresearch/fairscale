# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
import unittest

from parameterized import parameterized
import pytest
import torch.nn as nn

from fairscale.internal import torch_version

from .test_fsdp import (
    CONFIG_OPTIONS,
    DistributedTest,
    NestedWrappedModule,
    TransformerWithSharedParams,
    rename_test,
    spawn_and_init,
)


@pytest.mark.skipif(torch_version() < (1, 8, 0), reason="pytorch version >= 1.8.0 required")
class TestApply(DistributedTest):
    @parameterized.expand(CONFIG_OPTIONS, name_func=rename_test)
    def test_transformer_weight_init(self, config):
        model_init_fn = functools.partial(model_init_and_apply_custom_weight_init, TransformerWithSharedParams)
        test_fn = functools.partial(self._test_identical_outputs, model_init_fn, config, lr=0.01)
        spawn_and_init(test_fn)

    @parameterized.expand(CONFIG_OPTIONS, name_func=rename_test)
    def test_nested_wrapped_weight_init(self, config):
        model_init_fn = functools.partial(model_init_and_apply_custom_weight_init, NestedWrappedModule)
        test_fn = functools.partial(self._test_identical_outputs, model_init_fn, config, lr=0.01)
        spawn_and_init(test_fn)


def model_init_and_apply_custom_weight_init(model_init_fn, *args, **kwargs):
    model = model_init_fn(*args, **kwargs)
    model.apply(init_bert_params_)
    return model


def init_bert_params_(module):
    """
    Initialize the weights specific to the BERT Model.
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, nn.MultiheadAttention):
        normal_(module.in_proj_weight.data)


if __name__ == "__main__":
    unittest.main()
