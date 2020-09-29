# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2019 Kakao Brain
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from fairscale.nn.pipe.dependency import fork, join
from fairscale.nn.pipe.skip.portal import Portal
from fairscale.nn.pipe.stream import default_stream


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_copy_returns_on_next_device():
    portal = Portal(torch.rand(1), tensor_life=1, index=0)

    prev_stream = default_stream(torch.device("cpu"))
    next_stream = default_stream(torch.device("cuda"))

    phony = torch.zeros(0, requires_grad=True)
    assert phony.device.type == "cpu"

    phony = portal.copy(prev_stream, next_stream, phony)
    assert phony.device.type == "cuda"


def test_blue_orange():
    tensor1 = torch.rand(1, requires_grad=True)
    tensor2 = torch.rand(1, requires_grad=True)

    # Same with: output = tensor1*2 + tensor2
    #
    #                +----------------------+
    #                |                      |
    # tensor2 -- PortalBlue -+      +- PortalOrange -+
    #                        |      |                |
    # tensor1 ------------ Join -- Fork --- Mul --- Add -- output
    #
    main = tensor1
    portal = Portal(tensor2, tensor_life=2, index=0)
    phony = portal.blue()
    main = join(main, phony)
    main, phony = fork(main)
    sub = portal.orange(phony)
    output = main * 2 + sub

    output.backward()

    assert torch.allclose(tensor1.grad, torch.tensor([2.0]))
    assert torch.allclose(tensor2.grad, torch.tensor([1.0]))


def test_blue_orange_not_requires_grad():
    tensor1 = torch.rand(1, requires_grad=True)
    tensor2 = torch.rand(1)

    # Same with: output = tensor1*2 + tensor2
    #
    #                +----------------------+
    #                |                      |
    # tensor2 -- PortalBlue -+      +- PortalOrange -+
    #                        |      |                |
    # tensor1 ------------ Join -- Fork --- Mul --- Add -- output
    #
    main = tensor1
    portal = Portal(tensor2, tensor_life=2, index=0)
    phony = portal.blue()
    main = join(main, phony)
    main, phony = fork(main)
    sub = portal.orange(phony)
    output = main * 2 + sub

    output.backward()

    assert torch.allclose(tensor1.grad, torch.tensor([2.0]))
    assert tensor2.grad is None


def test_use_grad():
    tensor = torch.rand(1, requires_grad=True)
    portal = Portal(tensor, tensor_life=1, index=0)

    portal.put_grad(tensor)
    assert portal.use_grad() is tensor

    # Gradient in a portal is ephemeral.
    with pytest.raises(RuntimeError):
        portal.use_grad()


class TestTensorLife:
    @pytest.fixture
    def new_portal(self):
        portal = None

        def new_portal(tensor_life):
            nonlocal portal
            tensor = torch.rand(1, requires_grad=True)
            portal = Portal(tensor, tensor_life, 0)
            return portal, tensor

        yield new_portal

        # A test using this fixture must exhaust the tensor in the portal.
        with pytest.raises(RuntimeError):
            portal.check_tensor_life()
        assert portal.tensor is None

    def test_tensor_life_0(self, new_portal):
        portal, tensor = new_portal(0)
        assert portal.tensor is None

    def test_tensor_life_1(self, new_portal):
        portal, tensor = new_portal(1)
        assert portal.tensor is tensor

        portal.blue()

    def test_tensor_life_2(self, new_portal):
        portal, tensor = new_portal(2)
        assert portal.tensor is tensor

        phony = portal.blue()
        assert portal.orange(phony).data_ptr() == tensor.data_ptr()

    def test_tensor_life_3(self, new_portal):
        portal, tensor = new_portal(3)
        assert portal.tensor is tensor

        phony = portal.blue()
        assert portal.orange(phony).data_ptr() == tensor.data_ptr()
        assert portal.orange(phony).data_ptr() == tensor.data_ptr()

    def test_tensor_life_4(self, new_portal):
        portal, tensor = new_portal(4)
        assert portal.tensor is tensor

        phony = portal.blue()
        assert portal.orange(phony).data_ptr() == tensor.data_ptr()
        assert portal.orange(phony).data_ptr() == tensor.data_ptr()
        portal.blue()

    def test_tensor_life_3_plus_1(self, new_portal):
        portal, tensor = new_portal(3)
        assert portal.tensor is tensor

        phony = portal.blue()
        assert portal.orange(phony).data_ptr() == tensor.data_ptr()
        assert portal.orange(phony).data_ptr() == tensor.data_ptr()

        another_tensor = torch.rand(1, requires_grad=True)
        portal.put_tensor(another_tensor, tensor_life=1)
        portal.blue()
