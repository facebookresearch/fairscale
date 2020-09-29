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

from queue import Queue
import threading

import pytest
import torch
from torch import nn

from fairscale.nn.pipe.checkpoint import enable_checkpointing, enable_recomputing
from fairscale.nn.pipe.microbatch import Batch
from fairscale.nn.pipe.skip import pop, skippable, stash
from fairscale.nn.pipe.skip.layout import SkipLayout
from fairscale.nn.pipe.skip.tracker import SkipTracker, SkipTrackerThroughPotals, current_skip_tracker


def test_default_skip_tracker():
    q = Queue()

    def f():
        q.put(current_skip_tracker())

    t = threading.Thread(target=f)
    t.start()
    t.join()

    skip_tracker = q.get()

    assert type(skip_tracker) is SkipTracker
    assert type(skip_tracker) is not SkipTrackerThroughPotals


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_default_skip_tracker_by_data_parallel():
    @skippable(stash=["foo"])
    class Stash(nn.Module):
        def forward(self, input):
            yield stash("foo", input)
            return input * 2

    @skippable(pop=["foo"])
    class Pop(nn.Module):
        def forward(self, input):
            foo = yield pop("foo")
            return foo

    model = nn.Sequential(Stash(), Pop())
    model = nn.DataParallel(model, device_ids=[0, 0], output_device=0)

    input = torch.rand(10, device=0)
    output = model(input)

    assert torch.allclose(output, input)


def test_reuse_portal():
    skip_layout = SkipLayout(num_partitions=2, skip_routes={(None, "test"): (0, 1)})
    skip_tracker = SkipTrackerThroughPotals(skip_layout, 0)

    batch = Batch(torch.tensor([1.0]), 0)
    a = torch.tensor([2.0])
    b = torch.tensor([2.0])

    skip_tracker.save(batch, None, "test", a)
    portal = skip_tracker.portals[(None, "test")]

    skip_tracker.save(batch, None, "test", b)
    assert portal is skip_tracker.portals[(None, "test")]


def test_no_copy_no_portal():
    skip_layout = SkipLayout(num_partitions=2, skip_routes={(None, "copy"): (0, 1), (None, "not_copy"): (0, 0)})
    skip_tracker = SkipTrackerThroughPotals(skip_layout, 0)

    batch = Batch(torch.tensor([1.0]), 0)
    a = torch.tensor([2.0])
    b = torch.tensor([2.0])

    skip_tracker.save(batch, None, "copy", a)
    skip_tracker.save(batch, None, "not_copy", b)

    assert (None, "copy") in skip_tracker.portals
    assert (None, "copy") not in skip_tracker.tensors
    assert (None, "not_copy") in skip_tracker.tensors
    assert (None, "not_copy") not in skip_tracker.portals


def test_tensor_life_without_checkpointing():
    skip_layout = SkipLayout(num_partitions=2, skip_routes={(None, "test"): (0, 1)})
    skip_tracker = SkipTrackerThroughPotals(skip_layout, 0)

    batch = Batch(torch.tensor([1.0]), 0)
    tensor = torch.tensor([2.0])

    skip_tracker.save(batch, None, "test", tensor)
    assert skip_tracker.portals[(None, "test")].tensor_life == 1

    skip_tracker.load(batch, None, "test")
    assert skip_tracker.portals[(None, "test")].tensor_life == 0


def test_tensor_life_with_checkpointing():
    skip_layout = SkipLayout(num_partitions=2, skip_routes={(None, "test"): (0, 1)})
    skip_tracker = SkipTrackerThroughPotals(skip_layout, 0)

    batch = Batch(torch.tensor([1.0]), 0)
    tensor = torch.tensor([2.0])

    with enable_checkpointing():
        skip_tracker.save(batch, None, "test", tensor)
    assert skip_tracker.portals[(None, "test")].tensor_life == 2

    with enable_checkpointing():
        skip_tracker.load(batch, None, "test")
    assert skip_tracker.portals[(None, "test")].tensor_life == 1

    with enable_recomputing():
        skip_tracker.load(batch, None, "test")
    assert skip_tracker.portals[(None, "test")].tensor_life == 0

    with enable_recomputing():
        skip_tracker.save(batch, None, "test", tensor)
    assert skip_tracker.portals[(None, "test")].tensor_life == 0
