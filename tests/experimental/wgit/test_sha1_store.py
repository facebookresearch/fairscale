# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
import shutil

import pytest
import torch
from torch import nn

from fairscale.experimental.wgit.sha1_store import SHA1_Store

# Get the absolute path of the parent at the beginning before any os.chdir(),
# so that we can proper clean it up at any CWD.
PARENT_DIR = Path("sha1_store_testing").resolve()


@pytest.fixture(scope="function")
def sha1_store(request):
    """A fixture for setup and teardown.

    This only runs once per test function. So don't make this too slow.

    Tests must be written in a way that either all of the tests run
    in the order they appears in this file or a specific test is
    run separately by the user. Either way, the test should work.
    """
    # Attach a teardown function.
    def teardown():
        os.chdir(PARENT_DIR.joinpath("..").resolve())
        shutil.rmtree(PARENT_DIR, ignore_errors=True)

    request.addfinalizer(teardown)

    # Teardown in case last run didn't clean it up.
    teardown()

    # Get an empty sha1 store.
    PARENT_DIR.mkdir()
    sha1_store = SHA1_Store(PARENT_DIR, init=True)

    return sha1_store


def test_sha1_add_file(sha1_store):
    os.chdir(PARENT_DIR)

    # Create random checkpoints
    size_list = [25e5, 27e5, 30e5, 35e5, 40e5]
    chkpts = [
        "checkpoint_1a.pt",
        "checkpoint_1b.pt",
        "checkpoint_1c.pt",
        "checkpoint_2.pt",
        "checkpoint_3.pt",
    ]

    for file, size in zip(chkpts, size_list):
        torch.save(nn.Linear(1, int(size)), file)

    # Add those 5 random files.
    for c in chkpts:
        sha1_store.add(c)

    # Add a fixed data twice.
    module = nn.Linear(100, 100, bias=False)
    module.weight.data = torch.zeros(100, 100)
    zeros_file = "zeros.pt"
    torch.save(module.state_dict(), zeros_file)
    sha1_store.add(zeros_file)
    sha1_store.add(zeros_file)

    # Assert the ref counts are 1,1,1,1,1 and 2
    sha1_store._load_json_dict()
    json_dict = sha1_store._json_dict
    assert json_dict["da3e19590de8f77fcf7a09c888c526b0149863a0"] == 2
    del json_dict["created_on"]
    assert sorted(json_dict.values()) == [1, 1, 1, 1, 1, 2], json_dict


def test_sha1_add_state_dict(sha1_store):
    os.chdir(PARENT_DIR)
    # add once
    for i in range(3):
        sha1_store.add(nn.Linear(10, 10).state_dict())
    # add twice
    for i in range(3):
        sd = nn.Linear(8, 8).state_dict()
        sha1_store.add(sd)
        sha1_store.add(sd)

    sha1_store._load_json_dict()
    json_dict = sha1_store._json_dict
    del json_dict["created_on"]
    assert sorted(json_dict.values()) == [1, 1, 1, 2, 2, 2], json_dict


def test_sha1_add_tensor(sha1_store):
    os.chdir(PARENT_DIR)
    sha1_store.add(torch.Tensor([1.0, 5.5, 3.4]))
    sha1_store._load_json_dict()
    json_dict = sha1_store._json_dict
    assert json_dict["71df4069a03a766eacf9f03eea50968e87eae9f8"] == 1, json_dict
