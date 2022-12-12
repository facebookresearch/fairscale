# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
import shutil
import sys

import pytest
import torch
from torch import nn

from fairscale.experimental.wgit.sha1_store import SHA1_Store
from fairscale.fair_dev.testing.testing import objects_are_equal

# Get the absolute path of the parent at the beginning before any os.chdir(),
# so that we can proper clean it up at any CWD.
TESTING_STORE_DIR = Path("sha1_store_testing").resolve()

# Used to filter metadata json keys.
SHA1_KEY_STR_LEN = 40


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
        os.chdir(TESTING_STORE_DIR.joinpath("..").resolve())
        if TESTING_STORE_DIR.exists():
            shutil.rmtree(TESTING_STORE_DIR)

    request.addfinalizer(teardown)

    # Teardown in case last run didn't clean it up.
    teardown()

    # Get an empty sha1 store.
    sha1_store = SHA1_Store(TESTING_STORE_DIR, init=True)

    return sha1_store


@pytest.mark.parametrize("compress", [True, False])
def test_sha1_add_file(sha1_store, compress):
    os.chdir(TESTING_STORE_DIR)

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
        torch.save(nn.Linear(1, int(size)).state_dict(), file)

    # Add those 5 random files.
    for c in chkpts:
        sha1_store.add(c, compress)

    # Add a fixed data twice.
    module = nn.Linear(100, 100, bias=False)
    module.weight.data = torch.zeros(100, 100)
    zeros_file = "zeros.pt"
    torch.save(module.state_dict(), zeros_file)
    sha1_store.add(zeros_file, compress)
    sha1_store.add(zeros_file, compress)

    # Assert the ref counts are 1,1,1,1,1 and 2
    with sha1_store._readonly_json_ctx:
        json_dict = sha1_store._json_dict
    key = "3c06179202606573a4982d91c2829a1a675362b3"
    assert key in json_dict.keys() and json_dict[key]["ref_count"] == 2, json_dict
    json_dict = dict(filter(lambda item: len(item[0]) == SHA1_KEY_STR_LEN, json_dict.items()))
    assert sorted(map(lambda x: x["ref_count"], json_dict.values())) == [1, 1, 1, 1, 1, 2], json_dict


@pytest.mark.parametrize("compress", [True, False])
def test_sha1_add_state_dict(sha1_store, compress):
    os.chdir(TESTING_STORE_DIR)
    # add once
    for i in range(3):
        sha1_store.add(nn.Linear(100, 100).state_dict(), compress)
    # add twice
    for i in range(3):
        sd = nn.Linear(80, 80).state_dict()
        sha1_store.add(sd, compress)
        sha1_store.add(sd, compress)

    with sha1_store._readonly_json_ctx:
        json_dict = sha1_store._json_dict
    json_dict = dict(filter(lambda item: len(item[0]) == SHA1_KEY_STR_LEN, json_dict.items()))
    assert sorted(map(lambda x: x["ref_count"], json_dict.values())) == [1, 1, 1, 2, 2, 2], json_dict


@pytest.mark.parametrize("compress", [True, False])
def test_sha1_add_tensor(sha1_store, compress):
    os.chdir(TESTING_STORE_DIR)
    sha1_store.add(torch.Tensor([1.0, 5.5, 3.4]).repeat(100), compress)
    with sha1_store._readonly_json_ctx:
        json_dict = sha1_store._json_dict
    key = "81cb2a3f823cfb78da8dd390e29e685720974cc7"
    assert key in json_dict.keys() and json_dict[key]["ref_count"] == 1, json_dict


@pytest.mark.parametrize("compress", [True, False])
def test_sha1_get(sha1_store, compress):
    """Testing the get() API: normal and exception cases."""
    if sys.version_info.major == 3 and sys.version_info.minor > 10:
        pytest.skip("pgzip package doesn't work with 3.11's gzip module")

    os.chdir(TESTING_STORE_DIR)

    # Add a file, a state dict and a tensor.
    file = "test_get.pt"
    torch.save(nn.Linear(100, 100).state_dict(), file)
    state_dict = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 20)).state_dict()
    tensor = torch.ones(20, 30)

    # Check that we can get them back.
    file_sha1 = sha1_store.add(file, compress)
    sd = sha1_store.get(file_sha1)
    assert objects_are_equal(sd, torch.load(file))

    sd_sha1 = sha1_store.add(state_dict, compress)
    sd = sha1_store.get(sd_sha1)
    assert objects_are_equal(sd, state_dict)

    tensor_sha1 = sha1_store.add(tensor, compress)
    tensor_got = sha1_store.get(tensor_sha1)
    assert objects_are_equal(tensor_got, tensor)

    # Make sure invalid sha1 cause exceptions.
    with pytest.raises(ValueError):
        sha1_store.get(tensor_sha1[:-1])


@pytest.mark.parametrize("compress", [True, False])
def test_sha1_delete(sha1_store, compress):
    """Testing the delete() API: with ref counting behavior."""
    os.chdir(TESTING_STORE_DIR)

    # Add once and delete, second delete should throw an exception.
    tensor = torch.ones(30, 50)
    sha1 = sha1_store.add(tensor, compress)
    sha1_store.delete(sha1)
    with pytest.raises(ValueError):
        sha1_store.delete(sha1)

    # Add multiple times and delete should match that.
    state_dict = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 20)).state_dict()
    sha1 = sha1_store.add(state_dict, compress)
    for i in range(3):
        new_sha1 = sha1_store.add(state_dict, compress)
        assert sha1 == new_sha1, f"{sha1} vs. {new_sha1}"
    for i in range(4):
        sha1_store.delete(sha1)
    with pytest.raises(ValueError):
        sha1_store.delete(sha1)


@pytest.mark.parametrize("compress", [True, False])
def test_sha1_size_info_and_names(sha1_store, compress):
    """Testing the size_info() and names() APIs."""
    os.chdir(TESTING_STORE_DIR)

    # Add once & check.
    tensor = torch.ones(300, 500)
    sha1 = sha1_store.add(tensor, compress=compress, name="name1")
    orig, dedup, gzip = sha1_store.size_info(sha1)
    assert orig == dedup, "no dedup should happen"
    if not compress:
        assert orig == gzip, "no compression should happen"
    else:
        assert orig > gzip, "compression should be smaller"
    assert (orig, dedup, gzip) == sha1_store.size_info(), "store and entry sizes should match"

    names = sha1_store.names(sha1)
    assert names == {"name1": 1}, names

    # Add second time & check.
    sha1 = sha1_store.add(tensor, compress=compress, name="name2")
    orig2, dedup2, gzip2 = sha1_store.size_info(sha1)
    assert orig2 == orig * 2 == dedup2 * 2, "dedup not correct"
    assert gzip == gzip2, "compression shouldn't change"

    names = sha1_store.names(sha1)
    assert names == {"name1": 1, "name2": 1}, names
