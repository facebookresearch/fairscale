# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil

import pytest

from fairscale.experimental.wgit.repo import Repo
from fairscale.experimental.wgit.sha1_store import SHA1_Store


@pytest.fixture
def sha1_configs():
    @dataclass
    class Sha1StorePaths:
        test_dirs = Path("temp_wgit_testing/.wgit")
        test_path = Path.cwd().joinpath(test_dirs)
        sha1_ref = test_path.joinpath("sha1_refs.json")
        chkpt1a_dir = test_path.joinpath("checkpoint_1a")
        chkpt1b_dir = test_path.joinpath("checkpoint_1b")
        chkpt1c_dir = test_path.joinpath("checkpoint_1c")
        checkpoint_1a = test_path.joinpath("checkpoint_1a", "checkpoint_1.pt")
        checkpoint_1b = test_path.joinpath("checkpoint_1b", "checkpoint_1.pt")
        checkpoint_1c = test_path.joinpath("checkpoint_1c", "checkpoint_1.pt")
        checkpoint_2 = test_path.joinpath("checkpoint_1a", "checkpoint_2.pt")
        checkpoint_3 = test_path.joinpath("checkpoint_1a", "checkpoint_3.pt")
        metadata_1 = test_path.joinpath("checkpoint_1.pt")
        metadata_2 = test_path.joinpath("checkpoint_2.pt")
        metadata_3 = test_path.joinpath("checkpoint_3.pt")

    return Sha1StorePaths


@pytest.fixture
def sha1_store(sha1_configs):
    repo = Repo(sha1_configs.test_path.parent, init=False)
    sha1_store = SHA1_Store(sha1_configs.test_dirs, init=False)
    return repo, sha1_store


def test_setup(sha1_configs):
    # Set up the testing directory
    sha1_configs.test_dirs.mkdir(parents=True, exist_ok=True)  # create test .wgit dir

    # Create the test checkpoint files
    sha1_configs.chkpt1a_dir.mkdir(exist_ok=False)
    sha1_configs.chkpt1b_dir.mkdir(exist_ok=False)
    sha1_configs.chkpt1c_dir.mkdir(exist_ok=False)

    # Create random checkpoints
    size_list = [25e5, 27e5, 30e5, 35e5, 40e5]
    chkpts = [
        sha1_configs.checkpoint_1a,
        sha1_configs.checkpoint_1b,
        sha1_configs.checkpoint_1c,
        sha1_configs.checkpoint_2,
        sha1_configs.checkpoint_3,
    ]

    for file, size in zip(chkpts, size_list):
        with open(file, "wb") as f:
            f.write(os.urandom(int(size)))

    repo = Repo(sha1_configs.test_path.parent, init=True)
    sha1_store = SHA1_Store(sha1_configs.test_dirs, init=True)

    return sha1_store


def test_sha1_add(sha1_configs, sha1_store):
    repo, sha1_store = sha1_store

    # Add checkpoint_1: Create the meta_data
    chkpt1 = sha1_configs.checkpoint_1a
    metadata_file, parent_sha1 = repo._process_metadata_file(chkpt1.name)

    sha1_hash = sha1_store.add(sha1_configs.checkpoint_1a, parent_sha1)
    repo._write_metadata(metadata_file, chkpt1, sha1_hash)

    # for checkpoint 1
    metadata_file = sha1_configs.test_path.joinpath(sha1_configs.checkpoint_1a.name)

    with open(metadata_file, "r") as file:
        metadata = json.load(file)
    assert metadata["SHA1"]["__sha1_full__"] == sha1_hash


def test_sha1_refs(sha1_configs, sha1_store):
    repo, sha1_store = sha1_store

    def add_checkpoint(checkpoint):
        metadata_file, parent_sha1 = repo._process_metadata_file(checkpoint.name)
        sha1_hash = sha1_store.add(checkpoint, parent_sha1)
        repo._write_metadata(metadata_file, checkpoint, sha1_hash)
        return sha1_hash

    with open(sha1_configs.sha1_ref, "r") as file:
        refs_data = json.load(file)

    # get checkpoint1 sha1
    sha1_chkpt1a_hash = sha1_store._get_sha1_hash(sha1_configs.checkpoint_1a)
    assert refs_data[sha1_chkpt1a_hash]["parent"] == "ROOT"
    assert refs_data[sha1_chkpt1a_hash]["ref_count"] == 1

    ck1a_sha1_hash = sha1_store._get_sha1_hash(sha1_configs.checkpoint_1a)

    # add checkpoint new version of checkpoint-1
    ck1b_sha1_hash = add_checkpoint(sha1_configs.checkpoint_1b)

    # Add new checkpoints 2 and 3
    ck2_sha1_hash = add_checkpoint(sha1_configs.checkpoint_2)
    ck3_sha1_hash = add_checkpoint(sha1_configs.checkpoint_3)

    # add another version of checkpoint 1
    ck1c_sha1_hash = add_checkpoint(sha1_configs.checkpoint_1c)

    # load ref file after Sha1 add
    with open(sha1_configs.sha1_ref, "r") as file:
        refs_data = json.load(file)

    # Tests for same file versions
    assert refs_data[ck1b_sha1_hash]["parent"] == ck1a_sha1_hash
    assert refs_data[ck1c_sha1_hash]["parent"] == ck1b_sha1_hash
    assert refs_data[ck1b_sha1_hash]["ref_count"] == 1
    assert refs_data[ck1a_sha1_hash]["is_leaf"] is False
    assert refs_data[ck1a_sha1_hash]["is_leaf"] is False
    assert refs_data[ck1b_sha1_hash]["is_leaf"] is False
    assert refs_data[ck1c_sha1_hash]["is_leaf"] is True

    # Tests for new files
    assert refs_data[ck2_sha1_hash]["parent"] == "ROOT"
    assert refs_data[ck2_sha1_hash]["is_leaf"] is True
    assert refs_data[ck3_sha1_hash]["parent"] == "ROOT"
    assert refs_data[ck3_sha1_hash]["is_leaf"] is True


def test_tear_down(sha1_configs):
    # clean up: delete the .wgit directory created during this Test
    # Making sure the current directory is ./temp_wgit_testing before removing test dir
    test_parent_dir = sha1_configs.test_path.parent
    if (test_parent_dir.stem == "temp_wgit_testing") and (sha1_configs.test_path.stem == ".wgit"):
        shutil.rmtree(test_parent_dir)
    else:
        raise Exception("Exception in testing directory tear down!")
