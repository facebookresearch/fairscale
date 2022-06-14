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

from fairscale.experimental.wgit.sha1_store import SHA1_store


@pytest.fixture
def sha1_configs():
    @dataclass
    class Sha1StorePaths:
        test_dirs = Path("temp_wgit_testing/.wgit")
        test_path = Path.cwd().joinpath(test_dirs)
        metadata_file = test_path.joinpath("checkpoint.pt")
        sha1_ref = test_path.joinpath("sha1_refs.json")
        chkpt_dir = test_path.joinpath("checkpoint")
        checkpoint_1 = test_path.joinpath("checkpoint", "checkpoint_1.pt")
        checkpoint_2 = test_path.joinpath("checkpoint", "checkpoint_2.pt")
        checkpoint_3 = test_path.joinpath("checkpoint", "checkpoint_3.pt")

    return Sha1StorePaths


@pytest.fixture
def sha1_store(sha1_configs):
    sha1_store = SHA1_store(sha1_configs.test_dirs, sha1_configs.metadata_file, sha1_configs.sha1_ref, init=False)
    return sha1_store


def test_setup(sha1_configs):
    # Set up the testing directory
    sha1_configs.test_dirs.mkdir(parents=True, exist_ok=True)  # create test .wgit dir
    sha1_configs.metadata_file.touch()
    sha1_configs.sha1_ref.touch()

    # Create the test checkpoint files
    sha1_configs.chkpt_dir.mkdir(exist_ok=False)
    sha1_configs.checkpoint_1.touch()
    sha1_configs.checkpoint_2.touch()

    # Create random checkpoints
    size_list = [30e5, 35e5, 40e5]
    chkpts = [sha1_configs.checkpoint_1, sha1_configs.checkpoint_2, sha1_configs.checkpoint_3]
    for file, size in zip(chkpts, size_list):
        with open(file, "wb") as f:
            f.write(os.urandom(int(size)))

    sha1_store = SHA1_store(sha1_configs.test_dirs, sha1_configs.metadata_file, sha1_configs.sha1_ref, init=True)
    return sha1_store


def test_sha1_add(sha1_configs, sha1_store):
    # add the file to sha1_store
    sha1_store.add(sha1_configs.checkpoint_1)

    with open(sha1_configs.metadata_file, "r") as file:
        metadata = json.load(file)

    file_sha1 = metadata["SHA1"]["__sha1_full__"]

    # Check metadata file creation
    assert file_sha1 == sha1_store.get_sha1_hash(sha1_configs.checkpoint_1)
    assert metadata["file_path"] == str(sha1_configs.test_path.joinpath(sha1_store.name, file_sha1[:2], file_sha1[2:]))


def test_sha1_refs(sha1_configs, sha1_store):
    # Check reference creation
    with open(sha1_configs.sha1_ref, "r") as file:
        refs_data = json.load(file)

    # get checkpoint1 sha1
    sha1_chkpt1 = sha1_store.get_sha1_hash(sha1_configs.checkpoint_1)
    assert refs_data[sha1_chkpt1]["parent"] == "ROOT"
    assert refs_data[sha1_chkpt1]["child"] == "HEAD"
    assert refs_data[sha1_chkpt1]["ref_count"] == 0

    # add checkpoint 2 and checkpoint 3
    sha1_store.add(sha1_configs.checkpoint_2)
    sha1_store.add(sha1_configs.checkpoint_3)

    # load ref file after Sha1 add
    with open(sha1_configs.sha1_ref, "r") as file:
        refs_data = json.load(file)

    # get checkpoint1 sha1
    sha1_chkpt2 = sha1_store.get_sha1_hash(sha1_configs.checkpoint_2)
    sha1_chkpt3 = sha1_store.get_sha1_hash(sha1_configs.checkpoint_3)

    assert refs_data[sha1_chkpt2]["parent"] == sha1_chkpt1
    assert refs_data[sha1_chkpt2]["child"] == sha1_chkpt3
    assert refs_data[sha1_chkpt2]["ref_count"] == 1


def test_tear_down(sha1_configs):
    # clean up: delete the .wgit directory created during this Test
    # Making sure the current directory is ./temp_wgit_testing before removing test dir
    test_parent_dir = sha1_configs.test_path.parent
    if (test_parent_dir.stem == "temp_wgit_testing") and (sha1_configs.test_path.stem == ".wgit"):
        shutil.rmtree(test_parent_dir)
    else:
        raise Exception("Exception in testing directory tear down!")
