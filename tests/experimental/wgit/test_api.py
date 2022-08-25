# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from pathlib import Path
import random
import shutil

import pytest
import torch
from torch import nn

from fairscale.experimental.wgit.repo import Repo, RepoStatus


@pytest.fixture
def create_test_dir():
    curr_dir = Path.cwd()
    parent_dir = "experimental"
    test_dir = curr_dir.joinpath(parent_dir, "wgit_testing/")

    # creates a testing directory within ./experimental
    try:
        os.makedirs(test_dir)
    except FileExistsError:
        shutil.rmtree(test_dir)
        os.makedirs(test_dir)
    os.chdir(test_dir)

    # create random checkpoints
    size_list = [30e5, 35e5, 40e5, 40e5]
    for i, size in enumerate(size_list):
        sd = {}
        sd["model"] = nn.Linear(1, int(size)).state_dict()
        sd["step"] = 100
        torch.save(sd, f"checkpoint_{i}.pt")
    return test_dir


@pytest.fixture
def repo():
    repo = Repo(Path.cwd(), init=True)
    return repo


def test_setup(create_test_dir):
    assert str(create_test_dir.stem) == "wgit_testing"


def test_api_init(capsys, repo):
    repo = Repo(Path.cwd(), init=True)
    assert Path(".wgit/sha1_store").is_dir()
    assert Path(".wgit/.gitignore").is_file()
    assert Path(".wgit/.git").exists()
    assert Path(".wgit/.gitignore").exists()


@pytest.mark.parametrize("per_tensor", [True, False])
@pytest.mark.parametrize("gzip", [True, False])
def test_api_add(capsys, repo, per_tensor, gzip):
    fnum = random.randint(0, 2)
    chkpt0 = f"checkpoint_{fnum}.pt"
    repo.add(chkpt0, per_tensor=per_tensor, gzip=gzip)
    if per_tensor:
        # TODO (Min): test per_tensor add more.
        return
    sha1_hash = repo._sha1_store._get_sha1_hash(chkpt0)
    metadata_path = repo._rel_file_path(Path(chkpt0))

    with open(os.path.join(".wgit", metadata_path), "r") as f:
        json_data = json.load(f)

    sha1_dir_0 = f"{sha1_hash[:2]}/" + f"{sha1_hash[2:]}"
    # The sha1 are different because add internally use a different pickle method.
    assert json_data["SHA1"] != sha1_hash


def test_api_commit(capsys, repo):
    commit_msg = "epoch_1"
    repo.commit(message=commit_msg)
    with open(".wgit/.git/logs/HEAD") as f:
        line = f.readlines()
    assert line[0].rstrip().split()[-1] == commit_msg


@pytest.mark.parametrize("per_tensor", [True, False])
def test_api_status(capsys, repo, per_tensor):
    # delete the repo and initialize a new one:
    shutil.rmtree(".wgit")
    repo = Repo(Path.cwd(), init=True)

    # check status before any file is added
    out = repo.status()
    assert out == {"": RepoStatus.CLEAN}

    # check status before after a file is added but not committed
    chkpt0 = f"checkpoint_{random.randint(0, 1)}.pt"
    repo.add(chkpt0, per_tensor=per_tensor)
    out = repo.status()
    key_list = list(repo._get_metdata_files().keys())
    assert out == {key_list[0]: RepoStatus.CHANGES_ADDED_NOT_COMMITED}

    # check status after commit
    repo.commit("e1")
    out = repo.status()
    assert out == {key_list[0]: RepoStatus.CLEAN}

    # check status after a new change has been made to the file
    torch.save(nn.Linear(1, int(15e5)).state_dict(), chkpt0)
    out = repo.status()
    assert out == {key_list[0]: RepoStatus.CHANGES_NOT_ADDED}

    # add the new changes made to weigit
    repo.add(chkpt0, per_tensor=per_tensor)
    out = repo.status()
    assert out == {key_list[0]: RepoStatus.CHANGES_ADDED_NOT_COMMITED}

    # check status after a new different file is added to be tracked by weigit
    chkpt3 = "checkpoint_3.pt"
    repo.add(chkpt3, per_tensor=per_tensor)
    key_list = list(repo._get_metdata_files().keys())
    out = repo.status()
    assert out == {
        key_list[0]: RepoStatus.CHANGES_ADDED_NOT_COMMITED,
        key_list[1]: RepoStatus.CHANGES_ADDED_NOT_COMMITED,
    }

    # check status after the new file is commited to be tracked by weigit
    repo.commit("e2")
    out = repo.status()
    assert out == {key_list[0]: RepoStatus.CLEAN, key_list[1]: RepoStatus.CLEAN}


def test_api_log(capsys, repo):
    repo.log("testfile.pt")
    captured = capsys.readouterr()
    assert captured.out == "wgit log of the file: testfile.pt\n"
    assert captured.err == ""


def test_api_checkout(repo):
    try:
        repo.checkout("sha1")
    except NotImplementedError:
        assert True


def teardown_module(module):
    # clean up: delete the .wgit directory created during this Test
    # Making sure the current directory is ./experimental before removing test dir
    if (Path.cwd().parent.name == "experimental") and (Path.cwd().name == "wgit_testing"):
        os.chdir(Path.cwd().parent)
        shutil.rmtree("./wgit_testing/")
    else:
        raise Exception("Exception in testing directory tear down!")
