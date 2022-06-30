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

from fairscale.experimental.wgit import repo as api


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
        with open(f"checkpoint_{i}.pt", "wb") as f:
            f.write(os.urandom(int(size)))
    return test_dir


@pytest.fixture
def repo():
    repo = api.Repo(Path.cwd(), init=True)
    return repo


def test_setup(create_test_dir):
    assert str(create_test_dir.stem) == "wgit_testing"


def test_api_init(capsys, repo):
    repo = api.Repo(Path.cwd(), init=True)
    assert Path(".wgit/sha1_refs.json").is_file()
    assert Path(".wgit/.gitignore").is_file()
    assert Path(".wgit/.git").exists()
    assert Path(".wgit/.gitignore").exists()


def test_api_add(capsys, repo):
    fnum = random.randint(0, 2)
    chkpt0 = f"checkpoint_{fnum}.pt"
    repo.add(chkpt0)

    sha1_hash = repo._sha1_store._get_sha1_hash(chkpt0)
    with open(os.path.join(".wgit", f"checkpoint_{fnum}.pt"), "r") as f:
        json_data = json.load(f)

    sha1_dir_0 = f"{sha1_hash[:2]}/" + f"{sha1_hash[2:]}"
    assert json_data["SHA1"] == {"__sha1_full__": sha1_hash}


def test_api_commit(capsys, repo):
    commit_msg = "epoch_1"
    repo.commit(message=commit_msg)
    with open(".wgit/.git/logs/HEAD") as f:
        line = f.readlines()
    assert line[0].rstrip().split()[-1] == commit_msg


def test_api_status(capsys, repo):
    class StateMsg:
        def __init__(self, chkpt=None) -> None:
            self.clean = "Clean:  tracking no file\n"
            self.added = f"state - dirty:  change added, not commited:       {chkpt}\n"
            self.modified = f"state - dirty:  file modified, not added:   {chkpt}\n"
            self.committed = f"state - Clean:  tracking:     {chkpt}\n"

    # delete the repo and initialize a new one:
    shutil.rmtree(".wgit")
    repo = api.Repo(Path.cwd(), init=True)

    # check status before any file is added
    repo.status()
    captured = capsys.readouterr()
    assert captured.out == StateMsg().clean
    assert captured.err == ""

    # check status before after a file is added but not committed
    chkpt0 = f"checkpoint_{random.randint(0, 1)}.pt"
    repo.add(chkpt0)
    repo.status()
    captured = capsys.readouterr()
    assert captured.out == StateMsg(chkpt0).added
    assert captured.err == ""

    # check status after commit
    repo.commit("e1")
    repo.status()
    captured = capsys.readouterr()
    assert captured.out == StateMsg(chkpt0).committed
    assert captured.err == ""

    # check status after a new change has been made to the file
    with open(chkpt0, "wb") as f:
        f.write(os.urandom(int(15e5)))
    repo.status()
    captured = capsys.readouterr()
    assert captured.out == StateMsg(chkpt0).modified
    assert captured.err == ""

    # add the new changes made to weigit
    repo.add(chkpt0)
    repo.status()
    captured = capsys.readouterr()
    assert captured.out == StateMsg(chkpt0).added
    assert captured.err == ""

    # check status after a new different file is added to be tracked by weigit
    chkpt3 = "checkpoint_3.pt"
    repo.add(chkpt3)
    repo.status()
    captured = capsys.readouterr()
    assert captured.out == StateMsg(chkpt0).added + StateMsg(chkpt3).added
    assert captured.err == ""

    # check status after the new file is commited to be tracked by weigit
    repo.commit("e2")
    repo.status()
    captured = capsys.readouterr()
    assert captured.out == StateMsg(chkpt0).committed + StateMsg(chkpt3).committed
    assert captured.err == ""


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
