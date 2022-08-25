# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from pathlib import Path
import shutil

import pytest
import torch
from torch import nn

import fairscale.experimental.wgit.cli as cli
from fairscale.experimental.wgit.sha1_store import SHA1_Store


@pytest.fixture(scope="module")
def create_test_dir():
    """This setup function runs once per test of this module and
    it creates a repo, in the process, testing the init function.
    """
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
    size_list = [30e5, 35e5, 40e5]
    for i, size in enumerate(size_list):
        torch.save(nn.Linear(1, int(size)).state_dict(), f"checkpoint_{i}.pt")

    # Test init.
    cli.main(["init"])
    assert str(test_dir.stem) == "wgit_testing"

    return test_dir


def test_cli_init(create_test_dir, capsys):
    # Check if the json and other files have been created by the init
    assert Path(".wgit/sha1_store").is_dir()
    assert Path(".wgit/.gitignore").is_file()
    assert Path(".wgit/.git").exists()


def test_cli_add(create_test_dir, capsys):
    chkpt0 = "checkpoint_0.pt"
    cli.main(["add", "--no_per_tensor", chkpt0])

    sha1_store = SHA1_Store(
        Path.cwd().joinpath(".wgit", "sha1_store"),
        init=False,
    )
    sha1_hash = sha1_store._get_sha1_hash(chkpt0)
    with open(os.path.join(".wgit", "wgit_testing/checkpoint_0.pt"), "r") as f:
        json_data = json.load(f)

    sha1_dir_0 = f"{sha1_hash[:2]}/" + f"{sha1_hash[2:]}"
    # The sha1 are different because add internally use a different pickle method.
    assert json_data["SHA1"] != sha1_hash


def test_cli_commit(capsys):
    commit_msg = "epoch_1"
    cli.main(["commit", "-m", f"{commit_msg}"])
    with open(".wgit/.git/logs/HEAD") as f:
        line = f.readlines()
    assert line[0].rstrip().split()[-1] == commit_msg


def test_cli_status(capsys):
    cli.main(["status"])
    captured = capsys.readouterr()
    assert captured.out == "{'wgit_testing/checkpoint_0.pt': <RepoStatus.CLEAN: 1>}\n"
    assert captured.err == ""


def test_cli_log(capsys):
    cli.main(["log"])
    captured = capsys.readouterr()
    assert captured.out == "wgit log\n"
    assert captured.err == ""


def test_cli_checkout(capsys):
    try:
        cli.main(["checkout", "sha1"])
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
