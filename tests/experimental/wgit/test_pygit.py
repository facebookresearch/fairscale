# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import shutil

import pytest

from fairscale.experimental.wgit.pygit import PyGit


@pytest.fixture
def repo_data():
    test_dirs = Path("temp_wgit_testing/.wgit")
    file1, file2 = "test_file1", "test_file2"
    out_dict = {
        "test_path": Path.cwd().joinpath(test_dirs),
        "file1": file1,
        "file2": file2,
    }
    return out_dict


@pytest.fixture
def pygit_repo_wrap(repo_data):
    path = Path.cwd().joinpath(repo_data["test_path"])
    pygit_repo_wrap = PyGit(path, gitignore=[repo_data["file1"], repo_data["file2"]])
    return pygit_repo_wrap


def test_setup(repo_data):
    curr_dir = Path.cwd()
    test_dir = curr_dir.joinpath(repo_data["test_path"])

    # Initialize the repo for the first time
    pygit_repo = PyGit(test_dir, gitignore=["test_file1", "test_file2"])

    # create sample files
    test_dir.joinpath("test_file_1.pt").touch()
    test_dir.joinpath("test_file_2.pt").touch()

    assert test_dir.stem == str(pygit_repo.path.parent.stem)


def test_pygit_add(pygit_repo_wrap):
    """Tests the add functionality of the PyGit class"""
    assert str(pygit_repo_wrap.path.parent.stem) == ".wgit"

    repo = pygit_repo_wrap.repo
    # File status 128 in pygit2 signifies file has Not been added yet
    assert repo.status()[".gitignore"] == 128
    assert repo.status()["test_file_1.pt"] == 128
    assert repo.status()["test_file_2.pt"] == 128
    pygit_repo_wrap.add()
    # File status 1 in pygit2 signifies file has been added to git repo
    assert repo.status()[".gitignore"] == 1
    assert repo.status()["test_file_1.pt"] == 1
    assert repo.status()["test_file_2.pt"] == 1


def test_pygit_commit(pygit_repo_wrap):
    """Tests the add functionality of the PyGit class"""
    assert str(pygit_repo_wrap.path.parent.stem) == ".wgit"

    repo = pygit_repo_wrap.repo
    # File status 1 in pygit2 signifies file has been added
    assert repo.status()[".gitignore"] == 1
    assert repo.status()["test_file_1.pt"] == 1
    assert repo.status()["test_file_2.pt"] == 1
    pygit_repo_wrap.commit("random_message")
    # File status {} in pygit2 implies commit has been made
    assert repo.status() == {}


def test_tear_down(repo_data):
    # clean up: delete the .wgit directory created during this Test
    # Making sure the current directory is ./temp_wgit_testing before removing test dir
    if (repo_data["test_path"].parent.stem == "temp_wgit_testing") and (repo_data["test_path"].stem == ".wgit"):
        shutil.rmtree(repo_data["test_path"].parent)
    else:
        raise Exception("Exception in testing directory tear down!")
