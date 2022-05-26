# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
import shutil

import experimental.wgit.cli as cli


def setup_module(module):
    cli.main(["--init"])


def test_cli_init(capsys):
    # cli.main(["--init"])
    # Check if the json and other files have been created by the init
    assert pathlib.Path(".wgit/sha1_ref_count.json").is_file()
    assert pathlib.Path(".wgit/.gitignore").is_file()
    assert pathlib.Path(".wgit/.git").exists()


def test_cli_add(capsys):
    cli.main(["--add", "test"])
    captured = capsys.readouterr()
    assert captured.out == "wgit added\n"
    assert captured.err == ""


def test_cli_status(capsys):
    cli.main(["--status"])
    captured = capsys.readouterr()
    assert captured.out == "wgit status\n"
    assert captured.err == ""


def test_cli_log(capsys):
    cli.main(["--log"])
    captured = capsys.readouterr()
    assert captured.out == "wgit log\n"
    assert captured.err == ""


def test_cli_commit(capsys):
    cli.main(["--commit"])
    captured = capsys.readouterr()
    assert captured.out == "wgit commit\n"
    assert captured.err == ""


def test_cli_checkout(capsys):
    cli.main(["--checkout"])
    captured = capsys.readouterr()
    assert captured.out == "wgit checkout\n"
    assert captured.err == ""


def teardown_module(module):
    # clean up: delete the .wgit directory created during this Test
    shutil.rmtree(".wgit")
