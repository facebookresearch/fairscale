# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import experimental.wgit.cli as cli


def test_cli_init(capsys):
    cli.main(["--init"])
    out, err = capsys.readouterr()
    assert out == "Hello World, Wgit has been initialized!\n"
    assert err == ""
