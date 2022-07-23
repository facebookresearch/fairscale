# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path
from typing import List

from . import Repo, version


def main(argv: List[str] = None) -> None:
    desc = "WeiGit: A git-like tool for model weight tracking"

    # top level parser and corresponding subparser
    parser = argparse.ArgumentParser(description=desc)
    subparsers = parser.add_subparsers(dest="command")

    # Version
    version_parser = subparsers.add_parser("version", description="Display version")
    version_parser.set_defaults(command="version", subcommand="")

    # Repo
    init_parser = subparsers.add_parser("init", description="Initialize a weigit repo")
    init_parser.add_argument("init", action="store_true", help="initialize the repo")

    status_parser = subparsers.add_parser("status", description="Shows the repo's current status")
    status_parser.add_argument("status", action="store_true", help="Show the repo's current status")

    add_parser = subparsers.add_parser("add", description="add a file to the staged changeset (default: none)")
    add_parser.add_argument(
        "add",
        default="",
        type=str,
        metavar="FILE_PATH",
        help="add a file to the staged changeset (default: none)",
    )
    add_parser.add_argument(
        "--no_per_tensor",
        action="store_true",
        help="Disable per-tensor adding of a file",
    )

    commit_parser = subparsers.add_parser("commit", description="Commits the staged changes")
    commit_parser.add_argument("commit", action="store_true", help="Commit the staged changes")
    commit_parser.add_argument(
        "-m",
        "--message",
        default="",
        type=str,
        metavar="MESSAGE",
        required=True,
        help="commit message",
    )

    checkout_parser = subparsers.add_parser("checkout", description="checkout from a commit")
    checkout_parser.add_argument(
        "checkout",
        default="",
        type=str,
        metavar="FILE_SHA1",
        help="checkout from a commit",
    )

    log_parser = subparsers.add_parser("log", description="Show the history log of the repo or optionally of a file.")
    log_parser.add_argument("log", action="store_true", help="Show the repo's history log")
    log_parser.add_argument(
        "-f",
        "--file",
        default="",
        type=str,
        metavar="FILE_PATH",
        help="Show the history log of a file",
    )

    args = parser.parse_args(argv)

    if args.command == "init":
        repo = Repo(Path.cwd(), init=True)

    if args.command == "add":
        repo = Repo(Path.cwd())
        repo.add(args.add, per_tensor=not args.no_per_tensor)

    if args.command == "status":
        repo = Repo(Path.cwd())
        out = repo.status()
        print(out)

    if args.command == "log":
        repo = Repo(Path.cwd())
        repo.log(args.file)

    if args.command == "commit":
        repo = Repo(Path.cwd())
        repo.commit(args.message)

    if args.command == "checkout":
        repo = Repo(Path.cwd())
        repo.checkout(args.checkout)

    if args.command == "version":
        print(".".join([str(x) for x in version.__version_tuple__]))


if __name__ == "__main__":
    main()
