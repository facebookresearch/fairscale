# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import experimental.wgit.weigit_api as weigit_api


def main(argv=None):
    # create the parser for the "a" command
    desc = "WeiGit: A git-like tool for model weight tracking"

    # Create a top level parser
    parser = argparse.ArgumentParser(description=desc)
    subparsers = parser.add_subparsers()

    # Version
    version_parser = subparsers.add_parser("version", description="Display version")
    version_parser.set_defaults(command="version", subcommand="")

    # Repo
    init_parser = subparsers.add_parser("init", description="Initialize a weigit repo")
    init_parser.add_argument("init", action="store_true", help="initialize the repo")

    status_parser = subparsers.add_parser("status", description="Show the repo's current status")
    status_parser.add_argument("status", action="store_true", help="Show the repo's current status")

    log_parser = subparsers.add_parser("log", description="Show the repo's history log")
    log_parser.add_argument("log", action="store_true", help="Show the repo's history log")

    commit_parser = subparsers.add_parser("commit", description="Commit the staged changes")
    commit_parser.add_argument("commit", action="store_true", help="Commit the staged changes")

    add_parser = subparsers.add_parser("add", description="add a file to the staged changeset (default: none)")
    add_parser.add_argument(
        "add",
        default="",
        type=str,
        metavar="FILE_PATH",
        help="add a file to the staged changeset (default: none)",
    )

    checkout_parser = subparsers.add_parser("checkout", description="Initialize a weigit repo")
    checkout_parser.add_argument(
        "checkout",
        default="",
        type=str,
        metavar="FILE_SHA1",
        help="checkout from a commit",
    )
    parser.set_defaults(init=False, add="", status=False, log=False, commit=False, checkout="")

    args = parser.parse_args(argv)
    # print(f"Print args: {args}")

    if args.init:
        weigit = weigit_api.WeiGit()
        print("Wgit has been initialized!")

    if args.add:
        weigit_api.WeiGit.add(args.add)

    if args.status:
        weigit_api.WeiGit.status()

    if args.log:
        weigit_api.WeiGit.log()

    if args.commit:
        weigit_api.WeiGit.commit()

    if args.checkout:
        weigit_api.WeiGit.checkout()


if __name__ == "__main__":
    main()
