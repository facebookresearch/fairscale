# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import experimental.wgit.weigit_api as weigit_api


def main(argv=None):
    desc = "WeiGit: A git-like tool for Model checkpoint tracking"
    parser = argparse.ArgumentParser(description=desc)

    # flags
    parser.add_argument("-i", "--init", action="store_true", help="Initialize a weigit repository!")
    parser.add_argument(
        "-a",
        "--add",
        default="",
        type=str,
        metavar="FILE_PATH",
        help="path to the checkpoint to be added (default: none)",
    )
    parser.add_argument("-s", "--status", action="store_true", help="Show the .wgit status")
    parser.add_argument("-l", "--log", action="store_true", help="Show the log of .wgit")
    parser.add_argument("-c", "--commit", action="store_true", help="Commit to .wgit")
    parser.add_argument("-co", "--checkout", action="store_true", help="checkout from a commit")

    args = parser.parse_args(argv)

    if args.init:
        weigit = weigit_api.WeiGit()
        print("Wgit has been initialized!")

    if args.add:
        # weigit = weigit_api.WeiGit()
        weigit_api.WeiGit.add(args.add)

    if args.status:
        # weigit = weigit_api.WeiGit()
        weigit_api.WeiGit.status()

    if args.log:
        # weigit = weigit_api.WeiGit()
        weigit_api.WeiGit.log()

    if args.commit:
        # weigit = weigit_api.WeiGit()
        weigit_api.WeiGit.commit()

    if args.checkout:
        # weigit = weigit_api.WeiGit()
        weigit_api.WeiGit.checkout()


if __name__ == "__main__":
    main()
