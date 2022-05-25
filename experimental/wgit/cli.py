# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse


def main(argv=None):
    desc = "WeiGit checkpoint tracking"
    parser = argparse.ArgumentParser(description=desc)

    # flags
    parser.add_argument("-i", "--init", action="store_true", help="Initialize a weigit repository!")

    args = parser.parse_args(argv)

    if args.init:
        print("Hello World, Wgit has been initialized!")


if __name__ == "__main__":
    main()
