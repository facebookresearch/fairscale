# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
from pathlib import Path
import sys
import typing

from fairscale.experimental.wgit.utils import ExitCode


class SHA1_store:
    """
    Planned Features:
        1. def init
        2. def add <file or data> -> SHA1
        3. def remove (SHA1)
        4. def add_ref(children_SHA1, parent_SHA1)
        5. def read(SHA1): ->
        6. def lookup(SHA1): -> file path to the data. NotFound Exception if not found.
    """

    def __init__(self) -> None:
        try:
            sha1_store_path = Path.cwd().joinpath(".wgit", "sha1_store")
            if not sha1_store_path.exists():
                Path.mkdir(sha1_store_path, parents=False, exist_ok=False)
        except FileExistsError as error:
            sys.stderr.write(f"An exception occured while creating Sha1_store: {repr(error)}\n")
            sys.exit(ExitCode.FILE_EXISTS_ERROR)

    def add_ref(self, current_sha1_hash: str) -> None:
        # should use the sha1_refser to track the parent and children references. How exactly is open question to me!
        ref_file = ".wgit/sha1_refs.json"
        print(f"\n {type(current_sha1_hash)} \n")
        if not os.path.getsize(ref_file):  # If no entry yet
            with open(ref_file) as f:
                print(f"\n {current_sha1_hash} \n")
                ref_data = {
                    current_sha1_hash: {"parent": "ROOT", "child": "HEAD", "ref_count": 0},
                }
            with open(ref_file, "w", encoding="utf-8") as f:
                json.dump(ref_data, f, ensure_ascii=False, indent=4)
        else:
            with open(ref_file, "r") as f:
                ref_data = json.load(f)

            # get the last head and replace it's child from HEAD -> this sha1
            for key, vals in ref_data.items():
                if vals["child"] == "HEAD":
                    parent = key

            ref_data[parent]["child"] = current_sha1_hash

            # increase the ref counter of that (now parent sha1)
            ref_count = typing.cast(int, ref_data[parent]["ref_count"])
            ref_count += 1
            ref_data[parent]["ref_count"] = ref_count

            # Add this new sha1 as a new entry, make the earlier sha1 a parent
            # make "HEAD" as a child, and json dump
            ref_data[current_sha1_hash] = {"parent": parent, "child": "HEAD", "ref_count": 0}

            # Try
            with open(ref_file, "w", encoding="utf-8") as f:
                json.dump(ref_data, f, ensure_ascii=False, indent=4)
