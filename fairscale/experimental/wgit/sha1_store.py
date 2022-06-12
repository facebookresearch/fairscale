# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Union, cast

from .utils import ExitCode


class SHA1_store:
    def __init__(self, weigit_path: Path, metadata_file: Path, sha1_refs: Path, init: bool = False) -> None:
        """
        Planned Features:
            1. def init
            2. def add <file or data> -> SHA1
            3. def remove (SHA1)
            4. def add_ref(children_SHA1, parent_SHA1)
            5. def read(SHA1): ->
            6. def lookup(SHA1): -> file path to the data. NotFound Exception if not found.
        """
        # should use the sha1_refs.json to track the parent references.
        self.name = "sha1_store"
        self.path = weigit_path.joinpath(self.name)

        self._ref_file_name = Path.cwd().joinpath(sha1_refs)
        self._metadata_file = Path.cwd().joinpath(metadata_file)

        # initialize the sha1_store
        if init:
            try:
                if not self.path.exists():
                    Path.mkdir(self.path, parents=False, exist_ok=False)
            except FileExistsError as error:
                sys.stderr.write(f"An exception occured while creating Sha1_store: {repr(error)}\n")
                sys.exit(ExitCode.FILE_EXISTS_ERROR)

    def add(self, file_path: str) -> None:
        """Adds a file/checkpoint to the internal sha1_store and update the metadata and the
        sha1 references accordingly.
        """
        sha1_hash = self.get_sha1_hash(file_path)
        # use the sha1_hash to create a directory with first2 sha naming convention
        try:
            repo_fdir = self.path.joinpath(sha1_hash[:2])
            repo_fdir.mkdir(exist_ok=False)
        except FileExistsError as error:
            sys.stderr.write(f"An exception occured: {repr(error)}\n")
            sys.exit(ExitCode.FILE_EXISTS_ERROR)
        try:
            # First transfer the file to the internal sha1_store
            repo_fpath = Path.cwd().joinpath(repo_fdir, sha1_hash[2:])
            shutil.copy2(file_path, repo_fpath)
            change_time = Path(repo_fpath).stat().st_ctime

            # Create the dependency Graph and track reference
            self._add_ref(current_sha1_hash=sha1_hash)
            metadata = {
                "SHA1": {
                    "__sha1_full__": sha1_hash,
                },
                "file_path": str(repo_fpath),
                "time_stamp": change_time,
            }
            # Populate the meta_data file with the meta_data and git add
            self._add_metadata_to_json(metadata)

        except BaseException as error:
            # in case of failure: Cleans up the sub-directories created to store sha1-named checkpoints
            sys.stderr.write(f"An exception occured: {repr(error)}\n")
            shutil.rmtree(repo_fdir)

    def _add_ref(self, current_sha1_hash: str) -> None:
        """Populates the sha1_refs.json file when file is added and keeps track of reference to earlier commits"""

        if not os.path.getsize(self._ref_file_name):  # If no entry yet
            with open(self._ref_file_name) as f:
                ref_data = {
                    current_sha1_hash: {"parent": "ROOT", "child": "HEAD", "ref_count": 0},
                }
            with open(self._ref_file_name, "w", encoding="utf-8") as f:
                json.dump(ref_data, f, ensure_ascii=False, indent=4)
        else:
            with open(self._ref_file_name, "r") as f:
                ref_data = json.load(f)

            # get the last head and replace it's child from HEAD -> this sha1
            for key, vals in ref_data.items():
                if vals["child"] == "HEAD":
                    parent = key

            ref_data[parent]["child"] = current_sha1_hash

            # increase the ref counter of that (now parent sha1)
            ref_count = cast(int, ref_data[parent]["ref_count"])
            ref_count += 1
            ref_data[parent]["ref_count"] = ref_count

            # Add this new sha1 as a new entry, make the earlier sha1 a parent
            # make "HEAD" as a child, and json dump
            ref_data[current_sha1_hash] = {"parent": parent, "child": "HEAD", "ref_count": 0}

            # Try
            with open(self._ref_file_name, "w", encoding="utf-8") as f:
                json.dump(ref_data, f, ensure_ascii=False, indent=4)

    def get_sha1_hash(self, file_path: Union[str, Path]) -> str:
        """ " return the sha1 hash of a file"""
        SHA1_BUF_SIZE = 104857600  # Reading file in 100MB chunks

        sha1 = hashlib.sha1()
        with open(file_path, "rb") as f:
            while True:
                data = f.read(SHA1_BUF_SIZE)
                if not data:
                    break
                sha1.update(data)
        return sha1.hexdigest()

    def _add_metadata_to_json(self, metadata: dict) -> None:
        """Populates the meta_data_file: checkpoint.pt with the meta_data"""
        file_pt_json = self._metadata_file
        with open(file_pt_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
