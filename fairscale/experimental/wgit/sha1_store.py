# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Tuple, Union, cast

from .utils import ExitCode


class SHA1_store:
    """
    Represent the sha1_store within the WeiGit repo for handling added file to the store and managing references.

    Args:
        weigit_path (pathlib.Path)
            The path to the weigit repo where a sha1_store will be created, or if already exists will be wrapped.
        init (bool, optional)
            - If ``True``, initializes a new sha1_store in the weigit_path. Initialization creates a `sha1_store` directory within WeiGit repo in ./<weigit_path>/,
                and a `sha1_refs.json` withiin ./<weigit_path>/.
            - If ``False``, a new sha1_store is not initialized and the existing sha1_store is simply wrapped, populating the `name`, `path` and the `ref_file_path` attributes.
            - Default: False
    Methods:
        add(in_file_path: str)
            adds a file to the sha1_store. If a new version of an existing file is added, updates the reference to the existing file.

        get_sha1_hash(file_path: Union[str, Path])
            Takes a file and returns its SHA1 checksum
    """

    def __init__(self, weigit_path: Path, init: bool = False) -> None:
        """Create or wrap (if already exists) a sha1_store within the WeiGit repo."""
        # should use the sha1_refs.json to track the parent references.
        self.name = "sha1_store"
        self.path = weigit_path.joinpath(self.name)
        self.ref_file_path = weigit_path.joinpath("sha1_refs.json")

        self._weigit_path = weigit_path
        # initialize the sha1_store
        if init:
            try:
                if not self.path.exists():
                    Path.mkdir(self.path, parents=False, exist_ok=False)
                    self.ref_file_path.touch(exist_ok=False)
            except FileExistsError as error:
                sys.stderr.write(f"An exception occured while creating Sha1_store: {repr(error)}\n")
                sys.exit(ExitCode.FILE_EXISTS_ERROR)

    def add(self, in_file_path: str) -> None:
        """
        Adds a file/checkpoint to the internal sha1_store and updates the metadata and the sha1 references accordingly.
        First, a metadata file of the same name as the input file to be tracked is created in the  WeiGit repo.
        A sha1 hash is then calculated. Utilizing the sha1 hash string, the actual file in <in_file_path> is moved
        within the sha1_store, and the metadata file of the same name is populated with the metadata of the input file.

        Args:
            in_file_path (str): path to the file to be added to the sha1_store.
        """
        # create the metadata file
        file_path = Path(in_file_path)
        metadata_file = self._weigit_path.joinpath(file_path.name)
        metadata_file.touch()

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
            self._add_ref(file_path, current_sha1_hash=sha1_hash)
            metadata = {
                "SHA1": {
                    "__sha1_full__": sha1_hash,
                },
                "file_path": str(repo_fpath),
                "time_stamp": change_time,
            }
            # Populate the meta_data file with the meta_data and git add
            self._write_to_json(metadata_file, metadata)

        except BaseException as error:
            # in case of failure: Cleans up the sub-directories created to store sha1-named checkpoints
            sys.stderr.write(f"An exception occured: {repr(error)}\n")
            shutil.rmtree(repo_fdir)

    def get_sha1_hash(self, file_path: Union[str, Path]) -> str:
        """return the sha1 hash of a file

        Args:
            file_path (str, Path): Path to the file whose sha1 hash is to be calculalated and returned.

        """
        SHA1_BUF_SIZE = 104857600  # Reading file in 100MB chunks

        sha1 = hashlib.sha1()
        with open(file_path, "rb") as f:
            while True:
                data = f.read(SHA1_BUF_SIZE)
                if not data:
                    break
                sha1.update(data)
        return sha1.hexdigest()

    def _add_ref(self, file_path: Path, current_sha1_hash: str) -> None:
        """
        Populates the sha1_refs.json file when file is added and keeps track of reference to earlier file additions.
        If the sha1_refs.json file is empty, then a new tracking entry of the added file is logged in the sha1_refs file.
        If the file already has an entry, first it checks if the incoming new added file is a new version of any of the
        existing entries. If it is, then logs the tracking info as a new version of that existing entry.
        Otherwise a new entry for the new added file is created for tracking.

        Args:
            file_path (pathlib.Path)
                Path to the incoming added file.
            current_sha1_hash (str)
                The sha1 hash of the incoming added file.
        """
        # Check the current state of the reference file and check if the added file already has an entry.
        sha1_refs_empty, entry_exists = self._sha1_refs_file_state(file_path)

        if sha1_refs_empty:
            with open(self.ref_file_path) as f:
                ref_data = {file_path.name: {current_sha1_hash: {"parent": "ROOT", "child": "HEAD", "ref_count": 0}}}
            self._write_to_json(self.ref_file_path, ref_data)

        else:
            with open(self.ref_file_path, "r") as f:
                ref_data = json.load(f)

            if entry_exists:
                # get the last head and replace it's child from HEAD -> this sha1
                for key, vals in ref_data[file_path.name].items():
                    if vals["child"] == "HEAD":
                        parent = key

                ref_data[file_path.name][parent]["child"] = current_sha1_hash

                # increase the ref counter of that (now parent sha1)
                ref_count = cast(int, ref_data[file_path.name][parent]["ref_count"])
                ref_count += 1
                ref_data[file_path.name][parent]["ref_count"] = ref_count

                # Add this new sha1 as a new entry, make the earlier sha1 a parent
                # make "HEAD" as a child, and json dump
                ref_data[file_path.name][current_sha1_hash] = {"parent": parent, "child": "HEAD", "ref_count": 0}
            else:
                ref_data[file_path.name] = {current_sha1_hash: {"parent": "ROOT", "child": "HEAD", "ref_count": 0}}

            self._write_to_json(self.ref_file_path, ref_data)

    def _sha1_refs_file_state(self, file_path: Path) -> Tuple[bool, bool]:
        """
        Checks the state of the sha1 reference file, whether the file is empty or not.
        If not empty, it checks whether the input file in <file_path> has an older entry (version)
        in the reference file.

        Args:
            file_path (pathlib.Path)
                input File whose entry will be checked if it exists in the reference file.
        """
        try:
            with open(self.ref_file_path, "r") as f:
                ref_data = json.load(f)
            if file_path.name in ref_data.keys():
                entry_exists: bool = True
            else:
                entry_exists = False
            sha1_refs_empty: bool = False
        except json.JSONDecodeError as error:
            # print(f"Json decode error")
            if not self.ref_file_path.stat().st_size:
                sha1_refs_empty = True
            entry_exists = False
        return sha1_refs_empty, entry_exists

    def _write_to_json(self, file: Path, data: dict) -> None:
        """
        Populates a json file with data.
        Args:
            file (pathlib.Path)
                path to the file to be written in.
            data (pathlib.Path)
                Data to be written in the file.
        """
        with open(file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
