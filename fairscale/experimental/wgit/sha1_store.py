# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Union

from .utils import ExitCode

# This is a fixed dir name we use for sha1_store. It should not be changed
# for backward compatibility reasons.
SHA1_STORE_DIR_NAME = "sha1_store"


class SHA1_Store:
    """
    This class represents a SHA1 checksum based storage area within a WeiGit repo
    for handling added file to the store and managing references in a content based
    fashion. This means the same content will not be stored multiple times, end up
    being de-duplicated.

    Args:
        parent_path (pathlib.Path)
            The parent path in which a SHA1_Store will be created.
        init (bool, optional)
            - If ``True``, initializes a new SHA1_Store in the parent_path. Initialization
              creates a `sha1_store` directory within WeiGit repo in ./<parent_path>/,
              and a `ref_count.json` within ./<parent_path>/.
            - If ``False``, a new `sha1_store` dir is not initialized and the existing
              `sha1_store` is used to init this class, populating `created_on`, `path` and
              `ref_file_path` attributes.
            - Default: False
    """

    def __init__(self, parent_path: Path, init: bool = False) -> None:
        """Create or wrap (if already exists) a sha1_store within the WeiGit repo."""
        self.path = parent_path.joinpath(SHA1_STORE_DIR_NAME)
        self.ref_file_path = parent_path.joinpath("ref_count.json")

        # initialize the sha1_store if not exist and init==True
        if init and not self.path.exists():
            try:
                Path.mkdir(self.path, parents=False, exist_ok=False)
                self._write_to_json(self.ref_file_path, {"created_on": time.ctime()})
            except FileExistsError as error:
                sys.stderr.write(f"An exception occured while creating Sha1_store: {repr(error)}\n")
                sys.exit(ExitCode.FILE_EXISTS_ERROR)

        # By now, we can load the store in memory into this class.
        with open(self.ref_file_path, "r") as f:
            self.created_on = json.load(f)["created_on"]

    def add(self, file_path: Path, parent_sha1: str) -> str:
        """
        Adds a file/checkpoint to the internal sha1_store and the sha1 references
        accordingly.

        First, a sha1 hash is calculated. Utilizing the sha1 hash string, the actual file
        in <in_file_path> is moved within the sha1_store and the sha1 reference file is
        updated accordingly with the information of their parents
        node (if exists) and whether the new version is a leaf node or not.

        Args:
            file_path (str):
                path to the file to be added to the sha1_store.
            parent_sha1 (str):
                sha1 of the parent object (if any).
        """
        sha1_hash = self._get_sha1_hash(file_path)
        # use the sha1_hash to create a directory with first2 sha naming convention
        try:
            repo_fdir = self.path.joinpath(sha1_hash[:2])
            repo_fdir.mkdir(exist_ok=True)
        except FileExistsError as error:
            sys.stderr.write(f"An exception occured: {repr(error)}\n")
            sys.exit(ExitCode.FILE_EXISTS_ERROR)
        try:
            # First transfer the file to the internal sha1_store
            repo_fpath = Path.cwd().joinpath(repo_fdir, sha1_hash[2:])
            shutil.copy2(file_path, repo_fpath)

            # Create the dependency Graph and track reference
            self._add_ref(sha1_hash, parent_sha1)

        except BaseException as error:
            # in case of failure: Cleans up the sub-directories created to store sha1-named checkpoints
            sys.stderr.write(f"An exception occured: {repr(error)}\n")
            shutil.rmtree(repo_fdir)
        return sha1_hash

    def _get_sha1_hash(self, file_path: Union[str, Path]) -> str:
        """return the sha1 hash of a file

        Args:
            file_path (str, Path):
                Path to the file whose sha1 hash is to be calculalated and returned.
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

    def _add_ref(self, current_sha1_hash: str, parent_hash: str) -> None:
        """
        Populates the reference counting file when file is added and keeps track of
        reference to earlier file additions.

        If the reference counting file does not have this sha1, then a new tracking
        entry of the added file is logged in the file.

        If the file already has an entry for this sha1, first we check if the
        incoming new added file is a new version of any of the existing entries.
        If it is, then logs the tracking info as a new version of that existing
        entry.  Otherwise a new entry for the new added file is created for tracking.

        Args:
            current_sha1_hash (str):
                The sha1 hash of the incoming added file.
            parent_hash (str):
                The sha1 hash of the parent file.
        """
        # Check the current state of the reference file and check if the added file already has an entry.
        refs_empty = self._ref_count_file_state()

        # if the file is empty: add the first entry
        if refs_empty:
            with open(self.ref_file_path) as f:
                ref_data = {current_sha1_hash: {"parent": "ROOT", "ref_count": 1, "is_leaf": True}}
            self._write_to_json(self.ref_file_path, ref_data)
        else:
            # Open sha1 reference file and check if there is a parent_hash not equal to Root?
            # if Yes, find parent and add the child. Else, just add a new entry
            with open(self.ref_file_path, "r") as f:
                ref_data = json.load(f)
            if parent_hash != "ROOT":
                # get the last head and replace it's child from HEAD -> this sha1
                ref_data[parent_hash]["is_leaf"] = False
                ref_data[current_sha1_hash] = {"parent": parent_hash, "ref_count": 1, "is_leaf": True}
            else:
                ref_data[current_sha1_hash] = {"parent": "ROOT", "ref_count": 1, "is_leaf": True}

            self._write_to_json(self.ref_file_path, ref_data)

    def _ref_count_file_state(self) -> bool:
        """
        Checks the state of the sha1 reference file, whether the file is empty or not.
        If not empty, it checks whether the input file in <file_path> has an older
        entry (version) in the reference file.

        Args:
            file_path (pathlib.Path):
                input File whose entry will be checked if it exists in the reference file.
        """
        try:
            with open(self.ref_file_path, "r") as f:
                ref_data = json.load(f)
            refs_empty = False
        except json.JSONDecodeError as error:
            if not self.ref_file_path.stat().st_size:
                refs_empty = True
        return refs_empty

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
