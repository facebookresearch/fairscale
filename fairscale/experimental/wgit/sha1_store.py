# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, Union, cast

import torch
from torch import Tensor

from .utils import ExitCode

# This is a fixed dir name we use for sha1_store. It should not be changed
# for backward compatibility reasons.
SHA1_STORE_DIR_NAME = "sha1_store"


class SHA1_Store:
    """
    This class represents a SHA1 checksum based storage dir for state_dict
    and tensors.

    This means the same content will not be stored multiple times, resulting
    in space savings. (a.k.a. de-duplication)

    To make things easier for the callers, this class accept input data
    as files, state_dict or tensors. This class always returns in-memory
    data, not on-disk files. This class doesn't really care or know the actually
    data types. It uses torch.save() and torch.load() to do serialization.

    A key issue is dealing with content deletion. We use a reference counting
    algorithm, which means the caller must have symmetrical add/remove calls
    for each object.

    We used to support children-parent dependency graph and ref counting, but
    it is flawed since a grand-child can have the same SHA1 as the grand-parent,
    resulting in a cycle. This means caller must compute which parent is safe
    to delete in a version tracking graph. The lesson here is that content
    addressibility and dependency graphs do not mix well.

    Args:
        parent_path (Path):
            The parent path in which a SHA1_Store will be created.
        init (bool, optional):
            - If ``True``, initializes a new SHA1_Store in the parent_path. Initialization
              creates a `sha1_store` directory in ./<parent_path>/,
              and a `ref_count.json` within ./<parent_path>/.
            - If ``False``, a new `sha1_store` dir is not initialized and the existing
              `sha1_store` is used to init this class, populating `_json_dict`, and other
              attributes.
            - Default: False
        sha1_buf_size (int):
            Buffer size used for checksumming. Default: 100MB.
        tmp_dir (str):
            Dir for temporary files if input is an in-memory object.
    """

    def __init__(
        self, parent_path: Path, init: bool = False, sha1_buf_size: int = 100 * 1024 * 1024, tmp_dir: str = ""
    ) -> None:
        """Create or wrap (if already exists) a sha1_store."""
        self._path = parent_path.joinpath(SHA1_STORE_DIR_NAME)
        self._ref_file_path = self._path.joinpath("ref_count.json")
        self._sha1_buf_size = sha1_buf_size
        self._json_dict: Dict[str, Any] = {"created_on": time.ctime()}

        # Initialize the sha1_store if not exist and init==True.
        if init and not self._path.exists():
            try:
                Path.mkdir(self._path, parents=False, exist_ok=False)
            except FileExistsError as error:
                sys.stderr.write(f"An exception occured while creating Sha1_store: {repr(error)}\n")
                sys.exit(ExitCode.FILE_EXISTS_ERROR)
            # Create a new json file for this new store.
            self._store_json_dict()

        # This is an internal error since caller of this our own wgit code.
        assert self._path.exists(), "SHA1 store does not exist and init==False"

        # Init temp dir.
        if tmp_dir:
            # Caller supplied tmp dir
            assert Path(tmp_dir).is_dir(), "incorrect input"
            self._tmp_dir = Path(tmp_dir)
        else:
            # Default tmp dir, need to clean it.
            self._tmp_dir = self._path.joinpath("tmp")
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir.mkdir()

    def _load_json_dict(self) -> None:
        """Loading json dict from disk."""
        with open(self._ref_file_path, "r") as f:
            self._json_dict = json.load(f)

    def _store_json_dict(self) -> None:
        """Storing json dict to disk."""
        with open(self._ref_file_path, "w", encoding="utf-8") as f:
            json.dump(self._json_dict, f, ensure_ascii=False, indent=4)

    def add(self, file_or_obj: Union[Path, Tensor, OrderedDict]) -> str:
        """
        Adds a file/object to the internal sha1_store and the sha1 references
        accordingly.

        First, a sha1 hash is calculated. Utilizing the sha1 hash string, the actual file
        in <file_or_obj> is moved within the sha1_store and the reference file is updated.
        If the input is an object, it will be store in the self._tmp_dir and then moved.

        Args:
            file_or_obj (str or tensor or OrderedDict):
                Path to the file to be added to the sha1_store or an in-memory object
                that can be handled by torch.save.
        """
        # Use `isinstance` not type() == Path since pathlib returns OS specific
        # Path types, which inherit from the Path class.
        if isinstance(file_or_obj, (Path, str)):
            # Make sure it is a valid file.
            torch.load(cast(Union[Path, str], file_or_obj))
            file_path = Path(file_or_obj)
            remove_tmp = False
        elif isinstance(file_or_obj, (Tensor, OrderedDict)):
            # Serialize the object into a tmp file.
            file_path = self._get_tmp_file_path()
            torch.save(cast(Union[Tensor, OrderedDict], file_or_obj), file_path)
            remove_tmp = True
        else:
            assert False, f"incorrect input {type(file_or_obj)}"

        # Get SHA1 from the file.
        assert isinstance(file_path, Path), type(file_path)
        sha1_hash = self._get_sha1_hash(file_path)

        # Add reference.
        ref_count = self._add_ref(sha1_hash, True)

        if ref_count == 1:
            # First time adding

            # Create the file dir, if needed.
            repo_fdir = self._sha1_to_dir(sha1_hash)
            if not repo_fdir.exists():
                try:
                    repo_fdir.mkdir(exist_ok=True, parents=True)
                except FileExistsError as error:
                    sys.stderr.write(f"An exception occured: {repr(error)}\n")
                    sys.exit(ExitCode.FILE_EXISTS_ERROR)

            # Transfer the file to the internal sha1_store
            repo_fpath = repo_fdir.joinpath(sha1_hash)
            try:
                shutil.copy2(file_path, repo_fpath)
            except BaseException as error:
                # Something went wrong, perhaps out of space, or race condition due to lack of locking.
                # TODO (Min): proper handle the error and recover when we learn more here.
                sys.stderr.write(f"An exception occured: {repr(error)}\n")
                ref_count = self._add_ref(sha1_hash, False)

        # Clean up if needed.
        if remove_tmp:
            file_path.unlink()

        return sha1_hash

    def get(self, sha1: str) -> Union[Tensor, OrderedDict]:
        """Get data from a SHA1

        Args:
            sha1 (str):
                SHA1 of the object to get.

        Returns:
            (Tensor or OrderedDict):
                In-memory object.
        """
        raise NotImplementedError()

    def delete(self, sha1: str) -> None:
        """Delete a SHA1

        Args:
            sha1 (str):
                SHA1 of the object to delete.

        """
        raise NotImplementedError()

    def _get_sha1_hash(self, file_path: Union[str, Path]) -> str:
        """Return the sha1 hash of a file

        Args:
            file_path (str, Path):
                Path to the file whose sha1 hash is to be calculalated and returned.

        Returns:
            (str):
                The SHA1 computed.
        """
        sha1 = hashlib.sha1()
        with open(file_path, "rb") as f:
            while True:
                data = f.read(self._sha1_buf_size)
                if not data:
                    break
                sha1.update(data)
        return sha1.hexdigest()

    def _get_tmp_file_path(self) -> Path:
        """Helper to get a tmp file name under self.tmp_dir."""
        return Path(tempfile.mkstemp(dir=self._tmp_dir)[1])

    def _sha1_to_dir(self, sha1: str) -> Path:
        """Helper to get the internal dir for a file based on its SHA1"""
        # Using first 2 letters of the sha1, which results 26 * 26 = 676 subdirs under the top
        # level. Then, using another 2 letters for sub-sub-dir. If each dir holds 1000 files, this
        # can hold 450 millions files.
        # NOTE: this can NOT be changed for backward compatible reasons once in production.
        assert len(sha1) > 4, "sha1 too short"
        part1, part2 = sha1[:2], sha1[2:4]
        return self._path.joinpath(part1, part2)

    def _add_ref(self, current_sha1_hash: str, inc: bool) -> int:
        """
        Update the reference count.

        If the reference counting file does not have this sha1, then a new tracking
        entry of the added.

        Args:
            current_sha1_hash (str):
                The sha1 hash of the incoming added file.
            inc (bool):
                Increment or decrement.

        Returns:
            (int):
                Resulting ref count.
        """
        self._load_json_dict()

        # Init the entry if needed.
        if current_sha1_hash not in self._json_dict:
            self._json_dict[current_sha1_hash] = 0

        # Update the ref count.
        self._json_dict[current_sha1_hash] += 1 if inc else -1
        assert self._json_dict[current_sha1_hash] >= 0, "negative ref count"

        self._store_json_dict()

        return self._json_dict[current_sha1_hash]
