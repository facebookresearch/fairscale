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
from typing import Any, Dict, Optional, Union, cast

import pgzip
import torch
from torch import Tensor

from .utils import ExitCode

# Const string keys for json file. Do not change for backward compatibilities.
RF_KEY = "ref_count"
COMP_KEY = "compressed"


def _get_json_entry(d: Dict[str, Any]) -> Dict[str, Any]:
    """Get a dict from a json entry.

    This fills in any missing entries in case we load an older version
    json file from the disk.
    """
    if RF_KEY not in d.keys():
        d[RF_KEY] = 0
    return d


def _copy_compressed(src: Path, dest: Path, thread: Optional[int], blocksize: int) -> None:
    """Helper to copy a file and compress it at the same time."""
    with open(str(src), "rb") as srcf:
        with pgzip.open(str(dest), "wb", compresslevel=5, thread=thread, blocksize=blocksize) as destf:
            while True:
                buf = srcf.read(blocksize)
                if len(buf) == 0:
                    break
                destf.write(buf)


def _copy_uncompressed(src: Path, dest: Path, thread: Optional[int], blocksize: int) -> None:
    """Helper to copy a file and uncompress it at the same time."""
    with open(str(dest), "wb") as destf:
        with pgzip.open(str(src), "rb", thread=thread, blocksize=blocksize) as srcf:
            while True:
                buf = srcf.read(blocksize)
                if len(buf) == 0:
                    break
                destf.write(buf)


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

    We support multicore compression for the data to be store on per-object basis.
    The ``torch.save()`` API uses zip format to store the data, but it appears to
    be uncompressed. Even if it can be made compressed, it is likely a single
    threaded compression. Therefore, we use pgzip to do parallel
    compression/decompression on top of it to use all the cores.

    Args:
        path (Path):
            The path in which a SHA1_Store will be created.
        init (bool, optional):
            - If ``True``, a new SHA1_Store in the path if not already exists.
            - Default: False
        sha1_buf_size (int):
            Buffer size used for checksumming. Default: 100MB.
        tmp_dir (str):
            Dir for temporary files if input is an in-memory object or output data needs
            to be decompressed first.
        pgzip_threads (int, optional):
            Number of threads (cores) used in compression. Default: None to use all cores.
        pgzip_block_size (int):
            Per-thread block size for compression. Default: 10MB.
    """

    def __init__(
        self,
        path: Path,
        init: bool = False,
        sha1_buf_size: int = 100 * 1024 * 1024,
        tmp_dir: str = "",
        pgzip_threads: Optional[int] = None,
        pgzip_block_size: int = 10 * 1024 * 1024,
    ) -> None:
        """Create or wrap (if already exists) a store."""
        self._path = path
        self._metadata_file_path = self._path.joinpath("metadata.json")
        self._sha1_buf_size = sha1_buf_size
        self._pgzip_threads = pgzip_threads
        self._pgzip_block_size = pgzip_block_size
        self._json_dict: Dict[str, Any] = {"created_on": time.ctime()}

        # Initialize the store if not exist and if init is True.
        if init and not self._path.exists():
            try:
                Path.mkdir(self._path, parents=False, exist_ok=False)
            except FileExistsError as error:
                sys.stderr.write(f"An exception occured while creating Sha1_store: {repr(error)}\n")
                sys.exit(ExitCode.FILE_EXISTS_ERROR)
            # Create a new json file for this new store.
            self._store_json_dict()

        # This is an internal error since caller of this our own wgit code.
        assert (
            self._path.exists() and self._metadata_file_path.exists()
        ), f"SHA1 store {self._path} does not exist and init is False"

        # Make sure there is a valid metadata file.
        self._load_json_dict()
        assert "created_on" in self._json_dict, f"Invalid SHA1 Store in {self._path}"

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
        with open(self._metadata_file_path, "r") as f:
            self._json_dict = json.load(f)

    def _store_json_dict(self) -> None:
        """Storing json dict to disk."""
        with open(self._metadata_file_path, "w", encoding="utf-8") as f:
            json.dump(self._json_dict, f, ensure_ascii=False, indent=4)

    def add(self, file_or_obj: Union[Path, Tensor, OrderedDict], compress: bool = False) -> str:
        """ Adds a file/object to this store and the sha1 references accordingly.

        First, a sha1 hash is calculated. Utilizing the sha1 hash string, the actual file
        in <file_or_obj> is moved within the store and the reference file is updated.
        If the input is an object, it will be store in the self._tmp_dir and then moved.

        If compress is True, the stored file is also compressed, which is useful for tensors
        with a lot of zeros.

        Args:
            file_or_obj (str or tensor or OrderedDict):
                Path to the file to be added to the store or an in-memory object
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
        ref_count = self._add_ref(sha1_hash, True, compress)

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

            # Transfer the file to the store.
            repo_fpath = repo_fdir.joinpath(sha1_hash)
            try:
                if compress:
                    _copy_compressed(file_path, repo_fpath, self._pgzip_threads, self._pgzip_block_size)
                else:
                    shutil.copy2(file_path, repo_fpath)
            except BaseException as error:
                # Something went wrong, perhaps out of space, or race condition due to lack of locking.
                # TODO (Min): proper handle the error and recover when we learn more here.
                sys.stderr.write(f"An exception occured: {repr(error)}\n")
                ref_count = self._add_ref(sha1_hash, False, compress)

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

        Throws:
            ValueError if sha1 is not found.
        """
        path = self._sha1_to_dir(sha1).joinpath(sha1)
        if not path.exists():
            # This is potentially valid case for the caller, we need to inform the
            # the caller about it.
            raise ValueError(f"Try to get SHA1 {sha1} but it is not found")
        # Directly return the object after loading it. This could be throw an
        # exception but that indicates some internal error since we should never
        # have stored the (invalid) object in the first place with the add() API.
        #
        # TODO (Min): we could also keep a stats in the meta data on how many
        #             times the object is read. Will add if that's needed.
        self._load_json_dict()
        if self._json_dict[sha1][COMP_KEY]:
            # Compressed. Because pgzip doesn't support tell() yet, we need to
            # uncompress into a temp file and return it.
            tmp = self._get_tmp_file_path()
            _copy_uncompressed(path, tmp, self._pgzip_threads, self._pgzip_block_size)
            obj = torch.load(tmp)
            tmp.unlink()
            return obj
        else:
            # Uncompressed.
            return torch.load(path)

    def delete(self, sha1: str) -> None:
        """Delete a SHA1

        Args:
            sha1 (str):
                SHA1 of the object to delete.

        Throws:
            ValueError if sha1 is not found.
        """
        path = self._sha1_to_dir(sha1).joinpath(sha1)
        if not path.exists():
            # This is potentially a valid case for the caller, we need to inform the
            # the caller about it.
            raise ValueError(f"Try to delete SHA1 {sha1} but it is not found")

        self._load_json_dict()

        assert sha1 in self._json_dict.keys(), "internal error: sha1 not found in json"
        entry = _get_json_entry(self._json_dict[sha1])

        assert entry[RF_KEY] > 0, f"ref count {entry[RF_KEY]} should be positive"
        entry[RF_KEY] -= 1
        if entry[RF_KEY] == 0:
            # Now, since ref count is 0 now deleting the object.
            path.unlink()  # We may leave behind an empty dir, which is OK.
            entry = {}  # Below, we remove the entry because of this.

        # Put the entry back and store it or delete it.
        if entry:
            self._json_dict[sha1] = entry
        else:
            # empty entry, it means this sha1 is deleted.
            del self._json_dict[sha1]
        self._store_json_dict()

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

    def _add_ref(self, current_sha1_hash: str, inc: bool, compressed: bool) -> int:
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
            entry = {}
        else:
            entry = self._json_dict[current_sha1_hash]
        entry = _get_json_entry(entry)

        # Update the ref count.
        entry[RF_KEY] += 1 if inc else -1
        assert entry[RF_KEY] >= 0, "negative ref count"

        # Update compressed flag.
        entry[COMP_KEY] = compressed

        self._json_dict[current_sha1_hash] = entry
        self._store_json_dict()

        return entry[RF_KEY]
