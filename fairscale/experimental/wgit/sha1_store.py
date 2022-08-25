# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, Optional, Tuple, Union, cast

import pgzip
import torch
from torch import Tensor

from fairscale.internal.containers import from_np, to_np

from .utils import ExitCode

#
# Const string keys for json file. Do not change for backward compatibilities.
#

# For each object entry in the metadata json file.
ENTRY_RF_KEY = "ref_count"  # int, reference count for this object.
ENTRY_COMP_KEY = "compressed"  # bool, is compressed or not.
ENTRY_OS_KEY = "original_size"  # int, original size for all identical objects mapped to this object.
ENTRY_DS_KEY = "deduped_size"  # int, size after deduplication (always enabled).
ENTRY_CS_KEY = "compressed_size"  # int, size after gzip compression, if enabled.
ENTRY_NAMES_KEY = "names"  # dict, names of objects and their count mapped to this object.

# For the entire store in the metadata json file.
STORE_CREATE_DATE_KEY = "created_on"  # str, when is the store created.
STORE_OS_KEY = "original_size"  # int, original size for all objects added.
STORE_DS_KEY = "deduped_size"  # int, size after deduplication (always enabled).
STORE_CS_KEY = "compressed_size"  # int, size after gzip compression, if enabled on any object within the store.


def _get_json_entry(d: Dict[str, Any]) -> Dict[str, Any]:
    """Get a dict from a json entry.

    This fills in any missing entries in case we load an older version
    json file from the disk.
    """
    for int_key_init_zero in [ENTRY_RF_KEY, ENTRY_OS_KEY, STORE_DS_KEY, ENTRY_CS_KEY]:
        if int_key_init_zero not in d.keys():
            d[int_key_init_zero] = 0

    for bool_key_init_false in [ENTRY_COMP_KEY]:
        if bool_key_init_false not in d.keys():
            d[bool_key_init_false] = False

    for dict_key_init_empty in [ENTRY_NAMES_KEY]:
        if dict_key_init_empty not in d.keys():
            d[dict_key_init_empty] = {}

    return d


def _copy_compressed(src: Path, dest: Path, thread: Optional[int], blocksize: int) -> Tuple[int, int]:
    """Helper to copy a file and compress it at the same time.

    Returns:
        (int, int):
            original size and compressed size in bytes.
    """
    with open(str(src), "rb") as srcf:
        with pgzip.open(str(dest), "wb", compresslevel=5, thread=thread, blocksize=blocksize) as destf:
            while True:
                buf = srcf.read(blocksize)
                if len(buf) == 0:
                    break
                destf.write(buf)
    orig, comp = Path(src).stat().st_size, Path(dest).stat().st_size
    assert orig >= comp or comp < 1 * 1024 * 1024, f"Compressed size {comp} > original {orig} for large data"
    return orig, comp


def _copy_uncompressed(src: Path, dest: Path, thread: Optional[int], blocksize: int) -> None:
    """Helper to copy a file and uncompress it at the same time."""
    with open(str(dest), "wb") as destf:
        with pgzip.open(str(src), "rb", thread=thread, blocksize=blocksize) as srcf:
            while True:
                buf = srcf.read(blocksize)
                if len(buf) == 0:
                    break
                destf.write(buf)


class _JSON_DictContext:
    """Helper class that handles syncing of a json and a dict."""

    def __init__(self, s: "SHA1_Store", readonly: bool) -> None:
        self._s = s
        self._readonly = readonly

    def __enter__(self) -> None:
        """Load from file."""
        assert self._s._json_dict is None
        if self._s._metadata_file_path.exists():
            with open(self._s._metadata_file_path, "r") as f:
                self._s._json_dict = json.load(f)
        else:
            self._s._json_dict = {}

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        """Store back to file."""
        assert isinstance(self._s._json_dict, dict)
        if not self._readonly:
            with open(self._s._metadata_file_path, "w", encoding="utf-8") as f:
                json.dump(self._s._json_dict, f, ensure_ascii=False, indent=2)
        self._s._json_dict = None


class SHA1_Store:
    """
    This class represents a SHA1 checksum based storage dir for state_dict
    and tensors.

    This means the same content will not be stored multiple times, resulting
    in space savings. (a.k.a. de-duplication)

    To make things easier for the callers, this class accept input data
    as files, state_dict or tensors. This class always returns in-memory
    data, not on-disk files. This class doesn't really care or know the actually
    data types.

    A key issue is dealing with content deletion. We use a reference counting
    algorithm, which means the caller must have symmetrical add/remove calls
    for each object.

    We used to support children-parent dependency graph and ref counting, but
    it is flawed since a grand-child can have the same SHA1 as the grand-parent,
    resulting in a cycle. This means caller must compute which parent is safe
    to delete in a version tracking graph. The lesson here is that content
    addressibility and dependency graphs do not mix well.

    We support multicore compression for the data to be store on per-object basis.
    We use pgzip to do parallel compression/decompression on top of it to use all
    the cores.

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
        self._sha1_buf_size = sha1_buf_size
        self._pgzip_threads = pgzip_threads
        self._pgzip_block_size = pgzip_block_size

        # Metadata related.
        self._metadata_file_path = self._path.joinpath("metadata.json")
        self._json_dict: Optional[Dict[str, Any]] = None
        self._json_ctx = _JSON_DictContext(self, readonly=False)
        self._readonly_json_ctx = _JSON_DictContext(self, readonly=True)

        # Initialize the store if not exist and if init is True.
        if init and not self._path.exists():
            try:
                Path.mkdir(self._path, parents=False, exist_ok=False)
            except FileExistsError as error:
                sys.stderr.write(f"An exception occured while creating Sha1_store: {repr(error)}\n")
                sys.exit(ExitCode.FILE_EXISTS_ERROR)
            # Create a new json file for this new store.
            with self._json_ctx:
                self._json_dict = {
                    STORE_CREATE_DATE_KEY: time.ctime(),
                    STORE_OS_KEY: 0,
                    STORE_DS_KEY: 0,
                    STORE_CS_KEY: 0,
                }

        # This is an internal error since caller of this our own wgit code.
        assert (
            self._path.exists() and self._metadata_file_path.exists()
        ), f"SHA1 store {self._path} does not exist and init is False"

        # Make sure there is a valid metadata file.
        with self._readonly_json_ctx:
            assert STORE_CREATE_DATE_KEY in self._json_dict, f"Invalid SHA1 Store in {self._path}"

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

    def add(self, file_or_obj: Union[Path, Tensor, Dict], compress: bool = True, name: str = None) -> str:
        """Adds a file/object to this store and the sha1 references accordingly.

        First, a sha1 hash is calculated. Utilizing the sha1 hash string, the actual file
        in <file_or_obj> is moved within the store and the reference file is updated.
        If the input is an object, it will be store in the self._tmp_dir and then moved.

        If compress is True, the stored file is also compressed, which is useful for tensors
        with a lot of zeros.

        We use pickle and numpy for saving, loading because it is more deterministic
        in terms of serialized bytes. They do lose info on device and dtype of
        tensors. Will handle those later.

        Args:
            file_or_obj (str or tensor or Dict):
                Path to the file to be added to the store or an in-memory object
                that can be handled by pickle. Note, OrderedDict is used when
                you call `state_dict()` on a nn.Module, and it is an instance
                of a Dict too. A model's state_dict can be a simple dict because
                it may contain both model state_dict and other non-tensor info.
            compress (bool, optional):
                Use gzip compression on this object or not.
                Default: True
            name (str, optional):
                Optional name for this object.
                Default: None
        """
        start = time.time()
        is_pickle_file = None

        # Use `isinstance` not `type() == Path` since pathlib returns OS specific
        # Path types, which inherit from the Path class.
        if isinstance(file_or_obj, (Path, str)):
            # Make sure it is a valid file.
            try:
                pickle.load(open(file_or_obj, "rb"))
                is_pickle_file = True
            except Exception as e:
                is_pickle_file = False
                pass
            file_path = Path(file_or_obj)
            remove_tmp = False

        if is_pickle_file is False:
            # Continue to support torch.save()'ed files too by loading it
            # in memory and the next if condition will pickle it.
            file_or_obj = torch.load(cast(Union[Path, str], file_or_obj))

        if isinstance(file_or_obj, (Tensor, Dict)):
            # Serialize the object into a tmp file.
            file_path = self._get_tmp_file_path()
            pickle.dump(to_np(file_or_obj), open(file_path, "wb"))
            remove_tmp = True
        else:
            assert False, f"incorrect input {type(file_or_obj)}"

        # Get SHA1 from the file.
        assert isinstance(file_path, Path), type(file_path)
        sha1_hash = self._get_sha1_hash(file_path)

        # Load json for many meta data operations below. Repeatedly loading
        # can be very slow.
        with self._json_ctx:

            # Add reference.
            ref_count = self._add_ref(sha1_hash, True, compress)

            if ref_count == 1:  # First time adding.
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
                        orig_size, comp_size = _copy_compressed(
                            file_path, repo_fpath, self._pgzip_threads, self._pgzip_block_size
                        )
                    else:
                        shutil.copy2(file_path, repo_fpath)
                        orig_size = comp_size = file_path.stat().st_size
                except BaseException as error:
                    # Something went wrong, perhaps out of space, or race condition due to lack of locking.
                    # TODO (Min): proper handle the error and recover when we learn more here.
                    sys.stderr.write(f"An exception occured: {repr(error)}\n")
                    ref_count = self._add_ref(sha1_hash, False, compress)

            # Update the sizes for this entry.
            entry = _get_json_entry(self._json_dict[sha1_hash])
            assert (
                ref_count == 1 or entry[ENTRY_OS_KEY] % (ref_count - 1) == 0
            ), f"incorrect size: {entry[ENTRY_OS_KEY]} and {ref_count}"
            o_diff = orig_size if ref_count == 1 else (entry[ENTRY_OS_KEY] // (ref_count - 1))
            d_diff = orig_size if ref_count == 1 else 0
            c_diff = comp_size if ref_count == 1 else 0
            entry[ENTRY_OS_KEY] += o_diff
            entry[ENTRY_DS_KEY] += d_diff
            entry[ENTRY_CS_KEY] += c_diff

            # Update whole store's stats.
            self._json_dict[STORE_OS_KEY] += o_diff
            self._json_dict[STORE_DS_KEY] += d_diff
            self._json_dict[STORE_CS_KEY] += c_diff

            # Update the name list for this entry.
            if name:
                if name not in entry[ENTRY_NAMES_KEY].keys():
                    entry[ENTRY_NAMES_KEY][name] = 1
                else:
                    entry[ENTRY_NAMES_KEY][name] += 1

        # Clean up if needed.
        if remove_tmp:
            file_path.unlink()

        duration = time.time() - start
        if duration > 60:
            logging.warning(f"Add() is taking long: {duration}s")
        return sha1_hash

    def get(self, sha1: str) -> Union[Tensor, Dict]:
        """Get data from a SHA1

        Args:
            sha1 (str):
                SHA1 of the object to get.

        Returns:
            (Tensor or Dict):
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
        with self._readonly_json_ctx:
            if self._json_dict[sha1][ENTRY_COMP_KEY]:
                # Compressed. Because pgzip doesn't support tell() yet, we need to
                # uncompress into a temp file and return it.
                tmp = self._get_tmp_file_path()
                _copy_uncompressed(path, tmp, self._pgzip_threads, self._pgzip_block_size)
                obj = pickle.load(open(tmp, "rb"))
                tmp.unlink()
            else:
                # Uncompressed.
                obj = pickle.load(open(path, "rb"))
        return from_np(obj)

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

        with self._json_ctx:
            assert sha1 in self._json_dict.keys(), "internal error: sha1 not found in json"
            entry = _get_json_entry(self._json_dict[sha1])

            assert entry[ENTRY_RF_KEY] > 0, f"ref count {entry[ENTRY_RF_KEY]} should be positive"
            entry[ENTRY_RF_KEY] -= 1
            if entry[ENTRY_RF_KEY] == 0:
                # Now, since ref count is 0 now deleting the object.
                path.unlink()  # We may leave behind an empty dir, which is OK.
                entry = {}  # Below, we remove the entry because of this.

            # Put the entry back and store it or delete it.
            if entry:
                self._json_dict[sha1] = entry
            else:
                # empty entry, it means this sha1 is deleted.
                del self._json_dict[sha1]

    def size_info(self, sha1: Optional[str] = None) -> Tuple[int, int, int]:
        """Return original, deduped, gzipped sizes for an entry or the store."""
        with self._readonly_json_ctx:
            if sha1:
                if sha1 not in self._json_dict.keys():
                    raise ValueError(f"SHA1 {sha1} not found")
                entry = self._json_dict[sha1]
                return entry[ENTRY_OS_KEY], entry[ENTRY_DS_KEY], entry[ENTRY_CS_KEY]
            return self._json_dict[STORE_OS_KEY], self._json_dict[STORE_DS_KEY], self._json_dict[STORE_CS_KEY]

    def names(self, sha1: str = None) -> Dict[str, int]:
        """Return the names dict for an object."""
        with self._readonly_json_ctx:
            if sha1 not in self._json_dict.keys():
                raise ValueError(f"SHA1 {sha1} not found")
            entry = self._json_dict[sha1]
            return entry[ENTRY_NAMES_KEY]

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
        fd, name = tempfile.mkstemp(dir=self._tmp_dir)
        os.close(fd)  # Must close this FD or unlink() won't be able to release the space of the file.
        return Path(name)

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
        # Init the entry if needed.
        if current_sha1_hash not in self._json_dict:
            entry = {}
        else:
            entry = self._json_dict[current_sha1_hash]
        entry = _get_json_entry(entry)

        # Update the ref count.
        entry[ENTRY_RF_KEY] += 1 if inc else -1
        assert entry[ENTRY_RF_KEY] >= 0, "negative ref count"

        # Update compressed flag.
        entry[ENTRY_COMP_KEY] = compressed

        self._json_dict[current_sha1_hash] = entry

        return entry[ENTRY_RF_KEY]
