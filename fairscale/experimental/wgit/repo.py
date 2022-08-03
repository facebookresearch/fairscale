# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from .pygit import PyGit
from .sha1_store import SHA1_Store

# This is a fixed dir name we use for sha1_store. It should not be changed
# for backward compatibility reasons.
SHA1_STORE_DIR_NAME = "sha1_store"


# These are on-disk keys. Don't modify for backward compatibility.
SHA1_KEY = "SHA1"
LAST_MODIFIED_TS_KEY = "last_modified_time_stamp"
REL_PATH_KEY = "file_path"  # this will be removed from the json since it is redundant.


class RepoStatus(Enum):
    """Repo Statuses"""

    CLEAN = 1
    CHANGES_NOT_ADDED = 2
    CHANGES_ADDED_NOT_COMMITED = 3


@dataclass
class SizeInfo:
    """Size info for a file or the repo in bytes.

    Deduped size can't be disabled. So it is always performed.

    Both sparsified and gzipped are optional. They are applied in the following
    order if both are enabled:

        sparsify -> gzip

    Therefore, original >= deduped >= sparsified >= gzipped
    """

    original: int
    deduped: int
    sparsified: int
    gzipped: int


@dataclass
class _SHA1_Tensor:
    """Representing a tensor using sha1(s) from SHA1 store.

    It can be either a dense one or two sparse one (SST and DST).
    """

    is_dense: bool = True
    dense_sha1: str = ""
    sst_sha1: str = ""
    dst_sha1: str = ""


def _recursive_apply_to_elements(data: Union[List[Any], Dict[str, Any]], fn: Any, names: List[str]) -> None:
    """Helper function to traverse a dict recursively and apply a function to leafs.


    Args:
        data (dict or list):
            A dict or a list and it should only contain dict and list.
        fn (Any):
            A call back function on each element. Signature:
                fn(element: Any, names: List[str]) -> Any
        names (list):
            Stack of names for making the element path.
    """
    if isinstance(data, list):
        for i, _ in enumerate(data):
            names.append(str(i))
            if isinstance(data[i], (list, dict)):
                _recursive_apply_to_elements(data[i], fn, names)
            else:
                data[i] = fn(data[i], names)
            names.pop()
    elif isinstance(data, dict):
        for key in data.keys():
            names.append(str(key))
            if isinstance(data[key], (list, dict)):
                _recursive_apply_to_elements(data[key], fn, names)
            else:
                data[key] = fn(data[key], names)
            names.pop()
    else:
        assert False, f"Unexpected data type: {type(data)}"


class Repo:
    """
    Represents the WeiGit repo for tracking neural network weights and their versions.

    A WeiGit repo is like a git repo. It is a dir, in which a .wgit dir exists to keep
    track of the content.

    Args:
        parent_dir (Path, str):
            Parent dir in which to make or to load a .wgit dir.
            Default: "", which means CWD.
        init (bool, optional):
            - If ``True``, initializes a new WeiGit repo in the parent_dir. Initialization
              creates a `.wgit` directory within the <parent_dir>, triggers an initialization.
              of a sha1_store in the ./<parent_dir>/.wgit directory, and makes the
              ./<parent_dir>/.wgit a git repository through git initialization.
            - If ``False``, a new WeiGit repo is not initialized and the existing repo is
              wrapped, populating the `_wgit_parent` and other internal attributes.
            - Default: False
    """

    def __init__(self, parent_dir: Union[Path, str] = "", init: bool = False) -> None:
        # Set _wgit_parent.
        self._wgit_parent = Path(parent_dir if parent_dir != "" else Path.cwd())

        # Set _dot_wgit_dir_path.
        self._dot_wgit_dir_path: Optional[Path] = None
        exists = self._recursive_search_and_may_init_dot_wgit_dir_path(self._wgit_parent)

        if not exists and init:
            # No weigit repo exists and is being initialized with init=True
            # Make .wgit directory, create sha1_store
            self._dot_wgit_dir_path = self._wgit_parent.joinpath(".wgit")
            self._dot_wgit_dir_path.mkdir(parents=False, exist_ok=True)

            # Initializing sha1_store only after wgit has been initialized!
            self._sha1_store = SHA1_Store(self._dot_wgit_dir_path.joinpath(SHA1_STORE_DIR_NAME), init=True)

            # Create a git repo for the metadata versioning.
            self._pygit = PyGit(self._dot_wgit_dir_path, gitignore=[SHA1_STORE_DIR_NAME])

        elif exists:
            # Weigit repo already exists, populate this object.
            assert self._dot_wgit_dir_path is not None
            self._sha1_store = SHA1_Store(self._dot_wgit_dir_path.joinpath(SHA1_STORE_DIR_NAME))
            self._pygit = PyGit(self._dot_wgit_dir_path)

        else:
            # weigit doesn't exist and is not trying to be initialized (triggers
            # during non-init commands)
            sys.stderr.write("fatal: not a wgit repository!\n")
            sys.exit(1)

        # We are done init. Do a check.
        self._sanity_check()

    def _recursive_search_and_may_init_dot_wgit_dir_path(self, check_dir: Path) -> bool:
        """Search for a wgit repo top level dir from potentiall a subdir of a repo.

            This may set the self._dot_wgit_dir_path if a repo is found.

        Args:
           check_dir (Path):
               Path to the directory from where search is started.

        Returns:
           Returns True if a repo is found.
        """
        assert self._dot_wgit_dir_path is None, f"_dot_wgit_dir_path is already set to {self._dot_wgit_dir_path}"
        if self._weigit_repo_exists(check_dir):
            self._dot_wgit_dir_path = check_dir.joinpath(".wgit")
        else:
            root = Path(check_dir.parts[0])
            while check_dir != root:
                check_dir = check_dir.parent
                if self._weigit_repo_exists(check_dir):
                    self._dot_wgit_dir_path = check_dir.joinpath(".wgit")
                    break
        return True if self._dot_wgit_dir_path is not None else False

    def _weigit_repo_exists(self, check_dir: Path) -> bool:
        """Returns True if a valid WeiGit repo exists in the path: check_dir."""
        wgit_exists, git_exists, gitignore_exists = self._weigit_repo_file_check(check_dir)
        return wgit_exists and git_exists and gitignore_exists

    def _weigit_repo_file_check(self, check_dir: Path) -> tuple:
        """Returns a tuple of boolean corresponding to the existence of each
        .wgit internally required files.
        """
        wgit_exists = check_dir.joinpath(".wgit").exists()
        git_exists = check_dir.joinpath(".wgit/.git").exists()
        gitignore_exists = check_dir.joinpath(".wgit/.gitignore").exists()
        return wgit_exists, git_exists, gitignore_exists

    def _sanity_check(self) -> None:
        """Helper to check if on-disk state matches what we expect."""
        if not self._weigit_repo_exists(self._wgit_parent):
            sys.stderr.write("fatal: no wgit repo exists!\n")
            sys.exit(1)

    def add(
        self,
        in_file_path: str,
        per_tensor: bool = True,
        gzip: bool = True,
        sparsify: bool = False,
        sparsify_policy: Any = None,
    ) -> Optional[Dict[Any, Any]]:
        """Add a file to the wgit repo.

        This could a new file or a modified file. Adding an unmodified, existing file
        is allowed but it is a noop.

        Args:
            in_file_path (str):
                Path to the file to be added.
            per_tensor (bool, optional):
                Add a file in a per-tensor fashion. This enables more deduplication
                due to tensors being identical. Deduplication cannot be disabled
                completely because we use a content addressable SHA1_Store class.
                Default: True
            gzip (bool, optional):
                Enable gzip based lossless compression on the object being added.
                Default: True
            sparsify (bool, optional):
                Enable sparsify for the tensors, which is going to modify the values
                for all or some tensors, i.e. lossy compression.
                Default: False
            sparsify_policy (Any):
                TODO (Min): need to add a callback function to control which tensors
                            and how to sparsify.
                Default: None

        Returns:
            (Dict, optional)
                None if the content is added but not modified with lossy compression.
                Otherwise, returns a state_dict that contains the modified Tensors to
                be loaded back into the model, which means the tensors are dense, not
                SST and DST tensors.
        """
        self._sanity_check()

        if sparsify and not per_tensor:
            raise ValueError("Only support sparsity when per_tensor is true")

        # Create the corresponding metadata file or load it if the file is
        # not a newly added file.
        file_path = Path(in_file_path)
        rel_file_path = self._rel_file_path(file_path)
        metadata_file = self._process_metadata_file(rel_file_path)

        # Add the file to the sha1_store.
        ret_state_dict = None
        file_path_or_state_dict: Union[Path, Dict] = file_path
        # TODO (Min): We don't add parent sha1 tracking to sha1 store due to
        #             de-duplication & dependency tracking can create cycles.
        #             We need to figure out a way to handle deletion.
        # TODO (Min): We don't detect changes and compute delta on a modified file
        #             yet. Need to figure out a method for delta tracking.
        if per_tensor:

            def fn(element: Any, names: List[str]) -> Any:
                """Callback on each leaf object for _recursive_apply_to_elements below."""
                if isinstance(element, Tensor):
                    if sparsify:
                        # TODO (Min): here we will optionally do SST/DST and add those
                        #             tensors with sparsity.
                        #             Remember to update ret_state_dict
                        raise NotImplementedError()
                    sha1 = self._sha1_store.add(element, compress=gzip, name=".".join(names))
                    return _SHA1_Tensor(is_dense=True, dense_sha1=sha1)
                else:
                    return element

            state_dict = torch.load(file_path)
            ret_state_dict = copy.deepcopy(state_dict)  # This is only a temporary addition for testing.
            _recursive_apply_to_elements(state_dict, fn, [])
            file_path_or_state_dict = state_dict

        # Add this top-level object.
        sha1 = self._sha1_store.add(file_path_or_state_dict, compress=gzip)

        # write metadata to the metadata-file
        self._write_metadata(metadata_file, file_path, sha1)
        self._pygit.add()  # add to the .wgit/.git repo

        return ret_state_dict

    def commit(self, message: str) -> None:
        """Commits staged changes to the repo.

        Args:
            message (str):
                The commit message to be added.
        """
        self._sanity_check()

        # TODO (Min): make commit message a json for better handling of metadata like step count,
        #             LR, sparsity level, etc.
        self._pygit.commit(message)

    def size_info(self, path: Optional[str] = None) -> SizeInfo:
        """Get size info for a file or the whole repo.

        For the whole repo, just call size_info from sha1_store.

        For a file, needs to open the metadata and find the sha1 and then
        for per_tensor state_dict, collect size_info on all objects.

        TODO (Min): not exactly clear it is easy to compute this with
                    delta encoding, deduplication between objects, this
                    is possible to compute precisely.

        Args:
            path (str, optional):
                File path for the query. If None, return whole repo's info.
                Default: None

        Returns:
            (SizeInfo):
                The dataclass that contains the size info.
        """
        raise NotImplementedError()

    def status(self) -> Dict[str, RepoStatus]:
        """Show the state of the weigit working tree.

        State can be
          1. dirty with changes/modifications not added to weigit repo.
          2. dirty with a file changes added but not committed.
          3. clean and tracking files after a change has been committed,
             or clean with with an empty repo.

        TODO (Min): this needs to return repo status and dirty files and untracked
                    files too.
        Returns:
            (dict):
                A dict keyed with files and their status.
        """
        self._sanity_check()

        pygit_status = self._pygit.status()
        status = self._get_metdata_files()
        if status:
            out_status = dict()
            for metadata_file, is_modified in status.items():
                # if metadata_file is among the keys of pygit_status dict, it has not been commited to git yet.
                if is_modified:
                    out_status[str(metadata_file)] = RepoStatus.CHANGES_NOT_ADDED
                elif not is_modified and metadata_file in pygit_status.keys():
                    out_status[str(metadata_file)] = RepoStatus.CHANGES_ADDED_NOT_COMMITED
                elif not is_modified and metadata_file not in pygit_status.keys():
                    out_status[str(metadata_file)] = RepoStatus.CLEAN
            return out_status
        else:  # if status dict is empty, nothing has been added so far.
            return {"": RepoStatus.CLEAN}  # sub case of case-3, clean with an empty repo

    def log(self, file: str) -> None:
        """Returns the WeiGit log of commit history.

        Args:
            file (str, optional):
                Show the log of the commit history of the repo. Optionally, show
                the log history of a specific file.
        """
        self._sanity_check()

        # TODO (Min): this should return a list of sha1 for the history as well as
        #             each commit's message, which could be a dict from json commit msg.
        if file:
            print(f"wgit log of the file: {file}")
        else:
            print("wgit log")

    def checkout(self, sha1: str) -> None:
        """Checkout a previously commited version of the checkpoint.

        Args:
            sha1 (str):
               The sha1 hash of the file version to checkout.
        """
        self._sanity_check()
        raise NotImplementedError()

    def checkout_by_steps(self) -> None:
        """Not Implemented: Checkout by step count of the train process"""
        self._sanity_check()
        raise NotImplementedError()

    def _get_metdata_files(self) -> Dict[str, bool]:
        """Walk the directories that contain the metadata files and check the
        status of those files, whether they have been modified or not.

        Dict[str, bool] is a path in string and whether the file is_modified.
        """
        metadata_d = dict()
        for file in self._dot_wgit_dir_path.iterdir():  # iterate over the .wgit directory
            # exclude all the .wgit files and directory
            if file.name not in {"sha1_store", ".git", ".gitignore"}:
                # perform a directory walk on the metadata_file directories to find the metadata files
                for path in file.rglob("*"):
                    if path.is_file():
                        rel_path = str(path.relative_to(self._dot_wgit_dir_path))  # metadata path relative to .wgit dir
                        metadata_d[rel_path] = self._is_file_modified(path)
        return metadata_d

    def _is_metadata_file(self, file: Path) -> bool:
        """Checks whether a file is a valid metadata file by matching keys and
        checking if it has valid json data.
        """
        try:
            with open(file) as f:
                metadata = json.load(f)
            is_metadata = set(metadata.keys()) == {SHA1_KEY, LAST_MODIFIED_TS_KEY, REL_PATH_KEY}
        except json.JSONDecodeError:
            return False  # not a json file, so not valid metadata file
        return is_metadata

    def _is_file_modified(self, file: Path) -> bool:
        """Checks whether a file has been modified since its last recorded modification
        time recorded in the metadata_file.
        """
        with open(file) as f:
            data = json.load(f)
        # Get the last modified timestamp recorded by weigit and the current modified
        # timestamp. If not the same, then file has been modified since last weigit
        # updated metadata.
        last_mod_timestamp = data[LAST_MODIFIED_TS_KEY]
        curr_mod_timestamp = Path(data[REL_PATH_KEY]).stat().st_mtime
        return not curr_mod_timestamp == last_mod_timestamp

    def _process_metadata_file(self, metadata_fname: Path) -> Path:
        """Create a metadata_file corresponding to the file to be tracked by weigit if
        the first version of the file is encountered. If a version already exists, open
        the file and get the sha1_hash of the last version as parent_sha1.
        """
        metadata_file = self._dot_wgit_dir_path.joinpath(metadata_fname)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)  # create parent dirs for metadata file

        if not metadata_file.exists() or not metadata_file.stat().st_size:
            metadata_file.touch()
        else:
            with open(metadata_file, "r") as f:
                ref_data = json.load(f)
        return metadata_file

    def _write_metadata(self, metadata_file: Path, file_path: Path, sha1: str) -> None:
        """Write metadata to the metadata file"""
        change_time = Path(file_path).stat().st_mtime
        metadata = {
            SHA1_KEY: sha1,
            LAST_MODIFIED_TS_KEY: change_time,
            REL_PATH_KEY: str(file_path),
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

    def _rel_file_path(self, filepath: Path) -> Path:
        """Find the relative part to the filepath from the current working
        directory and return the relative path.
        """
        # get the absolute path
        filepath = filepath.resolve()
        # using zipped loop we get the path common to the filepath and cwd
        for i, (x, y) in enumerate(zip(filepath.parts, Path.cwd().parts)):
            pass
        # return the relative part (path not common to cwd)
        return Path(*filepath.parts[i:])
