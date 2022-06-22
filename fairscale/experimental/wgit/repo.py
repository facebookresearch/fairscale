# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import json
import pathlib
from pathlib import Path
import sys
from typing import Tuple, Union

from .pygit import PyGit
from .sha1_store import SHA1_store


class Repo:
    """
    Represents the WeiGit repo for tracking neural network weights and their versions.

    Args:
        parent_dir (pathlib.Path, str)
            The path to the parent directory where a weigit repo will be created. In the case a repo already exists, it will be wrapped with this class.
        init (bool, optional)
            - If ``True``, initializes a new WeiGit repo in the parent_dir. Initialization creates a `.wgit` directory within the <parent_dir>, triggers an initialization
                of a sha1_store in the ./<parent_dir>/.wgit directory, and makes the ./<parent_dir>/.wgit a git repository through git initialization.
            - If ``False``, a new WeiGit repo is not initialized and the existing repo is simply wrapped, populating the `wgit_parent` and other internal attributes.
            - Default: False
    """

    def __init__(self, parent_dir: Union[Path, str] = Path.cwd(), init: bool = False) -> None:
        """initialize a weigit repo: Subsequently, also initialize a sha1_store and a pygit git repo within as
        part of the weigit initialization process"""
        # If repo does not exist, creates a new wgit repo object with self.repo.path pointing to the path of repo
        # and notes all the internal files.
        # else, if repo already exists: create a pygit object from the .wgit/.git.
        self.wgit_parent = Path(parent_dir)
        self._repo_path: Union[None, Path] = None
        self._wgit_git_path = Path(".wgit/.git")
        self._sha1_store_path = Path(".wgit/sha1_store")

        exists = self._exists(self.wgit_parent)
        if not exists and init:
            # No weigit repo exists and is being initialized with init=True
            # Make .wgit directory, create sha1_refs
            weigit_dir = self.wgit_parent.joinpath(".wgit")
            weigit_dir.mkdir(parents=False, exist_ok=True)

            # Initializing sha1_store only after wgit has been initialized!
            self._sha1_store = SHA1_store(weigit_dir, init=True)

            # # Make the .wgit a git repo
            gitignore_files = [self._sha1_store_path.name, self._sha1_store.ref_file_path.name]
            self._pygit = PyGit(weigit_dir, gitignore=gitignore_files)

        elif exists and init:
            # if weigit repo already exists and init is being called, wrap the existing .wgit/.git repo with PyGit
            self._sha1_store = SHA1_store(self.path)
            self._pygit = PyGit(self.path)

        elif exists and not init:
            # weigit exists and non-init commands are triggered
            self._sha1_store = SHA1_store(self.path)
            self._pygit = PyGit(self.path)

        else:
            # weigit doesn't exist and is not trying to be initialized (triggers during non-init commands)
            sys.stderr.write("fatal: not a wgit repository!\n")

    def add(self, in_file_path: str) -> None:
        """
        Adds a file to the wgit repo.

        Args:
            file_path (str)
                Path to the file to be added to the weigit repo
        """
        if self._exists(self.wgit_parent):
            # create the corresponding metadata file
            file_path = Path(in_file_path)
            metadata_file, parent_sha1 = self._process_metadata_file(file_path.name)

            # add the file to the sha1_store
            sha1_hash = self._sha1_store.add(file_path, parent_sha1)

            # write metadata to the metadata-file
            self._write_metadata(metadata_file, sha1_hash)
            self._pygit.add()  # add to the .wgit/.git repo
        else:
            sys.stderr.write("fatal: no wgit repo exists!\n")

    def commit(self, message: str) -> None:
        """
        Commits staged changes to the repo.

        Args:
            message (str)
                The commit message
        """
        if self._exists(self.wgit_parent):
            self._pygit.commit(message)
        else:
            sys.stderr.write("fatal: no wgit repo exists!\n")

    def status(self) -> None:
        """Show the state of the working tree."""
        if self._exists(self.wgit_parent):
            print("wgit status")
        else:
            sys.stderr.write("fatal: no wgit repo exists!\n")

    def log(self, file: str) -> None:
        """
        Returns the WeiGit log of commit history.

        Args:
            file (str, optional)
                Show the log of the commit history of the repo. Optionally, show the log history of a specific file.
        """
        if self._exists(self.wgit_parent):
            if file:
                print(f"wgit log of the file: {file}")
            else:
                print("wgit log")
        else:
            sys.stderr.write("fatal: no wgit repo exists!\n")

    def checkout(self, sha1: str) -> None:
        """
        Checkout a previously commited version of the checkpoint.

        Args:
            sha1 (str) The sha1 hash of the file version to checkout.
        """
        raise NotImplementedError

    def compression(self) -> None:
        """Not Implemented: Compression functionalities"""
        raise NotImplementedError

    def checkout_by_steps(self) -> None:
        """Not Implemented: Checkout by steps"""
        raise NotImplementedError

    @property
    def path(self) -> Path:
        """Get the path to the WeiGit repo"""
        if self._repo_path is None:
            self._exists(self.wgit_parent)
        return self._repo_path

    def _process_metadata_file(self, metadata_fname: str) -> Tuple[Path, str]:
        """Create a metadata_file corresponding to the file to be tracked by weigit if the first version of the file is encountered.
        If a version already exists, open the file and get the sha1_hash of the last version as parent_sha1"""
        metadata_file = self.path.joinpath(metadata_fname)
        if not metadata_file.exists() or not metadata_file.stat().st_size:
            metadata_file.touch()
            parent_sha1 = "ROOT"
        else:
            with open(metadata_file, "r") as f:
                ref_data = json.load(f)
            parent_sha1 = ref_data["SHA1"]["__sha1_full__"]
        return metadata_file, parent_sha1

    def _write_metadata(self, metadata_file: Path, sha1_hash: str) -> None:
        """Write metadata to the metadata file file"""
        change_time = Path(metadata_file).stat().st_ctime
        metadata = {
            "SHA1": {
                "__sha1_full__": sha1_hash,
            },
            "last_modified_time_stamp": change_time,
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

    def _exists(self, check_dir: Path) -> bool:
        """Returns True if a valid wgit exists within the cwd and iteratively checks to the root directory and
        sets the self._repo_path attribute to the wgit path.

        Args:
           check_dir (Path)
               path to the directory from where search is started.
        """
        if self._weigit_repo_exists(check_dir):
            self._repo_path = check_dir.joinpath(".wgit")
        else:
            root = Path(check_dir.parts[0])
            while check_dir != root:
                check_dir = check_dir.parent
                if self._weigit_repo_exists(check_dir):
                    self._repo_path = check_dir.joinpath(".wgit")
                    break
        return True if self._repo_path is not None else False

    def _weigit_repo_exists(self, check_dir: pathlib.Path) -> bool:
        """Returns True if a valid WeiGit repo exists in the path: check_dir"""
        wgit_exists, sha1_refs, git_exists, gitignore_exists = self._weight_repo_file_check(check_dir)
        return wgit_exists and sha1_refs and git_exists and gitignore_exists

    def _weight_repo_file_check(self, check_dir: Path) -> tuple:
        """Returns a tuple of boolean corresponding to the existence of each .wgit internally required files."""
        wgit_exists = check_dir.joinpath(".wgit").exists()
        sha1_refs = check_dir.joinpath(".wgit/sha1_refs.json").exists()
        git_exists = check_dir.joinpath(".wgit/.git").exists()
        gitignore_exists = check_dir.joinpath(".wgit/.gitignore").exists()
        return wgit_exists, sha1_refs, git_exists, gitignore_exists
