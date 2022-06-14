# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
from pathlib import Path
import sys
from typing import Union

from .pygit import PyGit
from .sha1_store import SHA1_store


class Repo:
    def __init__(self, parent_dir: Path, init: bool = False) -> None:
        """Features:
        1. Create the wgit directory if it does not exist.
        2. SHA1Store.init()
        3. Create SHA1 .wgit/sha1_refs.json
        3. Initialize a .git directory within the .wgit using `git init`.
        4. add a .gitignore within the .wgit directory, so that the git repo within will ignore `sha1_refs.json`
        """
        # If repo does not exist, creates a new wgit repo object with self.repo.path pointing to the path of repo
        # and notes all the internal files.
        # else, if repo already exists: create a pygit object from the .wgit/.git.
        self.wgit_parent = parent_dir
        self._repo_path: Union[None, Path] = None
        self._wgit_dir = Path(".wgit")
        self._metadata_file = Path(".wgit/checkpoint.pt")
        self._sha1_ref = Path(".wgit/sha1_refs.json")
        self._wgit_git_path = Path(".wgit/.git")
        self._sha1_store_path = Path(".wgit/sha1_store")

        if not self._exists() and init:
            # No weigit repo exists and is being initialized with init=True
            # Make .wgit directory, create sha1_refs and metadata file
            self._wgit_dir.mkdir(exist_ok=True)
            self._metadata_file.touch(exist_ok=False)
            self._sha1_ref.touch(exist_ok=False)

            # # Make the .wgit a git repo
            gitignore_files = [self._sha1_store_path.name, self._sha1_ref.name]
            self._pygit = PyGit(self.wgit_parent.joinpath(self._wgit_dir), gitignore=gitignore_files)

            # Initializing sha1_store only after wgit has been initialized!
            self._sha1_store = SHA1_store(self._wgit_dir, self._metadata_file, self._sha1_ref, init=True)
        elif self._exists() and init:
            # if weigit repo already exists and init is being called, wrap the existing .wgit/.git repo with PyGit
            self._sha1_store = SHA1_store(
                self._wgit_dir,
                self._metadata_file,
                self._sha1_ref,
            )
            self._pygit = PyGit(self.wgit_parent.joinpath(self._wgit_dir))

        elif self._exists() and not init:
            # weigit exists and non-init commands are triggered
            self._sha1_store = SHA1_store(
                self._wgit_dir,
                self._metadata_file,
                self._sha1_ref,
            )
            self._pygit = PyGit(self.wgit_parent.joinpath(self._wgit_dir))

        else:
            # weigit doesn't exist and is not trying to be initialized (triggers during non-init commands)
            sys.stderr.write("fatal: not a wgit repository!\n")

    def add(self, file_path: str) -> None:
        """Adds a file to the wgit repo"""
        if self._exists():
            self._sha1_store.add(file_path)  # add the filefile to the sha1_store
            self._pygit.add()  # add to the .wgit/.git repo

    def commit(self, message: str) -> None:
        """Commits staged changes to the repo"""
        if self._exists():
            self._pygit.commit(message)

    def status(self) -> None:
        """Skeleton"""
        if self._exists():
            print("wgit status")

    def log(self, file: str) -> None:
        """Returns the WeiGit log of commit history."""
        if self._exists():
            if file:
                print(f"wgit log of the file: {file}")
            else:
                print("wgit log")

    def checkout(self, sha1: str) -> None:
        """Checkout a previously commited version of the checkpoint"""
        if self._exists():
            print("wgit checkout: sha1")

    def compression(self) -> None:
        """Not Implemented: Compression functionalities"""
        print("Not Implemented!")

    def checkout_by_steps(self) -> None:
        """Not Implemented: Checkout by steps"""
        print("Not Implemented!")

    @property
    def path(self) -> str:
        """Get the path to the WeiGit repo"""
        if self._repo_path is None:
            self._exists()
        return str(self._repo_path)

    def _exists(self) -> bool:
        """Returns True if a valid wgit exists within the cwd, and sets the self._repo_path to the wgit path."""
        if self._weigit_repo_exists(self.wgit_parent):
            self._repo_path = self.wgit_parent.joinpath(".wgit")
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
