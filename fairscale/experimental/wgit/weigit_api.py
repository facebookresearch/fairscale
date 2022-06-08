# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
import pathlib
from pathlib import Path
import shutil
import sys
import typing

import pygit2

from fairscale.experimental.wgit.pygit import PyGit
from fairscale.experimental.wgit.sha1_store import SHA1_store
from fairscale.experimental.wgit.utils import ExitCode, create_dir, create_file, get_sha1_hash, write_to_file


class WeiGitRepo:
    """
    Features:
    1. Create the wgit directory if it does not exist.
    2. SHA1Store.init()
    3. Create SHA1 .wgit/sha1_refs.json
    3. Initialize a .git directory within the .wgit using `git init`.
    4. add a .gitignore within the .wgit directory, so that the git repo within will ignore `sha1_refs.json`
    """

    def __init__(self) -> None:
        # If repo does not exist, creates a new wgit repo object with self.repo.path pointing to the path of repo
        # and notes all the internal files.
        # else, if repo already exists: create a pygit object from the .wgit/.git.
        self._cwd = Path.cwd()
        self._repo_path: typing.Union[None, Path] = None
        self._wgit_dir = Path(".wgit")
        self._metadata_file = Path(".wgit/checkpoint.pt")
        self._sha1_ref = Path(".wgit/sha1_refs.json")
        self._wgit_git_path = Path(".wgit/.git")
        self._sha1_store = Path(".wgit/sha1_store")

        if not self._exists():
            # Make .wgit directory, create sha1_refs and metadata file
            create_dir(dir_path=self._wgit_dir, exception_msg="An exception occured: WeiGit already Initialized")
            create_file(file_path=self._metadata_file)
            create_file(file_path=self._sha1_ref)

            # Make the .wgit a git repo
            try:
                pygit2.init_repository(str(self._wgit_git_path), False)
                self.pygit = PyGit(self._cwd.joinpath(self._wgit_dir))
                create_file(self._wgit_dir.joinpath(".gitignore"))
                write_to_file(
                    self._wgit_dir.joinpath(".gitignore"), msg=f"{self._sha1_ref.name}\n{self._sha1_store.name}"
                )
            except pygit2.GitError as error:
                sys.stderr.write(f"An exception occurred while initializing .wgit/.git: {repr(error)}\n")
                sys.exit(ExitCode.ERROR)

            # Initializing sha1_store only after wgit has been initialized!
            SHA1_store()
        else:
            self.pygit = PyGit(self._cwd.joinpath(self._wgit_dir))

    def add(self, file_path: str) -> None:
        if self._exists():
            # file_name = os.path.basename(file_path)
            sha1_hash = get_sha1_hash(file_path)

            # use the sha1_has to create a directory with first2 sha naming convention
            try:
                repo_fdir = os.path.join(self.path, "sha1_store", sha1_hash[:2]) + "/"
                os.makedirs(repo_fdir)
            except FileExistsError as error:
                sys.stderr.write(f"An exception occured: {repr(error)}\n")
                sys.exit(ExitCode.FILE_EXISTS_ERROR)
            try:
                # First transfer the file to the internal sha1_store
                repo_fpath = os.path.join(repo_fdir, sha1_hash[2:])
                shutil.copy2(file_path, os.path.join(repo_fdir, sha1_hash[2:]))
                change_time = Path(repo_fpath).stat().st_ctime

                # Create the dependency Graph and track reference
                sha1_store = SHA1_store()
                sha1_store.add_ref(current_sha1_hash=sha1_hash)
                metadata = {
                    "SHA1": {
                        "__sha1_full__": sha1_hash,
                    },
                    "file_path": repo_fpath,
                    "time_stamp": change_time,
                }
                # Populate the meta_data file with the meta_data and git add
                self._add_metadata_to_json(metadata)
                self.pygit.add()

            except BaseException as error:
                # Cleans up the sub-directories created to store sha1-named checkpoints
                sys.stderr.write(f"An exception occured: {repr(error)}\n")
                shutil.rmtree(repo_fdir)

    def commit(self, message: str) -> None:
        """Commits staged changes to the repo"""
        if self._exists():
            self.pygit.commit(message)

    def status(self) -> None:
        """Skeleton"""
        if self._exists():
            print("wgit status")

    def log(self, file: str) -> None:
        """Skeleton"""
        if self._exists():
            if file:
                print(f"wgit log of the file: {file}")
            else:
                print("wgit log")

    def checkout(self) -> None:
        """Skeleton"""
        if self._exists():
            print("wgit checkout")

    def compression(self) -> None:
        print("Not Implemented!")

    def checkout_by_steps(self) -> None:
        print("Not Implemented!")

    @property
    def path(self) -> str:
        if self._repo_path is None:
            self._exists()
        return str(self._repo_path)

    def _add_metadata_to_json(self, metadata: dict) -> None:
        """Populates the meta_data_file: checkpoint.pt with the meta_data"""
        file_pt_json = self._metadata_file
        with open(file_pt_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

    def _exists(self) -> bool:
        """
        Returns True if a valid wgit exists within the cwd, and sets the self._repo_path to the wgit path.
        """
        if self._weigit_repo_exists(self._cwd):
            self._repo_path = self._cwd.joinpath(".wgit")
        return True if self._repo_path is not None else False

    def _weigit_repo_exists(self, check_dir: pathlib.Path) -> bool:
        wgit_exists, sha1_refs, git_exists, gitignore_exists = self._weigit_repo_status(check_dir)
        return wgit_exists and sha1_refs and git_exists and gitignore_exists

    def _weigit_repo_status(self, check_dir: Path) -> tuple:
        """
        returns the state of the weigit repo and checks if all the required files are present.
        """
        wgit_exists = check_dir.joinpath(".wgit").exists()
        sha1_refs = check_dir.joinpath(".wgit/sha1_refs.json").exists()
        git_exists = check_dir.joinpath(".wgit/.git").exists()
        gitignore_exists = check_dir.joinpath(".wgit/.gitignore").exists()
        return wgit_exists, sha1_refs, git_exists, gitignore_exists
