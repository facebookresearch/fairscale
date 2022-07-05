# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Tuple

import pygit2


class PyGit:
    """
    PyGit class represents a git repo within a weigit repo.

    Args:
        parent_path (pathlib.Path)
            Has to be the full path of the parent!
        gitignore (List)
            a list of files to be added to the .gitignore
        name (str)
            Name of the author of the git repo. Optionally used if it can't be determined from user's .gitconfig.
        email (str)
            email address of the author of the git repo
    """

    def __init__(
        self,
        parent_path: Path,
        gitignore: List = list(),
        name: str = "user",
        email: str = "user@email.com",
    ) -> None:

        # Find if a git repo exists within .wgit repo:
        # If exists: then discover it and set the self.gitrepo path to its path
        self._parent_path = parent_path
        git_repo_found = pygit2.discover_repository(self._parent_path)

        # If gitconfig file exists use the name and email from the file
        self.name, self.email = self._set_author_config(name, email)

        if git_repo_found:
            # grab the parent dir of this git repo
            git_repo = Path(pygit2.discover_repository(self._parent_path))
            pygit_parent_p = git_repo.parent.absolute()

            # Check If the parent dir is a .wgit dir. If the .wgit is a git repo
            # just wrap the existing git repo with pygit2.Repository class
            if pygit_parent_p == self._parent_path:
                self.repo = pygit2.Repository(str(self._parent_path))
                self.path = self._parent_path.joinpath(".git")
            else:
                # if the parent is not a .wgit repo,
                # then the found-repo is a different git repo. Init new .wgit/.git
                self._init_wgit_git(gitignore)
        else:
            # no git repo found, make .wgit a git repo
            self._init_wgit_git(gitignore)

    def _init_wgit_git(self, gitignore: List) -> None:
        """
        Initializes a .git within .wgit directory, making it a git repo.

        Args:
            gitignore (List)
                a list of file paths to be ignored by the wgit git repo.
        """
        self.repo = pygit2.init_repository(str(self._parent_path), False)
        self.path = self._parent_path.joinpath(".git")

        # create and populate a .gitignore
        self._parent_path.joinpath(".gitignore").touch(exist_ok=False)

        with open(self._parent_path.joinpath(".gitignore"), "a") as file:
            for item in gitignore:
                file.write(f"{item}\n")

    def add(self) -> None:
        """
        git add all the untracked files not in gitignore, to the .wgit/.git repo.
        """
        # If .wgit is git repo, add all the files in .wgit not being ignored to git
        # TODO: Add functionalities for add specific files and add all files.
        if self._exists:
            self.repo.index.add_all()
            self.repo.index.write()
        else:
            sys.stderr.write("fatal: git repo does not exist")

    def commit(self, message: str) -> None:
        """
        git commit the staged changes to the .wgit/.git repo.

        Args:
            message (str)
                Commit message
        """
        # If .wgit is git repo, commit the staged files to git
        if self._exists:
            # if no commit exists, set ref to HEAD and parents to empty
            try:
                ref = self.repo.head.name
                parents = [self.repo.head.target]
            except pygit2.GitError:
                ref = "HEAD"
                parents = []

            author = pygit2.Signature(self.name, self.email)
            committer = pygit2.Signature(self.name, self.email)
            tree = self.repo.index.write_tree()
            self.repo.create_commit(ref, author, committer, message, tree, parents)

    @property
    def _exists(self) -> bool:
        """returns True if wgit is a git repository"""
        return self._parent_path == Path(self.repo.path).parent

    @property
    def _path(self) -> str:
        """returns the path of the git repository PyGit is wrapped around"""
        return self.repo.path

    def status(self) -> Dict:
        """Gathers the status of the git repo within wgit and returns a dictionary detailing the status.
        The dictionary contains the relative paths of the metadata files as keys and the values represent
        the status of the file in the form of an int number as status codes. These status codes are
        elaborated within PyGit2's documentation: https://www.pygit2.org/index_file.html#status and
        https://github.com/libgit2/pygit2/blob/320ee5e733039d4a3cc952b287498dbc5737c353/src/pygit2.c#L312-L320

        Returns: {"relative path to a file" : pygit2 status codes}
        """
        status_dict = self.repo.status()
        tracking_dict = dict(filter(lambda item: item[1] != pygit2.GIT_STATUS_IGNORED, status_dict.items()))
        return tracking_dict

    def _set_author_config(self, name: str, email: str) -> Tuple[str, str]:
        """Set the name and email for the pygit repo collecting from the gitconfig.
        If not available in gitconfig, set the values from the passed arguments."""
        gitconfig = Path("~/.gitconfig").expanduser()
        # parse the .gitconfig file for name and email
        try:
            set_name = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True).stdout.rstrip()
            set_email = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True).stdout.rstrip()
            if not set_name or not set_email:
                set_name = name
                set_email = email
        except BaseException:
            set_name = name
            set_email = email
        return set_name, set_email
