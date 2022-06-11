# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import List

import pygit2


class PyGit:
    def __init__(
        self,
        parent_path: Path,
        gitignore: List = list(),
        name: str = "user",
        email: str = "user@email.com",
    ) -> None:
        """
        PyGit class to wrap the wgit/.git repo and interact with the git repo.

        Args:
        parent_path: Has to be the full path of the parent!
        """
        # Find if a git repo exists within .wgit repo:
        # If exists: then discover it and set the self.gitrepo path to its path
        self._parent_path = parent_path
        self.name = name
        self.email = email

        git_repo_found = pygit2.discover_repository(self._parent_path)

        if git_repo_found:
            # grab the parent dir of this git repo
            git_repo = Path(pygit2.discover_repository(self._parent_path))
            pygit_parent_p = git_repo.parent.absolute()

            # Check If the parent dir is a .wgit dir. If the .wgit is a git repo
            # just wrap the existing git repo with pygit2.Repository class
            if pygit_parent_p == self._parent_path:
                self.repo = pygit2.Repository(str(self._parent_path))
                self.path = self._parent_path.joinpath(".git")
                print("\nJUST Wrapping!\n")
            else:
                # if the parent is not a .wgit repo,
                # then the found-repo is a different git repo. Init new .wgit/.git
                print("\nFound but not .wgit - SO Creating!\n")
                self._init_wgit_git(gitignore)
        else:
            # no git repo found, make .wgit a git repo
            print("\nCreating from scratch\n")
            self._init_wgit_git(gitignore)

    def _init_wgit_git(self, gitignore: List) -> None:
        """Initializes a .git within .wgit directory, making it a git repo."""
        self.repo = pygit2.init_repository(str(self._parent_path), False)
        self.path = self._parent_path.joinpath(".git")

        # create and populate a .gitignore
        self._parent_path.joinpath(".gitignore").touch(exist_ok=False)

        with open(self._parent_path.joinpath(".gitignore"), "a") as file:
            for item in gitignore:
                file.write(f"{item}\n")

    def add(self) -> None:
        """git add all the untracked files not in gitignore, to the .wgit/.git repo."""
        # If .wgit is git repo, add all the files in .wgit not being ignored to git
        if self._exists:
            self.repo.index.add_all()
            self.repo.index.write()

    def commit(self, message: str) -> None:
        """git commit the staged changes to the .wgit/.git repo."""
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
        return self._parent_path == Path(self.repo.path).parent

    @property
    def _path(self) -> str:
        return self.repo.path

    def status(self) -> None:
        print(self.repo.status())
