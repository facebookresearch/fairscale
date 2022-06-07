# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pygit2


class PyGit:
    """
    PyGit class to wrap the wgit/.git repo and interacta with the git repo.
    """

    def __init__(self, wgit_path) -> None:
        # Find if a git repo exists within .wgit repo:
        # If exists: then discover it and set the self.gitrepo path to its path
        self.exists = None
        self.wgit_path = wgit_path
        self.path = Path(pygit2.discover_repository(self.wgit_path))

        pygit_parent_p = self.path.parent.absolute()
        if pygit_parent_p != self.wgit_path:
            self.exists = False
            self.repo = None
            raise pygit2.GitError("No git repo exists within .wgit. Reinitialize weigit!")
        else:
            self.exists = True
            self.repo = pygit2.Repository(str(self.path))
            self.index = self.repo.index
            self.name = "user"  # Commit metadata
            self.email = "user@email.com"

    def add(self):
        if self.exists:
            self.index.add_all()
            self.index.write()

    def commit(self, message):
        if self.exists:
            # if no commit exists, set ref to HEAD and parents to empty
            try:
                ref = self.repo.head.name
                parents = [self.repo.head.target]
            except pygit2.GitError:
                ref = "HEAD"
                parents = []

            author = pygit2.Signature(self.name, self.email)
            committer = pygit2.Signature(self.name, self.email)
            tree = self.index.write_tree()
            self.repo.create_commit(ref, author, committer, message, tree, parents)

    def status(self):
        print(self.repo.status())
