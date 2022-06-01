# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import os
import pathlib
import sys

import pygit2

from experimental.wgit.utils import ExitCode


class WeiGit:
    def __init__(self) -> None:
        """
        Planned Features:
            1. create the wgit directory. Error, if already dir exists.
            2. SHA1Store.init()
            3. Create SHA1 .wgit/sha1_ref_count.json
            3. Initialize a .git directory within the .wgit using `git init`.
            4. add a .gitignore within the .wgit directory, so that the git repo within will ignore `sha1_ref_count.json`
        """

        # Make .wgit directory. If already exists, we error out
        try:
            os.mkdir(".wgit")
        except FileExistsError:
            sys.stderr.write("An exception occured while wgit initialization: WeiGit already Initialized\n")
            sys.exit(ExitCode.FILE_EXISTS_ERROR)

        # if no .wgit dir then initialize the following
        SHA1_store()

        # create sha1_ref_count and a .gitignore:
        # In general sha1_ref_count is only create only if .wgit already exists
        try:
            ref_count_json = ".wgit/sha1_ref_count.json"
            with open(ref_count_json, "w") as f:
                pass
        except FileExistsError as error:
            sys.stderr.write(f"An exception occured while creating {ref_count_json}: {repr(error)}\n")
            sys.exit(ExitCode.FILE_EXISTS_ERROR)

        # Make the .wgit a git repo
        try:
            pygit2.init_repository(".wgit/.git", False)
        except BaseException as error:
            sys.stderr.write(f"An exception occurred while initializing .wgit/.git: {repr(error)}\n")
            sys.exit(ExitCode.ERROR)

        # add a .gitignore:
        try:
            gitignore = ".wgit/.gitignore"
            with open(gitignore, "w") as f:
                f.write("sha1_ref_count.json")
        except FileExistsError as error:
            sys.stderr.write(f"An exception occured while creating {gitignore}: {repr(error)}\n")
            sys.exit(ExitCode.FILE_EXISTS_ERROR)

    @staticmethod
    def add(file):
        if Repo(os.getcwd()).exists():
            print("wgit added")

    @staticmethod
    def status():
        if Repo(os.getcwd()).exists():
            print("wgit status")

    @staticmethod
    def log(file):
        if Repo(os.getcwd()).exists():
            if file:
                print(f"wgit log of the file: {file}")
            else:
                print("wgit log")

    @staticmethod
    def commit(message):
        if Repo(os.getcwd()).exists():
            if message:
                print(f"commited with message: {message}")
            else:
                print("wgit commit")

    @staticmethod
    def checkout():
        if Repo(os.getcwd()).exists():
            print("wgit checkout")

    @staticmethod
    def compression():
        print("Not Implemented!")

    @staticmethod
    def checkout_by_steps():
        print("Not Implemented!")


class SHA1_store:
    """
    Planned Features:
        1. def init
        2. def add <file or data> -> SHA1
        3. def remove (SHA1)
        4. def add_ref(children_SHA1, parent_SHA1)
        5. def read(SHA1): ->
        6. def lookup(SHA1): -> file path to the data. NotFound Exception if not found.
    """

    def __init__(self) -> None:
        pass


class Repo:
    """
    Designates the weigit repo, which is identified by a path to the repo.
    """

    def __init__(self, check_dir) -> None:
        self.repo_path = None
        self.check_dir = os.path.realpath(check_dir)

    def exists(self):
        def weigit_repo_exists(check_dir):
            """
            checks if the input path to dir (check_dir) is a valid weigit repo
            with .git and sha1_ref_count in the repo.
            """
            is_wgit_in_curr = pathlib.Path(os.path.join(check_dir, ".wgit")).exists()
            is_refcount_in_wgit = pathlib.Path(os.path.join(check_dir, ".wgit/sha1_ref_count.json")).exists()
            is_git_in_wgit = pathlib.Path(os.path.join(check_dir, ".wgit/.git")).exists()
            return is_wgit_in_curr and is_refcount_in_wgit and is_git_in_wgit

        if weigit_repo_exists(self.check_dir):
            self.repo_path = os.path.join(self.check_dir, ".wgit")
        else:
            while self.check_dir != os.getcwd():
                self.check_dir = os.path.dirname(self.check_dir)

                if weigit_repo_exists(self.check_dir):
                    self.repo_path = os.path.join(self.check_dir, ".wgit")
                    break

        if self.repo_path is None:
            print("Initialize a weigit repo first!!")
            is_exist = False
        else:
            is_exist = True
        return is_exist

    def get_repo_path(self):
        if self.repo_path is None:
            self.exists()
        return self.repo_path
