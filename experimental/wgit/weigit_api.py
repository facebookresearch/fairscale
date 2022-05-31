# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import os
import pathlib
import sys

import pygit2


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

        # Make .wgit directory
        try:
            os.mkdir(".wgit")
        except FileExistsError:
            sys.stderr.write("WeiGit has been already Initialized \n")
            sys.exit(0)

        # if no .wgit dir then initialize the following
        SHA1_store()

        # create sha1_ref_count
        with open(".wgit/sha1_ref_count.json", "w") as f:
            print("The sha1_ref_count.json file has been created!")

        # Make the .wgit a git repo, add a gitignore and add SHA1_ref
        pygit2.init_repository(".wgit/.git", False)
        with open("./.wgit/.gitignore", "w") as f:
            f.write("sha1_ref_count.json")

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
                print(f"wgit log of the file {file}")
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
        print("SHA1_store has been initialized!!")


class Repo:
    """
    Designates the weigit repo, which is identified by a path to the repo.
    """

    def __init__(self, check_dir) -> None:
        self.repo_path = None
        self.check_dir = check_dir

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
            max_dir_depth = 100
            while self.check_dir != os.getcwd() and (max_dir_depth != 0):
                self.check_dir = os.path.dirname(self.check_dir)

                if weigit_repo_exists(self.check_dir):
                    self.repo_path = os.path.join(self.check_dir, ".wgit")
                    break
                max_dir_depth -= 1

        if self.repo_path is None:
            print("Initialize a weigit repo first!!")
            is_exist = False
        else:
            is_exist = True
        return is_exist

    def get_repo_path(self):
        if self.repo_path is None:
            self.find_existing_repo()
        return self.repo_path
