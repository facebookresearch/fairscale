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

            # if no .wgit dir then initialize the following
            SHA1_store()

            # create sha1_ref_count
            with open(".wgit/sha1_ref_count.json", "w") as f:
                print("The sha1_ref_count.json file has been created!")

            # Make the .wgit a git repo, add a gitignore and add SHA1_ref
            pygit2.init_repository(".wgit/.git", False)
            with open("./.wgit/.gitignore", "w") as f:
                f.write("sha1_ref_count.json")

        except FileExistsError:
            sys.stderr.write("WeiGit has been already Initialized \n")

    @staticmethod
    def add(file):
        if not pathlib.Path(".wgit").exists():
            print("Initialize wgit first!")
        else:
            print("wgit added")

    @staticmethod
    def status():
        if not pathlib.Path(".wgit").exists():
            print("Initialize wgit first!")
        else:
            print("wgit status")

    @staticmethod
    def log():
        if not pathlib.Path(".wgit").exists():
            print("Initialize wgit first!")
        else:
            print("wgit log")

    @staticmethod
    def commit():
        if not pathlib.Path(".wgit").exists():
            print("Initialize wgit first!")
        else:
            print("wgit commit")

    @staticmethod
    def checkout():
        if not pathlib.Path(".wgit").exists():
            print("Initialize wgit first!")
        else:
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
        print("\nSHA1_store has been initialized!!")


class Repo:
    def __init__(self, check_path) -> None:
        self.repo_path = None
        self.find_existing_repo()

    def find_existing_repo(self):
        check_dir = os.path.join(os.getcwd(), os.path.dirname(sys.argv[0]))

        # checks if wgit has been initialized
        is_wgit_in_curr = pathlib.Path(os.path.join(check_dir, ".wgit")).exists()
        is_refcount_in_wgit = pathlib.Path(os.path.join(check_dir, ".wgit/sha1_ref_count.json")).exists()
        is_git_in_wgit = pathlib.Path(os.path.join(check_dir, ".wgit/.git")).exists()

        if is_wgit_in_curr and is_refcount_in_wgit and is_git_in_wgit:
            self.repo_path = os.path.join(check_dir, ".wgit")
            print(f"repo path: {self.repo_path}")
        else:
            run_while_max = 100
            while check_dir != os.getcwd() and (run_while_max != 0):
                check_dir = os.path.dirname(check_dir)

                if pathlib.Path(os.path.join(check_dir, ".wgit")).exists():
                    self.repo_path = os.path.join(check_dir, ".wgit")
                    print(f"repo path: {self.repo_path}")
                    break

                run_while_max -= 1

        if self.repo_path is None:
            print("No weigit repo exists! Initialize one..")

    def get_repo_path(self):
        if self.repo_path is None:
            self.find_existing_repo()
        return self.repo_path
