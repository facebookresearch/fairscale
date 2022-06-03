# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# import datetime
import hashlib
import json
import os
import pathlib
import shutil
import sys

import pygit2

from experimental.wgit.utils import ExitCode


class WeiGit:
    def __init__(self) -> None:
        """
        Planned Features:
            1. create the wgit directory. Error, if already dir exists.
            2. SHA1Store.init()
            3. Create SHA1 .wgit/sha1_refs.json
            3. Initialize a .git directory within the .wgit using `git init`.
            4. add a .gitignore within the .wgit directory, so that the git repo within will ignore `sha1_refs.json`
        """

        # Make .wgit directory. If already exists, we error out
        create_dir(
            dir_path=".wgit", exception_msg="An exception occured while wgit initialization: WeiGit already Initialized"
        )

        # create sha1_refs and a .gitignore:
        create_file(file_path=".wgit/file.pt")

        # In general sha1_refs is only create only if .wgit already exists
        create_file(file_path=".wgit/sha1_refs.json")

        # Make the .wgit a git repo
        try:
            pygit2.init_repository(".wgit/.git", False)
        except BaseException as error:
            sys.stderr.write(f"An exception occurred while initializing .wgit/.git: {repr(error)}\n")
            sys.exit(ExitCode.ERROR)

        # add a .gitignore:
        create_file(file_path=".wgit/.gitignore")
        write_to_file(file_path=".wgit/.gitignore", msg="sha1_refs.json\nfiles")

        # Initializing sha1_store only after wgit has been initialized!
        SHA1_store()

    @staticmethod
    def add(file_path):
        repo = Repo(os.getcwd())
        repo_path = repo.get_repo_path()
        if repo.exists():
            # file_name = os.path.basename(file_path)
            sha1_hash = SHA1_store.get_sha1_hash(file_path)

            # use the sha1_has to create a directory with first2 sha naming convention
            try:
                repo_fdir = os.path.join(repo_path, "files", sha1_hash[:6])
                os.makedirs(repo_fdir)
            except FileExistsError:
                # TODO: Better error handling in case we already have a directory? or
                # Taking more letters from the sha1_hash might make more unique dir names.
                sys.stderr.write("An exception occured creating a dir for the checkpoint\n")
                sys.exit(ExitCode.FILE_EXISTS_ERROR)

            # First copy the file to: weigit_repo/files/xx/..xx..sha1..xx../
            repo_fpath = os.path.join(repo_fdir, sha1_hash[6:])
            shutil.copy2(file_path, os.path.join(repo_fdir, sha1_hash[6:]))

            # TODO: Do we want modify or change? Modify relates to content, change relates to even metadata.
            change_time = pathlib.Path(repo_fpath).stat().st_ctime

            # Create the dependency Graph and track reference
            SHA1_store.add_ref(current_sha1_hash=sha1_hash)

            metadata = {
                sha1_hash: {
                    "file_path": repo_fpath,
                    "time_stamp": change_time,
                }
            }

            # Add the metadata to the file.pt json file
            WeiGit._add_metadata_to_json(sha1_hash, metadata)

            # git add the files

    @staticmethod
    def _add_metadata_to_json(sha1_hash, metadata):
        # This for the NEXT SHA1 Commit
        # create a reference:

        # Write to the file.pt json
        file_pt_json = ".wgit/file.pt"
        if not os.path.getsize(file_pt_json):  # If no entry yet
            with open(file_pt_json, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
        else:
            with open(file_pt_json, "r") as f:
                file_pt_data = json.load(f)
                file_pt_data[sha1_hash] = metadata[sha1_hash]

            # dump the file_data in the json file
            with open(file_pt_json, "w", encoding="utf-8") as f:
                json.dump(file_pt_data, f, ensure_ascii=False, indent=4)

    @staticmethod
    def _git_commit():
        pass

    @staticmethod
    def commit(message):
        if Repo(os.getcwd()).exists():
            print(f"commited with message: {message}")

            # # First copy the file to: weigit_repo/files/xx/..xx..sha1..xx../
            # repo_fpath = os.path.join(repo_fdir, sha1_hash[6:])
            # shutil.copy2(file_path, os.path.join(repo_fdir, sha1_hash[6:]))

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
        weigit_repo = Repo(os.getcwd())
        self.repo_path = weigit_repo.get_repo_path()
        print(f"The Repo path: {self.repo_path}")  # DEBUG

        # should create the files directory for storing the checkpoint here,
        # since it is an immutable directory, should be handled by SHA1

    def add(self, file_path):
        """
        Input: File
        Output: SHA1
        """
        pass

    def add_ref(current_sha1_hash):
        # should use the sha1_refser to track the parent and children references. How exactly is open question to me!
        ref_file = ".wgit/sha1_refs.json"
        if not os.path.getsize(ref_file):  # If no entry yet
            with open(ref_file) as f:
                ref_data = {
                    current_sha1_hash: {"parent": "ROOT", "child": "HEAD", "ref_count": 0},
                }
            with open(ref_file, "w", encoding="utf-8") as f:
                json.dump(ref_data, f, ensure_ascii=False, indent=4)
        else:
            print("\n IN THE ELSE BLOCK \n")

            with open(ref_file, "r") as f:
                ref_data = json.load(f)

            # get the last head and replace it's child from HEAD -> this sha1
            for key, vals in ref_data.items():
                if vals["child"] == "HEAD":
                    parent = key

            ref_data[parent]["child"] = current_sha1_hash

            # increase the ref counter of that (now parent sha1)
            ref_data[parent]["ref_count"] += 1

            # Add this new sha1 as a new entry, make the earlier sha1 a parent
            # make "HEAD" as a child, and json dump

            ref_data[current_sha1_hash] = {"parent": parent, "child": "HEAD", "ref_count": 0}

            # Try
            with open(ref_file, "w", encoding="utf-8") as f:
                json.dump(ref_data, f, ensure_ascii=False, indent=4)
            # except: clean up TODO: Make the whole add method in try except block cleaning up the SHA1 directory ## TODO

    def read(SHA1):
        pass

    @staticmethod
    def get_sha1_hash(file_path):
        BUF_SIZE = 104857600  # Reading file in 100MB chunks

        sha1 = hashlib.sha1()
        with open(file_path, "rb") as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                sha1.update(data)
        return sha1.hexdigest()


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
            with .git and sha1_refs in the repo.
            """
            is_wgit_in_curr = pathlib.Path(os.path.join(check_dir, ".wgit")).exists()
            is_refcount_in_wgit = pathlib.Path(os.path.join(check_dir, ".wgit/sha1_refs.json")).exists()
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


def create_dir(dir_path, exception_msg=""):
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        sys.stderr.write(exception_msg + "\n")
        sys.exit(ExitCode.FILE_EXISTS_ERROR)


def create_file(file_path, exception_msg="An exception occured while creating "):
    if not os.path.isfile(file_path):
        with open(file_path, "w") as f:
            pass
    else:
        sys.stderr.write(exception_msg + f"{file_path}: {repr(FileExistsError)}\n")
        sys.exit(ExitCode.FILE_EXISTS_ERROR)


def write_to_file(file_path, msg):
    with open(file_path, "a") as file:
        file.write(msg)
