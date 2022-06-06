# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import pygit2

from experimental.wgit.utils import ExitCode


class WeiGit:
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
        self.repo = Repo(os.getcwd())

        self.wgit_dir = Path(".wgit")
        self.metadata_file = Path(".wgit/checkpoint.pt")
        self.sha1_ref = Path(".wgit/sha1_refs.json")
        self.wgit_git_path = Path(".wgit/.git")
        self.sha1_store = Path(".wgit/sha1_store")

        if not self.repo.exists():
            # Make .wgit directory.
            create_dir(dir_path=self.wgit_dir, exception_msg="An exception occured: WeiGit already Initialized")

            # create sha1_refs and metadata file
            create_file(file_path=self.metadata_file)
            create_file(file_path=self.sha1_ref)

            # Make the .wgit a git repo
            try:
                pygit2.init_repository(str(self.wgit_git_path), False)
                self.pygit = PyGit(os.path.join(os.getcwd(), str(self.wgit_dir)))
                create_file(os.path.join(str(self.wgit_dir), ".gitignore"))
                write_to_file(
                    os.path.join(str(self.wgit_dir), ".gitignore"), msg=f"{self.sha1_ref.name}\n{self.sha1_store.name}"
                )
            except BaseException as error:
                sys.stderr.write(f"An exception occurred while initializing .wgit/.git: {repr(error)}\n")
                sys.exit(ExitCode.ERROR)

            # Initializing sha1_store only after wgit has been initialized!
            SHA1_store()

        else:
            self.pygit = PyGit(os.path.join(os.getcwd(), self.wgit_dir))

    def add(self, file_path):
        repo = Repo(os.getcwd())
        repo_path = repo.path

        if repo.exists():
            # file_name = os.path.basename(file_path)
            sha1_hash = SHA1_store.get_sha1_hash(file_path)

            # use the sha1_has to create a directory with first2 sha naming convention
            try:
                repo_fdir = os.path.join(repo_path, "sha1_store", sha1_hash[:2])
                os.makedirs(repo_fdir)
            except FileExistsError as error:
                # TODO: Better error handling in case we already have a directory?
                sys.stderr.write(f"An exception occured: {repr(error)}\n")
                sys.exit(ExitCode.FILE_EXISTS_ERROR)
            try:
                # First copy the file to: weigit_repo/files/xx/..xx..sha1..xx../
                repo_fpath = os.path.join(repo_fdir, sha1_hash[2:])
                shutil.copy2(file_path, os.path.join(repo_fdir, sha1_hash[2:]))

                # TODO: Do we want modify or change? Modify relates to content, change relates to even metadata.
                change_time = Path(repo_fpath).stat().st_ctime

                # Create the dependency Graph and track reference
                SHA1_store.add_ref(current_sha1_hash=sha1_hash)

                metadata = {
                    "SHA1": {
                        "__sha1_full__": sha1_hash,
                    },
                    "file_path": repo_fpath,
                    "time_stamp": change_time,
                }

                # Add the metadata to the file.pt json file
                self._add_metadata_to_json(sha1_hash, metadata)

                # git add the files
                self.pygit.add()

            except BaseException as error:
                # Cleans up the sub-directories created to store sha1-named checkpoints
                shutil.rmtree(repo_fdir)

    def _add_metadata_to_json(self, sha1_hash, metadata):
        # Write to the checkpoint.pt json
        file_pt_json = self.metadata_file
        # if not os.path.getsize(file_pt_json):  # If no entry yet
        with open(file_pt_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

    def commit(self, message):
        if Repo(os.getcwd()).exists():
            print(f"commited with message: {message}")
            self.pygit.commit(message)

    def status(self):
        if Repo(os.getcwd()).exists():
            print("wgit status")

    def log(self, file):
        if Repo(os.getcwd()).exists():
            if file:
                print(f"wgit log of the file: {file}")
            else:
                print("wgit log")

    def checkout(self):
        if Repo(os.getcwd()).exists():
            print("wgit checkout")

    def compression(self):
        print("Not Implemented!")

    def checkout_by_steps(self):
        print("Not Implemented!")


class PyGit:
    """
    PyGit class to wrap the wgit/.git repo and interacta with the git repo.
    """

    def __init__(self, wgit_path) -> None:
        # Find if a git repo exists within .wgit repo:
        # If exists: then discover it and set the self.gitrepo path to its path
        self.exists = None
        self.wgit_path = Path(wgit_path)
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
            self.name = self.repo.default_signature.name  # Commit metadata
            self.email = self.repo.default_signature.email

    def add(self):
        if self.exists:
            self.index.add_all()
            self.index.write()

    def commit(self, message):
        if self.exists:
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
        self.repo_path = weigit_repo.path

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
        self.check_dir = Path(os.path.realpath(check_dir))

    def exists(self):
        """
        Recursively checks up the directory path from the input `check_dir` upto the cwd for a valid weigit repo.
        Returns True if a valid wgit exists, and sets the self.repo_path to the wgit path.
        """
        if self.weigit_repo_exists(self.check_dir):
            self.repo_path = self.check_dir.joinpath(".wgit")
        else:
            while self.check_dir != Path.cwd():
                self.check_dir = self.check_dir.parent

                if self.weigit_repo_exists(self.check_dir):
                    self.repo_path = self.check_dir.joinpath(".wgit")
                    break

        if self.repo_path is None:
            is_exist = False
        else:
            is_exist = True
        return is_exist

    def weigit_repo_exists(self, check_dir):
        wgit_exists, sha1_refs, git_exists, gitignore_exists = self._weigit_repo_status(check_dir)
        return wgit_exists and sha1_refs and git_exists and gitignore_exists

    def _weigit_repo_status(self, check_dir):
        """
        returns the state of the weigit repo and checks if all the required files are present.
        """
        wgit_exists = check_dir.joinpath(".wgit").exists()
        sha1_refs = check_dir.joinpath(".wgit/sha1_refs.json").exists()
        git_exists = check_dir.joinpath(".wgit/.git").exists()
        gitignore_exists = check_dir.joinpath(".wgit/.gitignore").exists()
        return wgit_exists, sha1_refs, git_exists, gitignore_exists

    def corrupt(self):
        if self.repo_path is None:
            self.exists()
        return not self.weigit_repo_exists(self.repo_path.parent)

    def _raise_wgit_corrupt(self):
        """
        Identify and catch exception if weigit is corrupted.
        """
        if self.corrupt:
            wgit_exists, sha1_refs, git_exists, gitignore_exists = self._weigit_repo_status(self.repo_path.parent)
            try:
                if not wgit_exists:
                    raise FileNotFoundError("weigit repo doesn't exist!")
                if not sha1_refs:
                    raise FileNotFoundError(
                        "Corrupt weigit repo: sha1_refs.json doesn't exist within .wgit! Reinitialize wgit"
                    )
                if not git_exists:
                    raise FileNotFoundError("Corrupt weigit repo: .git doesn't exist within .wgit! Reinitialize wgit")
                if not gitignore_exists:
                    raise FileNotFoundError(
                        "Corrupt weigit repo: .gitignore doesn't exist within .wgit! Reinitialize wgit"
                    )
            except FileNotFoundError as error:
                sys.stderr.write(f"{repr(error)}\n")
                sys.exit(ExitCode.FILE_DOES_NOT_EXIST_ERROR)

    @property
    def path(self):
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
