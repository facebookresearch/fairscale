import argparse
import re
from typing import Tuple

from setup import find_version


def get_next_version(release_type) -> Tuple[Tuple[int, int, int], str, str]:
    current_ver = find_version("fairscale/version.py")
    version_list = [int(x) for x in current_ver.strip("'").split(".")]
    major, minor, patch = version_list[0], version_list[1], version_list[2]
    if release_type == "patch":
        patch += 1
    elif release_type == "minor":
        minor += 1
        patch = 0
    elif release_type == "major":
        major += 1
        minor = patch = 0
    else:
        raise ValueError("Incorrect release type specified. Acceptable types are major, minor and patch.")

    new_version_tuple = (major, minor, patch)
    new_version_str = ".".join([str(x) for x in new_version_tuple])
    new_tag_str = "v" + new_version_str
    return new_version_tuple, new_version_str, new_tag_str


def update_version(new_version_tuple) -> None:
    """
    given the current version, update the version to the
    next version depending on the type of release.
    """

    with open("fairscale/version.py", "r") as reader:
        current_version_data = reader.read()

    # for line in current_version_data:
    version_match = re.search(r"^__version_tuple__ ", current_version_data)

    if version_match:
        new_version_data = "__version_tuple__ = %s\n" % str(new_version_tuple)
        current_version_data = current_version_data.replace(version_match.string, new_version_data)

        with open("fairscale/version.py", "w") as writer:
            writer.write(current_version_data)
    else:
        raise RuntimeError("__version_tuple__ not found in version.py")


def main(args):
    if args.release_type in ["major", "minor", "patch"]:
        new_version_tuple, new_version, new_tag = get_next_version(args.release_type)
    else:
        raise ValueError("Incorrect release type specified")

    if args.update_version:
        update_version(new_version_tuple)

    print(new_version, new_tag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Versioning utils")
    parser.add_argument("--release-type", type=str, required=True, help="type of release = major/minor/patch")
    parser.add_argument(
        "--update-version", action="store_true", required=False, help="updates the version in fairscale/version.py"
    )

    args = parser.parse_args()
    main(args)
