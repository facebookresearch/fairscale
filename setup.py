#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os
import re

import setuptools

this_dir = os.path.dirname(os.path.abspath(__file__))


def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs


# https://packaging.python.org/guides/single-sourcing-package-version/
def find_version(version_file_path) -> str:
    with open(version_file_path) as version_file:
        version_match = re.search(r"^__version_tuple__ = (.*)", version_file.read(), re.M)
        if version_match:
            ver_tup = eval(version_match.group(1))
            ver_str = ".".join([str(x) for x in ver_tup])
            return ver_str
        raise RuntimeError("Unable to find version tuple.")


extensions = []
cmdclass = {}

if os.getenv("BUILD_CUDA_EXTENSIONS", "0") == "1":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    extensions.extend(
        [
            CUDAExtension(
                name="fairscale.fused_adam_cuda",
                include_dirs=[os.path.join(this_dir, "fairscale/clib/fused_adam_cuda")],
                sources=[
                    "fairscale/clib/fused_adam_cuda/fused_adam_cuda.cpp",
                    "fairscale/clib/fused_adam_cuda/fused_adam_cuda_kernel.cu",
                ],
                extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
            )
        ]
    )

    cmdclass["build_ext"] = BuildExtension


if __name__ == "__main__":
    setuptools.setup(
        name="fairscale",
        description="FairScale: A PyTorch library for large-scale and high-performance training.",
        version=find_version("fairscale/version.py"),
        setup_requires=["ninja"],  # ninja is required to build extensions
        install_requires=fetch_requirements(),
        include_package_data=True,
        packages=setuptools.find_packages(exclude=("tests", "tests.*")),
        ext_modules=extensions,
        cmdclass=cmdclass,
        python_requires=">=3.6",
        author="Facebook AI Research",
        author_email="todo@fb.com",
        long_description="FairScale is a PyTorch extension library for high performance and large scale training on one or multiple machines/nodes. This library extends basic PyTorch capabilities while adding new experimental ones.",
        long_description_content_type="text/markdown",
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Operating System :: OS Independent",
        ],
    )
