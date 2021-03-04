#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import warnings

import setuptools
import torch
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))


def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs


# https://packaging.python.org/guides/single-sourcing-package-version/
def find_version(version_file_path):
    with open(version_file_path) as version_file:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


extensions = []
cmdclass = {}

force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
if (torch.cuda.is_available() and CUDA_HOME is not None) or force_cuda:
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
else:
    warnings.warn("Cannot install FusedAdam cuda.")


if __name__ == "__main__":
    setuptools.setup(
        name="fairscale",
        description="FairScale: A PyTorch library for large-scale and high-performance training.",
        version=find_version("fairscale/__init__.py"),
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


# Bump this number if you want to force a CI cache invalidation on the pip venv.
# CI cache version: 1
