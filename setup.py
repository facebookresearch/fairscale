#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
import sys
import warnings

import setuptools


def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs


extensions = []
cmdclass = {}

if "--fused-adam" in sys.argv:
    if "CUDA_HOME" not in os.environ:
        warnings.warn("Cannot install FusedAdam cuda because CUDA_HOME environment variable is not set. ")
    else:
        from torch.utils import cpp_extension

        extensions.extend(
            [
                cpp_extension.CUDAExtension(
                    name="fairscale.fused_adam_cuda",
                    sources=[
                        "fairscale/clib/fused_adam_cuda/fused_adam_cuda.cpp",
                        "fairscale/clib/fused_adam_cuda/fused_adam_cuda_kernel.cu",
                    ],
                    extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
                )
            ]
        )

        cmdclass["build_ext"] = cpp_extension.BuildExtension

    sys.argv.remove("--fused-adam")


if __name__ == "__main__":
    setuptools.setup(
        name="fairscale",
        description="fairscale: Utility library for large-scale and high-performance training.",
        install_requires=fetch_requirements(),
        include_package_data=True,
        packages=setuptools.find_packages(exclude=("tests", "tests.*")),
        ext_modules=extensions,
        cmdclass=cmdclass,
        python_requires=">=3.6",
        author="Facebook AI Research",
        author_email="todo@fb.com",
        classifiers=[
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Operating System :: OS Independent",
        ],
    )
