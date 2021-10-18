# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
import shutil
import tempfile

from torchvision.datasets import MNIST

TEMPDIR = tempfile.gettempdir()


def setup_cached_mnist():
    done, tentatives = False, 0
    while not done and tentatives < 5:
        # Monkey patch the resource URLs to work around a possible blacklist
        MNIST.mirrors = ["https://github.com/blefaudeux/mnist_dataset/raw/main/"] + MNIST.mirrors

        # This will automatically skip the download if the dataset is already there, and check the checksum
        try:
            _ = MNIST(transform=None, download=True, root=TEMPDIR)
            done = True
        except RuntimeError as e:
            logging.warning(e)
            mnist_root = Path(TEMPDIR + "/MNIST")
            # Corrupted data, erase and restart
            shutil.rmtree(str(mnist_root))

        tentatives += 1

    if done is False:
        logging.error("Could not download MNIST dataset")
        exit(-1)
    else:
        logging.info("Dataset downloaded")
