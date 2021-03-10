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
        MNIST.resources = [
            (
                "https://github.com/blefaudeux/mnist_dataset/raw/main/train-images-idx3-ubyte.gz",
                "f68b3c2dcbeaaa9fbdd348bbdeb94873",
            ),
            (
                "https://github.com/blefaudeux/mnist_dataset/raw/main/train-labels-idx1-ubyte.gz",
                "d53e105ee54ea40749a09fcbcd1e9432",
            ),
            (
                "https://github.com/blefaudeux/mnist_dataset/raw/main/t10k-images-idx3-ubyte.gz",
                "9fb629c4189551a2d022fa330f9573f3",
            ),
            (
                "https://github.com/blefaudeux/mnist_dataset/raw/main/t10k-labels-idx1-ubyte.gz",
                "ec29112dd5afa0611ce80d1b7f02629c",
            ),
        ]

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
