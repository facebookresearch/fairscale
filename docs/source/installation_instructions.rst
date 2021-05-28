Installing FairScale
====================

Installing FairScale is extremely simple with pre-built binaries(pip) that we provide. You can also build
from source using the instructions below.

### Requirements

* PyTorch>= 1.6.0

### Installing the pip package (stable)

.. code-block:: bash

	pip install fairscale
	

### Installing from source

.. code-block:: bash
    
    git clone https://github.com/facebookresearch/fairscale.git
    cd fairscale
    pip install -r requirements.txt
    # -e signified dev mode since e stands for editable
    pip install -e .


Note: If either of the above fails, add `--no-build-isolation` to the `pip install` command (this could be a problem with recent versions of pip).