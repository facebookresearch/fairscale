Installing FairScale
====================

Installing FairScale is extremely simple with pre-built binaries (pip) that we provide. You can also build
from source using the instructions below.


Requirements
~~~~~~~~~~~~

* PyTorch>= 1.8.1


Installing the pip package (stable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

	pip install fairscale


Installing with conda
~~~~~~~~~~~~~~~~~~~~~

Fairscale is packaged by conda-forge (see `here <https://github.com/conda-forge/fairscale-feedstock>`_)
for both linux & osx, with GPU-enabled builds available on linux.

.. code-block:: bash

	conda install -c conda-forge fairscale


Installing from source
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/facebookresearch/fairscale.git
    cd fairscale
    pip install -r requirements.txt
    # -e signified dev mode since e stands for editable
    pip install -e .

To build with GPU-support enabled, be sure to set ``BUILD_CUDA_EXTENSIONS=1``
as well as an appropriate ``TORCH_CUDA_ARCH_LIST``.

Note: If either of the above fails, add ``--no-build-isolation`` to the ``pip install``
command (this could be a problem with recent versions of pip).
