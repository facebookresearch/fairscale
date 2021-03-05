.. FairScale documentation master file, created by
   sphinx-quickstart on Tue Sep  8 16:19:17 2020.
   You can adapt this file completely to your liking,
   but it should at least contain the root `toctree`
   directive.

Welcome to FairScale's documentation!
=====================================

*FairScale* is a PyTorch extension library for high performance and
large scale training for optimizing training on one or across multiple
machines/nodes. This library extend basic pytorch capabilities while
adding new experimental ones.


Components
----------

* Parallelism:
   * `Pipeline parallelism <../../en/latest/api/nn/pipe.html>`_

* Sharded training:
    * `Optimizer state sharding <../../en/latest/api/optim/oss.html>`_
    * `Sharded grad scaler - automatic mixed precision <../../en/latest/api/optim/grad_scaler.html>`_
    * `Sharded distributed data parallel <../../en/latest/api/nn/sharded_ddp.html>`_
    * `Fully Sharded Data Parallel FSDP <../../en/latest/api/nn/fsdp.html>`_, `FSDP Tips <../../en/latest/api/nn/fsdp_tips.html>`_

* Optimization at scale:
   * `AdaScale SGD <../../en/latest/api/optim/adascale.html>`_

* GPU memory optimization:
   * `Activation checkpointing wrapper <../../en/latest/api/nn/misc/checkpoint_activations.html>`_


* `Tutorials <../../en/latest/tutorials/index.html>`_


.. warning::
    This library is under active development.
    Please be mindful and create an
    `issue <https://github.com/facebookresearch/fairscale/issues>`_
    if you have any trouble and/or suggestions.

.. toctree::
   :maxdepth: 5
   :caption: Contents:
   :hidden:

   tutorials/index
   api/index


Reference
=========

:ref:`genindex` | :ref:`modindex` | :ref:`search`
