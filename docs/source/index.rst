.. fairscale documentation master file, created by
   sphinx-quickstart on Tue Sep  8 16:19:17 2020.
   You can adapt this file completely to your liking,
   but it should at least contain the root `toctree`
   directive.

Welcome to fairscale's documentation!
=====================================

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   :hidden:

   tutorials/index
   api/index

*fairscale* is a PyTorch extension library for high performance and
large scale training for optimizing training on one or across multiple
machines/nodes. This library extend basic pytorch capabilities while
adding new experimental ones.


Components
----------

* Parallelism:
   * `pipeline parallelism <../../en/latest/api/nn/pipe.html>`_
   * `sharded distributed data parallel <../../en/latest/api/nn/sharded_ddp.html>`_

* Optimization:
   * `optimizer state sharding <../../en/latest/api/optim/oss.html>`_
   * `sharded grad scaler - AMP <../../en/latest/api/optim/grad_scaler.html>`_
   * `AdaScale SGD <../../en/latest/api/optim/adascale.html>`_


.. warning::
    This library is under active development.
    Please be mindful and create an
    `issue <https://github.com/facebookresearch/fairscale/issues>`_
    if you have any trouble and/or suggestion.


Reference
=========

:ref:`genindex` | :ref:`modindex` | :ref:`search`
