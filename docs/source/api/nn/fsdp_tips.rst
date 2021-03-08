FSDP Notes
========================================
This document describes how ``FSDP`` works, including subtle behaviors that can change performance significantly.
See :doc:`FullyShardedDataParallel <fsdp>` for python docstrings.

Overview
---------

Recent work by `Microsoft <https://arxiv.org/abs/1910.02054>`__ and
`Google <https://arxiv.org/abs/2004.13336>`__ has shown that data
parallel training can be made significantly more efficient by sharding
the model parameters and optimizer state across data parallel workers.
These ideas are encapsulated in the new  ``FullyShardedDataParallel``_
(FSDP) wrapper, which is a drop-in replacement for the PyTorch
``DistributedDataParallel`` (DDP) wrapper.

Compared to PyTorch ``DistributedDataParallel``:

* FSDP shards parameters (FP16 + FP32) and optimizer state across data parallel GPUs
* FSDP with ``reshard_after_forward=False`` has the same communication cost as PyTorch DDP and is similar to ZeRO-2
* FSDP with ``reshard_after_forward=True`` increases total communication by 50% and is similar to ZeRO-3:
    * all-gather parameters at start of forward pass and start of backward pass
    * reduce-scatter grads at end of the backward pass
* In practice, FSDP is faster than DDP because the optimizer step is sharded, and the extra communication can be overlapped with the forward pass.
* FSDP enables training 13B parameter models on 8 GPUs and 175B parameter models on 128 GPUs. When using the ``cpu_offload=True`` option, it's possible to train 1T parameter models on 256 GPUs.


General usage notes
--------------------

-  For best memory efficiency use ``auto_wrap`` to wrap each layer in your network with ``FSDP`` and set ``reshard_after_forward=True``
-  For best training speed set ``reshard_after_forward=False`` (wrapping each layer is not required, but will improve speed further)
-  If you're using ``torch.cuda.amp.autocast`` for mixed precision, that's fully compatible with the FSDP wrapper, just set ``mixed_precision=True``
-  If combining with `activation checkpointing <https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/misc/checkpoint_activations.py>`__,
   prefer ``FSDP(checkpoint_wrapper(module))`` over ``checkpoint_wrapper(FSDP(module))``. The latter will result in more communication and will be slower.
-  Results should be identical to DDP with pointwise Optimizers, e.g.,
   Adam, AdamW, Adadelta, Adamax, SGD, etc.. However, the sharding will
   result in slightly different results when using non-pointwise
   Optimizers, e.g., Adagrad, Adafactor, LAMB, etc.
- In `fairseq <https://github.com/pytorch/fairseq>`_, FSDP is activated by the command line option ``--ddp-backend=fully_sharded``.

How it works
------------
In standard distributed data parallel (DDP) training every worker processes a separate batch and the gradients are
summed across workers using an `all-reduce operation <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce>`__.
While DDP has become very popular, it wastes GPU memory because the model weights and optimizer states are replicated across all DDP workers.

The key insight to unlock full parameter sharding is that we can decompose the
`all-reduce <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce>`__
operation in DDP into separate
`all-gather <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather>`__
and
`reduce-scatter <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#reducescatter>`__
operations:

.. |Figure 1| image:: https://user-images.githubusercontent.com/23240128/110170085-a67b6280-7dc7-11eb-9128-88d813fc7037.png

|Figure 1|

Then, we can rearrange the reduce-scatter + all-gather so that each DDP worker only needs to store a single shard of parameters and optimizer state. The figure below illustrates standard DDP training (left) and fully sharded training (right):

.. |Figure 2| image:: https://user-images.githubusercontent.com/231798/109069252-f9199800-76be-11eb-96f8-86767edf1eb9.png

|Figure 2|

To maximize memory efficiency we can discard the full weights after each
layer's forward pass, saving memory for subsequent layers. This can be
implemented by applying the FSDP wrapper to every layer in your network
(with ``reshard_after_forward=True``). In pseudo-code:

::

    FSDP forward pass:
        for layer_i in layers:
            all-gather full weights for layer_i
            forward pass for layer_i
            discard full weights for layer_i
    FSDP backward pass:
        for layer_i in layers:
            all-gather full weights for layer_i
            backward pass for layer_i
            discard full weights for layer_i
            reduce-scatter gradients for layer_i

Saving and Loading
------------------

There are two ways to load and save FSDP instances,

- ``state_dict()`` returns a dictionary containing all parameters, which can be loaded with ``load_local_state_dict()``
- ``local_state_dict()`` returns a dictionary containing a shard's parameters, which can be loaded with ``load_local_state_dict()``


Mixed Precision
---------------

When ``mixed_precision=True``:

-  Sharded parameters are downcast to ``fp16`` before ``forward``, promoted to ``fp32`` after forward.
-  buffers are kept in ``fp16``, unless ``buffer_dtype=torch.float32`` is passed. Buffers are not sharded regardless of arguments.
-  By default, gradients will be computed and reduced
-  ``fp32_reduce_scatter=True`` controls the quantization of the gradient communication
-  If ``torch.amp.autocast`` is enabled it will override the output dtypes of some operations, like ``BatchNorm2D``


Auto-wrap
~~~~~~~~~
Auto wrapping sub-modules with ``FSDP`` is a convenient way to improve training speed by overlapping the all-gather step across the forward passes of different submodules.



.. code-block:: python

    import torch
    from fairscale.nn.wrap import auto_wrap, enable_wrap, wrap
    from fairscale.nn.data_parallel import FullyShardedDataParallel
    from fairscale.utils.testing import DummyProcessGroup
    tfmr = torch.nn.Transformer(num_encoder_layers=2, num_decoder_layers=2)

    group = DummyProcessGroup(rank=0, size=1)
    fsdp_params = dict(mixed_precision=True, flatten_parameters=True)
    with enable_wrap(process_group=group, **fsdp_params):

        # Wraps layer in FSDP by default if within context
        l1 = wrap(torch.nn.Linear(5, 5))
        assert isinstance(l1, FullyShardedDataParallel)
        assert l1.mixed_precision and l1.flatten_parameters
        # Separately Wraps children modules with more than 1e8 params
        tfmr_auto_wrapped = auto_wrap(tfmr, min_num_params=1e6)
        assert isinstance(l2, nn.Transformer)
        for l in l2.encoder.layers:
            assert isinstance(l, FullyShardedDataParallel)
            assert l.mixed_precision and l.flatten_parameters
            assert isinstance(l.linear1, FullyShardedDataParallel)
            assert isinstance(l.linear2, FullyShardedDataParallel)
            assert not isinstance(l.self_attn, FullyShardedDataParallel) # self attention is not auto-wrapped


.. warning:: It is not recommended to use :func:`auto_wrap` with
    :class:`FullyShardedDataParallel` on modules that have shared
    parameters, as the parameter sharing may be broken (i.e. end up not
    shared) if the shared parameters are not (auto-)wrapped under the same
    FSDP wrapper instance.


Using CPU RAM
-------------

``move_grads_to_cpu`` and ``cpu_offload`` control which tensors get
moved to CPU.

-  ``cpu_offload`` moves weights to CPU when they are not being used.
-  ``move_grads_to_cpu`` moves gradients to CPU. The use of this option
   requires that the optimizer has a copy of the model parameters on
   CPU.

Gradient Clipping
-----------------

By default,

.. code-block:: python

    sharded_module = FullyShardedDataParallel(my_module)
    torch.nn.utils.clip_grad_norm_(sharded_module.parameters(), max_norm=1.0)

will use an incorrect norm (the norm over all params in a shard) when
clipping gradients. To overcome this, you can either call
``sharded_module.clip_grad_norm(1.0)`` which does the extra computation
required to compute the norm properly, or use
``torch.nn.utils.clip_grad_value_``.


State Management with extra parameter attributes
------------------------------------------------

We manage several attributes on each Parameter instance. The first two
are set by :func:`_shard_parameters_`:

- ``_is_sharded``: ``True`` if the Parameter is sharded or ``False``
    if the Parameter is intentionally not sharded (in which case we
    will all-reduce grads for this param).
- ``_orig_size``: the size of the original Parameter (before sharding)


The remaining attributes are set in ``_init_param_attributes()``:

- ``_fp32_shard``: a single shard of the parameters in full precision
    (typically FP32, but this is dependent on the dtype of the model
    as it's passed in by the user). This can be on CPU or GPU depending on the value of *``cpu_offload``*.
- ``_fp16_shard``: if ``mixed_precision`` is ``True``, this will be
    a single shard of the parameters in FP16, used for all-gather.
- ``_full_param_padded``: the full weight (padded to be evenly divisible by ``world_size``), used for computation in the
    forward and backward pass. This will be resized in place and only materialized (via all-gather) as needed.

Misc
----
-  we don't start the FP32 -> FP16 transfer until after the optimization step completes.
- any direct weight accesses outside of the fwd/bwd, should be in the ``_summon_full_params`` context

