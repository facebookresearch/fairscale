FullyShardedDataParallel
========================



Signatures
==========
.. autoclass:: fairscale.nn.FullyShardedDataParallel
    :members:
    :undoc-members:


Wrapping
=========================

There are three cases where the `enable_wrap` context can be useful:

* When you'd like to apply the same parameters to all child modules that you wrap with FSDP.
    * Calling the `wrap` function within the `enable_wrap` context will save you from passing the same set of FSDP kwargs explicitly.
    * We recommend it since it will also allow more overlapping!

* When wrapping large models that do NOT fit within the CPU memory.
    I.e. you don't first create the full model and then traverse it to wrap it with FSDP at different parts. Instead, you create a wrapped instance of the model incrementally as you build up the model, allowing large modules to be sharded in-place.

.. code-block:: python

        from fairscale.nn.wrap import auto_wrap, enable_wrap
        from fairscale.
        fsdp_params = dict(mixed_precision=True, flatten_parameters=True)
        with enable_wrap(**fsdp_params):
            # Wraps layer in FSDP by default if within context
            self.l1 = wrap(torch.nn.Linear(5, 5))
            assert isinstance(self.l1)
            # Separately Wraps children modules with more than 1e8 params
            self.l2 = auto_wrap(TransformerBlock(), min_num_params=1e8)
