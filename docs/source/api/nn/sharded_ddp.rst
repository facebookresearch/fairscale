ShardedDataParallel
====================

.. autoclass:: fairscale.nn.ShardedDataParallel
    :members:
    :undoc-members:



Performance tips
====================
Using OSS and ShardedDDP changes the communication pattern when compared to DDP, and depending on the training hardware a couple of changes can be beneficial.

* If using multiple nodes, make sure that the reduce buckets are activated. This mitigates some of the communication latency cost
* If using Torch AMP, the forward and backward passes are mostly computed in fp16, but by default the communications will still be fp32.
    * ShardedDDP can compress the gradients back to fp16, using the `reduce_fp16` option.
    * OSS can compress the model shards to fp16 when broadcasting, using the `broadcast_fp16` option. This could have a major effect on performance.
