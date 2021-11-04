SlowMo Distributed Data Parallel
================================

Training neural networks in a distributed data-parallel manner results in non-linear scaling (slowdown) due to the time spent on communication
between the different nodes (as well as, to a lesser extent though, synchronization between the different nodes). So, a distributed training run
with 8 nodes is not 8x faster than a run with 1 node as we would expect it to be.

SlowMo Distributed Data Parallel aims to solve this by replacing the exact allreduce between gradients, which is typically done, with an approximate
averaging of parameters. This approximate averaging reduces both the time spent on communication as well as the synchronization between different
nodes.  It uses one of the following two algorithms (configurable) as a base algorithm for this purpose -

* `Local <https://arxiv.org/abs/1602.05629>`_ `SGD <https://arxiv.org/abs/1705.09056>`_. This algorithm does an allreduce of the parameters every few iterations.

* `Stochastic Gradient Push <https://arxiv.org/abs/1811.10792>`_ (SGP). This algorithm involves one-to-one communications between nodes.

These base algorithms (LocalSGD and SGP), when used only by themselves, result in reduced accuracy. The `SlowMo <https://arxiv.org/abs/1910.00643>`_
algorithm removes this accuracy loss by doing a slow momentum step, typically, every 48 iterations.

The training process with SlowMo looks as follows:

1. Compute the forward pass.

2. Compute the backward pass.

3. During the backward pass, using a backward hook, the gradients are synchronized using allreduce across the different GPUs on a node.

4. Perform the ``optimizer.step()`` to update parameters on a node with the gradients of that node.

5. Approximately average the parameters using a base algorithm - one of LocalSGD or SGP (both are described above).

6. Perform the slow momentum update step once every ``slowmo_frequency`` (typically 48) iterations. In this step, the parameters on different
   nodes are reduced, followed by a ``slowmo_optimizer.step()``. Note that this ``slowmo_optimizer`` is different from the original optimizer,
   and it is done in a `Zero-1 <https://fairscale.readthedocs.io/en/latest/deep_dive/oss_sdp_fsdp.html>`_ like manner to save memory.

Best practices for using ``SlowMoDistributedDataParallel``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. SlowMo will be useful in deep learning workloads which run on greater than 2 nodes in clusters with a slow interconnect, eg Ethernet.

2. SlowMo should be useful in your workload if the following condition holds:

   :math:`\textrm{time_taken_for_all_reduce_of_gradients} \times (1 - \frac{1}{\textrm{localsgd_frequency}} ) > \textrm{time_taken_for_backward_pass}`

   In clusters with slower interconnect, ``time_taken_for_all_reduce_of_gradients`` will go up, leading to SlowMo being more useful. ``localsgd_frequency``
   is also an important factor here. More details on varying that to affect performance are in tip 2 of
   `Performance tips for SlowMoDistributedDataParallel`_.

3. ``slowmo_momentum`` will need to be tuned for obtaining good accuracy. A random search across 4 values from [0.1, 0.2, ..., 0.7] should be good enough
   for tuning. This ``slowmo_momentum`` value holds consistent across multiple runs with similar settings.  When the number of nodes used is increased,
   however, a higher value of ``slow_momentum`` should be needed. More details about this can be found in the
   `documentation <https://fairscale.readthedocs.io/en/latest/api/experimental/nn/slowmo_ddp.html>`_.

4. Adding SlowMo involves two steps, which can be found in the `tutorial <https://fairscale.readthedocs.io/en/latest/tutorials/slowmo_ddp.html>`_.

Performance tips for ``SlowMoDistributedDataParallel``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. ``nprocs_per_node`` should be set to the number of GPUs in a node. This allows the API to exploit the fast interconnect between different GPUs
   on a node.

2. Increasing the ``localsgd_frequency`` results in an increase in speed. However, it comes with a tradeoff of reducing the accuracy.
   We recommend keeping the ``localsgd_frequency`` at 3.

3. ``slowmo_memory_efficient`` should typically be used. It reduces memory usage by sharding the extra slow momentum optimizer's parameters in
   a `Zero-1`_ like manner.
