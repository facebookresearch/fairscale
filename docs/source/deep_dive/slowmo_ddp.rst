SlowMo Distributed Data Parallel
================================

Training neural networks in a distributed data-parallel manner results in non-linear scaling (slowdown) due to the time spent on communication
between the different nodes (as well as, to a lesser extent though, synchronization between the different nodes). So, a distributed training run
with 8 nodes is not 8x faster than a run with 1 node as we would expect it to be.

SlowMo Distributed Data Parallel aims to solve this by replacing the typical exact allreduce between gradients with an approximate
averaging of parameters. This approximate averaging reduces both the time spent on communication as well as the synchronization between different
nodes.  It uses one of the following two algorithms (configurable) as a base algorithm for this purpose:

* Local SGD (papers `#1 <https://arxiv.org/abs/1602.05629>`_ and `#2 <https://arxiv.org/abs/1705.09056>`_). This algorithm does an allreduce of the parameters every few iterations.

* `Stochastic Gradient Push <https://arxiv.org/abs/1811.10792>`_ (SGP). This algorithm involves one-to-one communications between nodes.

These base algorithms (LocalSGD and SGP), when used only by themselves, result in reduced model quality (measured as accuracy in a classification
setting). The `SlowMo <https://arxiv.org/abs/1910.00643>`_ algorithm alleviates this issue by doing a slow momentum step, typically, every 48 iterations.

The training process with SlowMo looks as follows:

1. Compute the forward pass.

2. Compute the backward pass.

3. During the backward pass, using a backward hook, on each node, the gradients are synchronized using allreduce across the different GPUs on
   that node.

4. Perform the ``optimizer.step()`` to update parameters on each node with the gradients of that node.

5. Approximately average the parameters using a base algorithm - one of LocalSGD or SGP (both are described above).

6. Perform the slow momentum update step once every ``slowmo_frequency`` (typically 48) iterations. In this step, the parameters on different
   nodes are (exactly) averaged, followed by a ``slowmo_optimizer.step()``. Note that this ``slowmo_optimizer`` is different from the original optimizer,
   and it is done in a `Zero-1 <./oss_sdp_fsdp.html>`_ like manner to save memory.

Best practices for using ``SlowMoDistributedDataParallel``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. SlowMo will be useful in deep learning workloads which run on more than 2 nodes in clusters with a slow interconnect, eg Ethernet.

2. SlowMo should be useful in your workload if the following condition holds:

   :math:`\textrm{time_taken_for_all_reduce_of_gradients} \times (1 - \frac{1}{\textrm{localsgd_frequency}} ) > \textrm{time_taken_for_backward_pass}`

   Notes:

   * In case you are using SGP as the base algorithm, the value of ``localsgd_frequency`` can be plugged in as 2.

   * The formula above is a simplified version of:
     :math:`\textrm{time_taken_for_all_reduce_of_gradients} > \textrm{time_taken_for_backward_pass} + \frac{\textrm{time_taken_for_all_reduce_of_gradients}}{\textrm{localsgd_frequency}}`
     The left and right hand sides denote the total backward duration (combining the computation of gradients in the backward pass and the
     communication cost) for DDP and SlowMo DDP, respectively. Since DDP overlaps the computation of gradients with their communication, it is
     bottlenecked by the latter.  In contrast, there is an extra ``time_taken_for_backward_pass`` on the right hand side because we do not
     overlap the backward pass with communication in the current implementation of SlowMo.

   * In clusters with slower interconnect, ``time_taken_for_all_reduce_of_gradients`` will go up, leading to SlowMo being more useful. ``localsgd_frequency``
     is also an important factor here. More details on varying that to affect performance are in tip 2 of
     `Performance tips for SlowMoDistributedDataParallel`_.

3. ``slowmo_momentum`` will need to be tuned for obtaining good model quality. A grid search across {0.0, 0.1, 0.2, 0.4, 0.6} should be good enough
   for tuning. This ``slowmo_momentum`` value holds consistent across multiple runs with similar settings.  When the number of nodes used is increased,
   however, a higher value of ``slow_momentum`` should be needed. More details about this can be found in the
   `documentation <../api/experimental/nn/slowmo_ddp.html>`_.

4. Adding SlowMo to existing Distributed Data Parallel code involves two steps, which can be found in the `tutorial <../tutorials/slowmo_ddp.html>`_.

Performance tips for ``SlowMoDistributedDataParallel``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. ``nprocs_per_node`` should be set to the number of GPUs on a node (this number should be the same on each node). This allows the API
   to exploit the fast interconnect between different GPUs on a node.

2. Increasing the ``localsgd_frequency`` results in an increase in speed. However, it comes with a tradeoff of reducing the model quality.
   We recommend keeping the ``localsgd_frequency`` at 3.

3. ``slowmo_memory_efficient`` should typically be used (this is the default behavior). It reduces memory usage by sharding the additional
   slow momentum optimizer's parameters in a `Zero-1`_ like manner.

4. A call to ``model.zero_grad(set_to_none=True)`` should be made after ``optimizer.step()`` in order to save memory for the
   ``model.perform_slowmo()`` step. More details about this can be found in the
   `documentation for perform_slowmo() <../api/experimental/nn/slowmo_ddp.html#:~:text=net.perform_slowmo(optimizer)-,perform_slowmo,-(optimizer%3A%20torch.optim>`_.
