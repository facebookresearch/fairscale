Efficient Data Parallel Training with SlowMo Distributed Data Parallel
======================================================================

SlowMo Distributed Data Parallel reduces the communication between different
nodes while performing data parallel training. It is mainly useful for use on
clusters with low interconnect speeds between different nodes. When using
SlowMo, the models on the different nodes are no longer kept in sync after each
iteration, which leads to the optimization dynamics being affected. The end
result is close to the results of Distributed Data Parallel, but is not exactly
the same.

If you have code that is setup to use Distributed Data Parallel, using SlowMo Distributed Data Parallel
is simply replacing the DDP call with a call to
``fairscale.experimental.nn.data_parallel.SlowMoDistributedDataParallel``, and adding a
``model.perform_slowmo(optimizer)`` call after ``optimizer.step()`` -- preceded by
``model.zero_grad(set_to_none=True)`` in order to reduce peak memory usage.
The different points at which ``use_slowmo`` is used below help demonstrate these changes:

.. code-block:: python


    import torch
    from fairscale.experimental.nn.data_parallel import SlowMoDistributedDataParallel as SlowMoDDP


    def train(
        rank: int,
        world_size: int,
        epochs: int,
        use_slowmo: bool):

        # process group init
        dist_init(rank, world_size)

        # Problem statement
        model = MyAwesomeModel().to(rank)
        if use_slowmo:
            # Wrap the model into SlowMoDDP
            model = SlowMoDDP(model, slowmo_momentum=0.5, nprocs_per_node=8)
        else:
            model = DDP(model, device_ids=[rank])

        dataloader = MySuperFastDataloader()
        loss_ln = MyVeryRelevantLoss()
        optimizer = MyAmazingOptimizer()

        # Any relevant training loop, with a line at the very end specific to SlowMoDDP, e.g.:
        model.train()
        for e in range(epochs):
            for (data, target) in dataloader:
                data, target = data.to(rank), target.to(rank)
                # Train
                outputs = model(data)
                loss = loss_fn(outputs, target)
                loss.backward()
                optimizer.step()
                model.zero_grad(set_to_none=use_slowmo)  # free memory for the perform_slowmo() call below
                if use_slowmo:
                    model.perform_slowmo(optimizer)

In the example above, when using SlowMoDDP, we are reducing the total communication between
nodes by 3 times as the default ``localsgd_frequency`` is set to 3.
SlowMoDDP takes in ``slowmo_momentum`` as a parameter. This parameter may need to be tuned
depending on your use case. It also takes in ``nproces_per_node`` which should be typically set
to the number of GPUs on a node. Please look at the
`documentation <../api/experimental/nn/slowmo_ddp.html>`_
for more details on these parameters as well as other advanced settings of the SlowMo algorithm.
