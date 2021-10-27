Efficient Data Parallel Training with SlowMo Distributed Data Parallel
======================================================================

SlowMo Distributed Data Parallel reduces the communication between different
nodes while performing data parallel training. Let's suppose that your trainer
looks like -

.. code-block:: python


    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP


    def train(
        rank: int,
        world_size: int,
        epochs: int):

        # process group init
        dist_init(rank, world_size)

        # Problem statement
        model = myAwesomeModel().to(rank)
        model = DDP(model, device_ids=[rank])
        dataloader = mySuperFastDataloader()
        loss_ln = myVeryRelevantLoss()
        optimizer = myAmazingOptimizer()

        # Any relevant training loop, nothing specific to SlowMoDDP
        # For example:
        model.train()
        for e in range(epochs):
            for (data, target) in dataloader:
                data, target = data.to(rank), target.to(rank)
                # Train
                model.zero_grad()
                outputs = model(data)
                loss = loss_fn(outputs, target)
                loss.backward()
                optimizer.step()


Then using SlowMo Distributed Data Parallel is simply replacing the DDP call with a call to
`fairscale.experimental.nn.data_parallel.SlowMoDistributedDataParallel` and adding a
`model.perform_slowmo(optimizer)` call after `optimizer.step()`, as follows.

.. code-block:: python


    import torch
    from fairscale.experimental.nn.data_parallel import SlowMoDistributedDataParallel as SlowMoDDP


    def train(
        rank: int,
        world_size: int,
        epochs: int):

        # process group init
        dist_init(rank, world_size)

        # Problem statement
        model = myAwesomeModel().to(rank)
        model = SlowMoDDP(model, nprocs_per_node=8)  # Wrap the model into SlowMoDDP
        dataloader = mySuperFastDataloader()
        loss_ln = myVeryRelevantLoss()
        optimizer = myAmazingOptimizer()

        # Any relevant training loop, with a line at the very end specific to SlowMoDDP. For example:
        model.train()
        for e in range(epochs):
            for (data, target) in dataloader:
                data, target = data.to(rank), target.to(rank)
                # Train
                model.zero_grad()
                outputs = model(data)
                loss = loss_fn(outputs, target)
                loss.backward()
                optimizer.step()
                model.perform_slowmo(optimizer)  # SlowMoDDP specific

SlowMoDDP takes in slowmo_momentum as a parameter. This parameter may need to
be tuned depending on your use case. Please look at the
`documentation <https://fairscale.readthedocs.io/en/latest/api/experimental/nn/slowmo_ddp.html>`_
for `slowmo_momentum` to know more.