Optimizer state sharding
========================

Using torch.nn.parallel.DistributedDataParallel leads to some wasted communications, but it is possible and makes OSS a drop in solution in your existing torch distributed code.
Let's suppose that your trainer looks like

.. code-block:: python


    import torch

    def train(
        rank: int,
        world_size: int,
        epochs: int):

        # DDP
        dist_init(rank, world_size)

        # Problem statement
        model = myAwesomeModel().to(rank)
        dataloader = mySuperFastDataloader()
        loss_ln = myVeryRelevantLoss()

        # optimizer specific arguments e.g. LR, momentum, etc...
        base_optimizer_arguments = { "lr": 1e-4} 
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            **base_optimizer_arguments)

        # Any relevant training loop, nothing specific to OSS. For example:
        model.train()
        for e in range(epochs):
            for (data, target) in dataloader:
                data, target = data.to(rank), target.to(rank)
                # Train
                model.zero_grad()
                outputs = model(data)
                loss = loss_fn(outputs, target)
                loss /= world_size
                loss.backward()
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                optimizer.step()


Then sharding the optimizer state is merely a matter of wrapping your optimizer in fairscale.optim.OSS, as follows

.. code-block:: python


    import torch
    from fairscale.optim.oss import OSS

    def train(
        rank: int,
        world_size: int,
        epochs: int):

        # DDP
        dist_init(rank, world_size)

        # Problem statement
        model = myAwesomeModel().to(rank)
        dataloader = mySuperFastDataloader()
        loss_ln = myVeryRelevantLoss()

        # optimizer specific arguments e.g. LR, momentum, etc...
        base_optimizer_arguments = { "lr": 1e-4} 

        # ** NEW ** Wrap a base optimizer into OSS
        base_optimizer = torch.optim.SGD  # any pytorch compliant optimizer
        optimizer = OSS(
            params=model.parameters(),
            optim=base_optimizer,
            **base_optimizer_arguments)

        # Any relevant training loop, nothing specific to OSS. For example:
        model.train()
        for e in range(epochs):
            for (data, target) in dataloader:
                data, target = data.to(rank), target.to(rank)
                # Train
                model.zero_grad()
                outputs = model(data)
                loss = loss_fn(outputs, target)
                loss /= world_size
                loss.backward()
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                optimizer.step()


The above `train` function will then need to be run via a `multiprocessing.spawn` function.

.. code-block:: python


    mp.spawn(
            train,
            args=(WORLD_SIZE, EPOCHS),
            nprocs=WORLD_SIZE,
            join=True
        )
    
to see it in action, you can test it with the following script _`tutorial_oss.py <../../../examples/tutorial_oss.py>`_
