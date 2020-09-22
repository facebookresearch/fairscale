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
        model = myAwesomeModel()
        dataloader = mySuperFastDataloader()
        loss = myVeryRelevantLoss()

        base_optimizer_arguments = {} # any optimizer specific arguments, LR, momentum, etc...
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            **base_optimizer_arguments)

        # Any relevant training loop, nothing specific to OSS. For example:
        model.train()
        for e in range(epochs):
            for batch in dataloader:
                # Train
                model.zero_grad()
                outputs = model(batch["inputs"])
                loss = loss_fn(outputs, batch["label"])
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
        model = myAwesomeModel()
        dataloader = mySuperFastDataloader()
        loss = myVeryRelevantLoss()

        base_optimizer_arguments = {}  # any optimizer specific arguments, LR, momentum, etc...

        # ** NEW ** Wrap a base optimizer into OSS
        base_optimizer = torch.optim.SGD  # any pytorch compliant optimizer
        optimizer = OSS(
            params=model.parameters(),
            optim=base_optimizer,
            **base_optimizer_arguments)

        # Any relevant training loop, nothing specific to OSS. For example:
        model.train()
        for e in range(epochs):
            for batch in dataloader:
                # Train
                model.zero_grad()
                outputs = model(batch["inputs"])
                loss = loss_fn(outputs, batch["label"])
                loss /= world_size
                loss.backward()
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                optimizer.step()
