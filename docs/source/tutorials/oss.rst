Optimizer state sharding
========================

Using torch.nn.parallel.DistributedDataParallel leads to some wasted communications in the case of OSS, but it is possible and makes OSS a drop in solution in your existing torch distributed code.
Let's suppose that your trainer looks like

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
                loss.backward()
                optimizer.step()


Then sharding the optimizer state is merely a matter of wrapping your optimizer in fairscale.optim.OSS, as follows.
DDP can be used in place of ShardedDDP in the example below, but the memory savings will be reduced (the gradients are not as efficiently sharded)

.. code-block:: python


    import torch
    from fairscale.optim.oss import OSS
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP


    def train(
        rank: int,
        world_size: int,
        epochs: int):

        # process group init
        dist_init(rank, world_size)

        # Problem statement
        model = myAwesomeModel().to(rank)
        model = ShardedDDP(model, device_ids=[rank])
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
                loss.backward()
                optimizer.step()


The above `train` function will then need to be run via a `multiprocessing.spawn` function.

.. code-block:: python


    mp.spawn(
            train,
            args=(WORLD_SIZE, EPOCHS),
            nprocs=WORLD_SIZE,
            join=True
        )


to see it in action, you can test it with the following script `here <../../../examples/tutorial_oss.py>`_.


Using PyTorch Automatic Mixed Precision is possible, but it requires a shard-aware GradScaler, which is available in
`fairscale.optim.grad_scaler`. Autocast can be used as is, and the loss will be scaled and handled in the same way.
See [the original documentation] (https://pytorch.org/docs/stable/notes/amp_examples.html?highlight=automatic%20mixed%20precision)
for more information.

.. code-block:: python



    from fairscale.optim.grad_scaler import ShardedGradScaler


    # Creates model and optimizer in default precision
    model = Net().cuda()
    optimizer = optim.SGD(model.parameters(), ...)

    # Creates a ShardedGradScaler once at the beginning of training.
    scaler = ShardedGradScaler()

    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()

            # Runs the forward pass with autocasting.
            with autocast():
                output = model(input)
                loss = loss_fn(output, target)

            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()
