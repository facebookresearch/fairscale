Optimizer, Gradient and Model Sharding
=======================================

Using torch.nn.parallel.DistributedDataParallel leads to some wasted communications in the case of OSS,
but it is possible and makes OSS a drop in solution in your existing torch distributed code.
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


Then sharding the optimizer state is merely a matter of wrapping your optimizer in `fairscale.optim.OSS`,
as follows.
DDP can be used in place of ShardedDDP in the example below, but the memory savings will be reduced
(the gradients are not as efficiently sharded).

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
        dataloader = mySuperFastDataloader()
        loss_ln = myVeryRelevantLoss()

        # optimizer specific arguments e.g. LR, momentum, etc...
        base_optimizer_arguments = { "lr": 1e-4}

        # Wrap a base optimizer into OSS
        base_optimizer = torch.optim.SGD  # any pytorch compliant optimizer
        optimizer = OSS(
            params=model.parameters(),
            optim=base_optimizer,
            **base_optimizer_arguments)

        # Wrap the model into ShardedDDP, which will reduce gradients to the proper ranks
        model = ShardedDDP(model, optimizer)

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


The above `train` function can then be run via a `multiprocessing.spawn` call. Note that any launcher
can be used, the only assumption being that each of the ranks lives in its own python process.

.. code-block:: python


    mp.spawn(
            train,
            args=(WORLD_SIZE, EPOCHS),
            nprocs=WORLD_SIZE,
            join=True
        )


Using PyTorch Automatic Mixed Precision is possible, and its actual usage will depend on whether OSS
is used with DDP or with ShardedDDP.
If OSS is used with DDP, then the normal PyTorch GradScaler can be used, nothing needs to be changed.
If OSS is used with ShardedDDP (to
get the gradient sharding), then a very similar flow can be used, but it requires a shard-aware GradScaler,
which is available in `fairscale.optim.grad_scaler`. In both cases Autocast can be used as is, and the
loss will be scaled and handled in the same way.
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


Parameters can be sharded using the FullyShardedDataParallel (FSDP) API. It involves wrapping your model similar to the
SDP API above.

.. code-block:: python


    import torch
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP


    def train(
        rank: int,
        world_size: int,
        epochs: int):

        # process group init
        dist_init(rank, world_size)

        # Problem statement
        model = myAwesomeModel().to(rank)
        dataloader = mySuperFastDataloader()
        loss_ln = myVeryRelevantLoss()

        # optimizer specific arguments e.g. LR, momentum, etc...
        base_optimizer_arguments = { "lr": 1e-4}

        # Wrap a base optimizer into OSS
        base_optimizer = torch.optim.SGD  # any pytorch compliant optimizer

        # Wrap the model into FSDP, which will reduce parameters to the proper ranks
        model = FSDP(model)

        # Any relevant training loop. For example:
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


Auto wrapping sub-modules with FSDP is a convenient way to improve training speed by overlapping
the allgather step across the forward passes of different submodules.
It also improves memory efficiency by freeing gathered parameters after each layer finishes executing.
For example:

.. code-block:: python


    import torch
    from fairscale.nn.wrap import auto_wrap, enable_wrap, wrap
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
    from fairscale.utils.testing import DummyProcessGroup


    tfmr = torch.nn.Transformer(num_encoder_layers=2, num_decoder_layers=2)

    group = DummyProcessGroup(rank=0, size=1)
    fsdp_params = dict(mixed_precision=True, flatten_parameters=True)
    with enable_wrap(wrapper_cls=FSDP, process_group=group, **fsdp_params):

        # Wraps layer in FSDP by default if within context
        l1 = wrap(torch.nn.Linear(5, 5))
        assert isinstance(l1, FSDP)
        assert l1.mixed_precision and l1.flatten_parameters
        # Separately Wraps children modules with more than 1e8 params
        tfmr_auto_wrapped = auto_wrap(tfmr, min_num_params=1e6)
        assert isinstance(l2, nn.Transformer)
        for l in l2.encoder.layers:
            assert isinstance(l, FSDP)
            assert l.mixed_precision and l.flatten_parameters
            assert isinstance(l.linear1, FSDP)
            assert isinstance(l.linear2, FSDP)
            assert not isinstance(l.self_attn, FSDP) # self attention is not auto-wrapped
