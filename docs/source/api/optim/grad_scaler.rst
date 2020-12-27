Sharded Grad Scaler
========================
Enabling PyTorch's automatic mixed precision usually means using a `GradScaler` to detect underflows.
This grad scaler is not aware of the state sharding when Fairscale OSS is involved, and will lead to deadlocks.
Make sure that you use `ShardedGradScaler` in that case, which is a shard-aware wrapper of PyTorch's implementation.

.. code-block:: python

    import torch
    from fairscale.optim.oss import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP

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

        # ** NEW ** Wrap the model into ShardedDDP
        model = ShardedDDP(model, optimizer)

        # ** NEW ** Use a ShardedGradScaler instead of the default Pytorch GradScaler
        scaler = ShardedGradScaler()

        # Any relevant training loop, nothing specific to OSS. For example:
        model.train()
        for e in range(epochs):
            for (data, target) in dataloader:
                data, target = data.to(rank), target.to(rank)

                # Automatically computes the FW pass in half precision
                with torch.cuda.amp.autocast():
                    model.zero_grad()
                    outputs = model(data)
                    loss = loss_fn(outputs, target)

                # Automatically handle scaled gradients
                scaler.scale(loss).backward()
                optimizer.step()
