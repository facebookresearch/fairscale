Scale without modifying learning rate using Adascale
====================================================

`AdaScale <https://arxiv.org/pdf/2007.05105.pdf>`_ adaptively scales the learning rate when
using larger batch sizes for data-parallel training. Let's suppose that your trainer looks
like the following.

.. code-block:: python


    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP


    def train(
        rank: int,
        world_size: int,
        epochs: int):

        # DDP
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
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lr_lambda = lambda x: 1/10**x)

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
            scheduler.step()


Applying AdaScale is as simple as wrapping your SGD optimizer with
`fairscale.optim.AdaScale`, as follows and uses its gain() to update
the effective step and compute learning rate schedule accordingly.

.. code-block:: python


    import torch
    from fairscale.optim.adascale import AdaScale
    from torch.nn.parallel import DistributedDataParallel as DDP


    def train(
        rank: int,
        world_size: int,
        epochs: int):

        # DDP
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
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lr_lambda = lambda x: 1/10**x)

        # Wrap optimizer with AdaScale
        optimizer = AdaScale(optimizer)

        # Any relevant training loop. For example:
        model.train()
        last_epoch = 0
        step = 0
        done = False
        while not done:
            for (data, target) in dataloader:
                data, target = data.to(rank), target.to(rank)
                # Train
                model.zero_grad()
                outputs = model(data)
                loss = loss_fn(outputs, target)
                loss.backward()
                step += optimizer.gain()
                optimizer.step()
                epoch = step // len(dataloader)
                if last_epoch != epoch:
                    scheduler.step()
                    last_epoch = epoch
                if epoch >= epochs:
                    done = True
