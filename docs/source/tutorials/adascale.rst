AdaScale SGD
============

`AdaScale <https://arxiv.org/pdf/2007.05105.pdf>`_ adaptively scales the learning rate when using larger batch sizes for data-parallel training. Let's suppose that your trainer looks like

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


Applying AdaScale is as simple as wrapping your SGD optimizer with fairscale.optim.AdaScale, as follows

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

        # Wrap optimizer with AdaScale
        optimizer = AdaScale(optimizer)

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
