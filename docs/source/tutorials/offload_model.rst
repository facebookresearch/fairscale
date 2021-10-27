Scale your model on a single GPU using OffloadModel
====================================================

`fairscale.experimental.nn.offload.OffloadModel` API democratizes large scale distributed training by enabling
users to train large models on limited GPU resources that would have traditionally resulted in OOM errors.
`OffloadModel` API wraps the given model and shards it almost equally. Each shard of the model is copied
from the CPU to the GPU for the forward pass and then copied back. The same process is repeated in the reverse
order for the backward pass. `OffloadModel` supports mixed precision training, activation checkpointing for reducing
the memory footprint and using micro batches to reduce throughput.

Note: We currently require the model to be a `nn.Sequential` model.

Consider a training loop as described below:

.. code-block:: python


    from torch.utils.data.dataloader import DataLoader
    from torchvision.datasets import FakeData
    from torchvision.transforms import ToTensor

    from fairscale.experimental.nn.offload import OffloadModel


    num_inputs = 8
    num_outputs = 8
    num_hidden =  4
    num_layers =  2
    batch_size =  8

    transform = ToTensor()
    dataloader = DataLoader(
        FakeData(
            image_size=(1, num_inputs, num_inputs),
            num_classes=num_outputs,
            transform=transform,
        ),
        batch_size=batch_size,
    )

    model = torch.nn.Sequential(
        torch.nn.Linear(num_inputs * num_inputs, num_hidden),
        *([torch.nn.Linear(num_hidden, num_hidden) for _ in range(num_layers)]),
        torch.nn.Linear(num_hidden, num_outputs),
    )


To use the `OffloadModel` API, we should wrap the model as shown below. You can specify the device that you want
to use for computing the forward and backward pass, the offload device on which the model will be stored and the number
of slices that the model should be sharded into. By default activation checkpointing is turned off and number of microbatches is 1.

.. code-block:: python


    offload_model = OffloadModel(
        model=model,
        device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        num_slices=3,
        checkpoint_activation=True,
        num_microbatches=1,
    )

    torch.cuda.set_device(0)
    device = torch.device("cuda")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(offload_model.parameters(), lr=0.001)

    # To train 1 epoch.
    offload_model.train()
    for batch_inputs, batch_outputs in dataloader:
        batch_inputs, batch_outputs = batch_inputs.to("cuda"), batch_outputs.to("cuda")
        start = time.time_ns()
        optimizer.zero_grad()
        inputs = batch_inputs.reshape(-1, num_inputs * num_inputs)
        with torch.cuda.amp.autocast():
            output = offload_model(inputs)
            loss = criterion(output, target=batch_outputs)
            loss.backward()
        optimizer.step()
