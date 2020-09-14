Pipeline Parallel
=================

Let us start with a toy model that contains two linear layers.

.. code-block:: default


    import torch
    import torch.nn as nn

    class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.net1 = torch.nn.Linear(10, 10)
            self.relu = torch.nn.ReLU()
            self.net2 = torch.nn.Linear(10, 5)

        def forward(self, x):
            x = self.relu(self.net1(x))
            return self.net2(x)

    model = ToyModel()

To run this model on 2 GPUs we need to convert the model
to ``torch.nn.Sequential`` and then wrap it with ``fairscale.nn.Pipe``.

.. code-block:: default


    import fairscale
    import torch
    import torch.nn as nn

    model = nn.Sequential(
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 5)
            )

    model = fairscale.nn.Pipe(model, balance=[2, 1])

This will run the first two layers on ``cuda:0`` and the last
layer on ``cuda:1``. To learn more, visit the `Pipe <../api/nn/pipe.html>`_ documentation.
