Efficient memory usage using Activation Checkpointing
=====================================================

Adapted from `torch.utils.checkpoint`, this is a friendlier wrapper for performing activation checkpointing.

Compared to the PyTorch version, this version wraps a `nn.Module` and allows for all subsequent calls to be
checkpointed.

.. code-block:: python


    from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper


    class CheckpointModel(nn.Module):

        def __init__(self, **kwargs):
            super().__init__()
            torch.manual_seed(0)  # make sure weights are deterministic.
            self.ffn_module = nn.Sequential(
                nn.Linear(32, 128),
                nn.Dropout(p=0.5),
                nn.Linear(128, 32),
            )

            self.ffn_module = checkpoint_wrapper(self.ffn_module, **kwargs)
            self.last_linear = nn.Linear(32, 1)

        def forward(self, input):
            output = self.ffn_module(input)
            return self.last_linear(output)
