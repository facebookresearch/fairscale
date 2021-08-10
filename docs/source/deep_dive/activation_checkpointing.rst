Enhanced Activation Checkpointing
=================================

Activation checkpointing is a technique used to reduce GPU memory usage during training. This is
done by avoiding the need to store intermediate activation tensors during the forward pass. Instead,
the forward pass is recomputed by keeping track of the original input during the backward pass.
There is a slight increase in computation cost (about 33%) but this reduces the need to store
large activation tensors which allows us to increase the batch size and thereby the net throughput
of the model.


Activation checkpointing is implemented by overriding `torch.autograd.Function`. In the `forward`
function which handles the forward pass of the module, using `no_grad`, we can prevent the creation
of the forward graph and materialization of intermediate activation tensors for a long period of
time (i.e till the backward pass). Instead, during the backward pass, the forward pass is executed
again followed by the backward pass. The inputs to the forward pass are saved using a context object
that is then accessed in the backward pass to retrieve the original inputs. We also save the
Random Number Generator(RNG) state for the forward and backward passes as required for Dropout layers.

The above functionality is already implemented as part of the `torch.utils.checkpoint.checkpoint_wrapper`
API whereby different modules in the forward pass can be wrapped. The wrapper in FairScale offers
functionality beyond that provided by the PyTorch API specifically you can use
`fairscale.nn.checkpoint.checkpoint_wrapper` to wrap a `nn.Module`, handle kwargs in the forward
pass, offload intermediate activations to the CPU and handle non-tensor outputs returned from the
forward function.

Best practices for `fairscale.nn.checkpoint.checkpoint_wrapper`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Memory savings depends entirely on the model and the segmentation of checkpoint wrapping.
Each backprop consists of several mini-forward and backprop passes. The gain is entirely dependent
on the memory footprint of the layer’s activations.

2. When using BatchNormalization you may need to freeze the calculation of statistics since we run
the forward pass twice.

3. Ensure that the input tensor’s `requires_grad` field is set to True. In order to trigger the
backward function, the output needs to have this field set. By setting it on the input tensor we
ensure that this is propagated to the output and the `backward` function is triggered.
