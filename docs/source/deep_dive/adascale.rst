Adascale
=========

`Adascale <https://arxiv.org/abs/2007.05105>`_ is a technique used to enable large batch training that allows you to increase batch size
without loss of accuracy. When increasing batch size with the number of devices, the learning rate
is typically tuned based on the batch size. With Adascale, users no longer need to modify the
learning rate schedule and still achieve the desired accuracy. Adascale has been implemented as
the Adascale API in FairScale. This technique typically works well for SGD (with and without momentum)
The assumption is that you already have a good learning rate schedule that works well for small
batch sizes. (AdaScale has not been validated to work effectively with Adam, further research in
that direction is needed.)

AdaScale adapts the learning rate schedule and determines when to stop based on comparing statistics
of large-batch gradients with those of small-batch gradients. Small batch gradients are gradients that
have been computed on each GPU and large batch gradients are the average of gradients computed on N
such GPUs. Adascale uses the concept of gain ratio which is intuitively a measure of how much the
variance has reduced by averaging N small batch gradients. It is a quantity between 1 and N.
In practice, the implementation tracks estimates of the gradient variance and norm-squared which
are smoothed using an exponentially-weighted moving average. If T is the number of steps used to
train the original small batch size before scaling, Adascale stops training once the accumulated
gain ratio is greater than T. As you use more and more GPUs in the training the total steps needed
to train decreases, but due to the value of gain ratio between [1, N], the total steps does not
linearly decrease as you increase the GPUs. Additional training steps are taken to maintain the
model accuracy, when compared with original_total_step/N (i.e. linear scaling). In other words,
whenever the gain ratio is less than N, we could not take as large a step as we may have hoped for,
and so the total number of iterations ends up being larger than T / N.
The current implementation in FairScale supports gradient accumulation training, can be used
with Optimizer State Sharding (OSS), and works with PyTorch LR scheduler classes.

The training process is as follows:

1. Compute the forward pass

2. During the backward pass, hooks attached to each of the parameters fire before the allreduce operation. This is to enable us to calculate the accumulated squares of the local gradients.

3. A final backward hook fires after all the gradients have been reduced using the allreduce op. Using the global gradient square and the accumulated local gradient square, the gradient square average and gradient variance average is calculated.

4. These values are then used to calculate the gain ratio. During the `step` call of the optimizer, the learning rate is updated using this gain ratio value.

5. The training loop terminates once maximum number of steps has been reached

Best practices for `fairscale.optim.AdaScale`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Adascale only works for the SGD optimizer (with and without momentum)
