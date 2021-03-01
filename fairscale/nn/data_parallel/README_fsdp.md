### Overview
Recent work by [Microsoft](https://arxiv.org/abs/1910.02054) and [Google](https://arxiv.org/abs/2004.13336) has shown that data parallel training can be made significantly more efficient by sharding the model parameters and optimizer state across data parallel workers. These ideas are encapsulated in the new **`FullyShardedDataParallel` (FSDP)** wrapper, which is a drop-in replacement for PyTorch's `DistributedDataParallel` (DDP) wrapper.

Compared to PyTorch `DistributedDataParallel` (DDP):
* FSDP shards parameters (FP16 + FP32) and optimizer state across data parallel GPUs
* FSDP with `reshard_after_forward=False` has the same communication cost as PyTorch DDP and is similar to ZeRO-2
* FSDP with `reshard_after_forward=True` increases total communication by 50% and is similar to ZeRO-3:
    * all-gather parameters at start of forward pass and start of backward pass
    * reduce-scatter grads at end of backward pass
* in practice, FSDP is faster than PyTorch DDP because the optimizer step is sharded, and the extra communication can be overlapped with the forward pass
* FSDP enables training 13B parameter models on 8 GPUs and 175B parameter models on 128 GPUs. When using the `cpu_offload=True` option, it's possible to train 1T parameter models on 256 GPUs.

### General usage notes
- for best memory efficiency wrap each layer in your network with FSDP and set `reshard_after_forward=True`
- for best training speed set `reshard_after_forward=False` (wrapping each layer is not required, but will improve speed further)
- if you're using `torch.cuda.amp.autocast` for mixed precision, that's fully compatible with the FSDP wrapper, just set `mixed_precision=True`
- if combining with [activation checkpointing](https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/misc/checkpoint_activations.py), prefer `FSDP(checkpoint_wrapper(module))` over `checkpoint_wrapper(FSDP(module))`. The latter will result in more communication and will be slower.
- this is full compatible with pointwise Optimizers, e.g., Adam, AdamW, Adadelta, Adamax, SGD, etc.. However, the sharding will result in slightly different results when using non-pointwise Optimizers, e.g., Adagrad, Adafactor, LAMB, etc.

### How it works
In standard distributed data parallel (DDP) training every worker processes a separate batch and the gradients are summed across workers using an [all-reduce operation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce). While DDP has become very popular, it wastes GPU memory because the model weights and optimizer states are replicated across all DDP workers.

The key insight to unlock full parameter sharding is that we can decompose the [all-reduce](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce) operation in DDP into separate [all-gather](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather) and [reduce-scatter](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#reducescatter) operations:

<img width="864" alt="Screen Shot 2021-01-12 at 12 35 19 PM" src="https://user-images.githubusercontent.com/231798/108780259-26870a00-7536-11eb-890d-51720f39d098.png">

Then, we can rearrange the reduce-scatter + all-gather so that each DDP worker only needs to store a single shard of parameters and optimizer state. The figure below illustrates standard DDP training (left) and fully sharded training (right):

<img width="1119" alt="Screen Shot 2021-02-24 at 4 39 55 PM" src="https://user-images.githubusercontent.com/231798/109069252-f9199800-76be-11eb-96f8-86767edf1eb9.png">

To maximize memory efficiency we can discard the full weights after each layer's forward pass, saving memory for subsequent layers. This can be implemented by applying the FSDP wrapper to every layer in your network (with `reshard_after_forward=True`). In pseudo-code:
```
FSDP forward pass:
    for layer_i in layers:
        all-gather full weights for layer_i
        forward pass for layer_i
        discard full weights for layer_i
FSDP backward pass:
    for layer_i in layers:
        all-gather full weights for layer_i
        backward pass for layer_i
        discard full weights for layer_i
        reduce-scatter gradients for layer_i
```
#### Mixed Precision

When `mixed_precision=True`:

- Sharded parameters are downcast to `fp16` before `forward`, promoted to `fp32` after forward.
- buffers: batch norm not handled in any special way, buffers are kept in `fp16`. Buffers are not sharded regardless of arguments.

- By default, gradients will be computed and reduced `fp32_reduce_scatter` controls 
- FIXME: If `torch.amp.autocast` is enabled it will over-ride the output dtypes of some operations


#### Using CPU RAM

`move_grads_to_cpu` and `cpu_offload` control which tensors get moved to CPU.

- `cpu_offload` moves weights to CPU when they are not being used. 
- `move_grads_to_cpu` moves gradients to CPU. The use of this option requires that the optimizer has a copy of the model parameters on CPU.

#### Gradient Clipping
By default, 
```python
sharded_module = FullyShardedDataParallel(my_module)
torch.nn.utils.clip_grad_norm_(sharded_module.parameters(), max_norm=1.0)
```
will use an incorrect norm (the norm over all params in a shard) when clipping gradients.
To overcome this, you can either call
`sharded_module.clip_grad_norm(1.0)`
which does the extra computation required to compute the norm properly, or use `torch.nn.utils.clip_grad_value_`.
```



#### Misc

- we don't start the FP32 -> FP16
        # transfer until after the optimization step completes.
