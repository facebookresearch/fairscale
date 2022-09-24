# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.4.11] - TBD

- cleaned up some old issues and fixed a few bug in FSDP
- removing SSD offload to simplify the FSDP code

## [0.4.8]/[0.4.9]/[0.4.10]

### Added

- FSDP: Add pickle/unpickle support for SsdTensorHandle (and derived classes),
  verified that FSDP models w/ ssd_offload enabled can correctly call model.state_dict()
  and model.load_state_dict(...) and thus successfully checkpoint and recover parameters
  stored as SsdFlatParameters. [#964]
- FSDP Ssd Offload: [#974]
   * add __setattr__ function for SsdTensorHandle/SsdFlatParameter to capture when .data is overridden
     and perform necessary checks/updates before letting the tensor metadata be updated.
   * There was a bug in pytorch core that caused re-assigning TensorSubclass .data to
     disable the __torch_dispatch__ mechanism and effectively revert back to a normal
     tensor. Since ssd_offload feature uses __torch_dispatch__ extensively and FSDP overrides
     .data, ssd_offload now can only be used if pytorch version is 1.12.0 or later
     (currently pytorch-nightly release). pytest tests are disabled and trying to import
     ssd_offload or use FSDP with ssd_offload enabled will raise ImportError Exceptions.
   * Enhance storage_state ON_CPU into ON_CPU_CLEAN and ON_CPU_DIRTY. ON_CPU_CLEAN
     indicates that the .tensor value and the value stored on disk are identical, and no
     writes are needed when calling .to_file(). ON_CPU_DIRTY indicates .tensor does not
     match value stored on disk, and a write is necessary. This is disabled in FSDP as
     it does not currently flush variables to disk as soon as optimizer modifies values.
   * Fix detection in SsdTensorHandle __torch_dispatch__ when it is used in an in-place
     operator, or as an output, then calling mark_dirty().
   * Fix grad_fn on SsdFlatParameterViews so that both ssd_flat_param.view.grad_fn points
     to a split and view on ssd_flat_parameter so that when X.backwards() is called. ssd_flat_param.grad
     is properly updated in the backward pass.
   * Add unit tests to verify grad_fn and backward hooks are called appropriately on
     SsdFlatParameters/SsdFlatParameterViews
   * Implement SsdFlatParameter.from_tensor() direct_to_file option. This allows creating
     and SsdFlatParameter and directly writing it to disk, rather than creating a new tensor
     first. This prevents another copy of tensors to be instantiated and can save memory when
     creating very large SsdFlatParameters.
   * Add SsdFlatParameterViewProperty for overriding a parameter in an nn.Module
     similar to pytorch core's parameterization code path. This allows SsdFlatParameterViews
     to be updated as SsdFlatParameter.data is overridden by replacing Layer.weight with a
     property whose getter returns ssd_flat_param.views[view_id].
   * Added example SsdFlatParameterViewParameterization on how the existing parameterization
     method could be used instead of SsdFlatParameterViewProperty. But this method will result
     in memory inefficiencies due to parameterization always keeping a copy of the original
     parameter internally.

### Fixed
- fixed some bugs in FSDP related to supporting data2vec EMA modules.


[0.4.6] - 2022-03-08

### Added
- CosFace's LMCL is added to MEVO. This is a loss function that is suitable
  for large number of prediction target classes. It added normalization,
  class separation margins and feature vector scaling to the standard
  output projection + cross-entropy loss. MEVO supported this with its
  memory saving techniques so that peak GPU memory is much reduced. [#916]
- FSDP: Added skip_params_check_for_root flag to default_auto_wrap_policy which,
  if set, wraps the root module regardless of how many unwrapped params there were
  left after children were wrapped. [#930]
- FSDP: Add support for saving optimizer state when using expert replicas with FSDP.
- OSS: Add a new arg "forced_broadcast_object" to OSS __init__ to apply "_broadcast_object"
  for rebuilding the sharded optimizer. [#937]
- FSDP: Add an arg disable_reshard_on_root for FSDP __init__ [#878]

### Fixed
- FSDP: fixed handling of internal states with state_dict and load_state_dict
  function so that they don't change lazy init state if training hasn't started. [#922]
- FSDP: added support of optimizer state handling when some of the parameters are
  not used. An example is that in a model with a EMA copy that doesn't get trained
  but still wants to be sharded. [#922]

## [0.4.5] - 2022-01-14

### Added
- Layer-wise Gradient Scaling [new feature][experimental] Layer-wise gradient
scaling helps overcomes gradient overflow issues. When used in conjunction with
mixed precision, it enables training larger models and makes the training
process more stable, especially in deep networks [#879]
- FSDP: Added process_group_reduce_scatter parameter to allow users to pass in the process group that is used for reduce scatter operation. [#897]
- FSDP: Added state_dict_on_rank_0_only flag allow user choose to return full state dict on rank 0 and return empty dict non-rank 0 to prevent OOM [#844]

### Changed
- FSDP: Enabled reduce_scatter operation to overlap with all_gather stream and computation stream in backward propagation.
This change increases FSDP's throughput. To roll back this change, please pass ProcessGroupName.default to the process_group_reduce_scatter API parameter in FSDP [#897]

### Fixed
- FSDP: Fixed the issue that the padding size of the input tensor of the reduce scatter is not equal to the reduce scatter process group size # [#907]

## [0.4.4] - 2021-12-21

### Fixed
- Inf/nan check on all gradient tensors in ShardedGradScaler [#890]
- Allow user to supress warning on trainable_parameter in sharded_ddp.py [#886]
- KeyError in certain FSDP wrapping scenarios [#881]

### Added
- Support eval mode in MEVO [#884]
- A new benchmark for the MOE model [#866]

### Changed
- Fixed a corner case of FSDP init order and losing one of the flags [#880]
- FSDP: Adding basic training support for SSD Offload, it now only supports flattened parameters. Renamed OffloadConfig.ssd_filepath_dir to more generic OffloadConfig.dir. SSD Offload remains an experimental feature. [#887]

## [0.4.3] - 2021-11-18

### Added
- Sharded Grad Scaler works with cpu offload in mixed and full precision. [#831]
- API for specifying SSD offload for params with FSDP. You can use a OffloadConfig to specify the type of offload
  and the file path for storing params on SSD. Note: This is an experimental feature. [#855]

### Changed
- MEVO: fixed eval and checkpointing code paths [#851]
- Cleanup: Moving forward we would be testing all of our code with Python 3.9.7, CUDA 11.2 and the following three versions of PyTorch [#847]:
  - the most recent stable version
  - the most recent LTS version
  - a recent nightly build

## [0.4.2] - 2021-11-08
### Fixed
- FSDP: Fixed an pre-backward hook bug for certain type of models and FSDP config. [#833]

### Added
- FSDP: Add support for SSD offload for eval workloads. This is a new experimental feature and should be
        used with caution.
- LayerwiseMemoryTracker[feature][experimental]: This is a new experimental tool to help track, visualize and suggest fix for memory issues occurring during the forward/backward pass of your models. [#808]
- FSDP: limited support of shared weights between FSDP wrappers. This allows large parameter
          and gradient memory to be sharded despite being needed from different layers due to
          weight sharing. [#836]
- OffloadModel: Fix node names to enable correct sharding in auto_shard.py [#830]
- OSS: Relaxed speed and memory constraints on OSS golden data due to regression when we bumped up the
       PyTorch version to 1.9. [#828] [#825]
- Chore: Update PyTorch version that we run benchmarks with. [#823]
- Chore: Update PyTorch version that we run test with. [#809]
- OffloadModel: Extend auto_shard.py to allow dealing with conditionals automatically when tracing with
                torch.fx. This will work for most cases except when the conditional is part of the root instance. [#817]
- [MEVO]: a custom layer to help big vocab trainings. Experimental. Docs is still TBD. [#840]
- SlowMoDistributedDataParallel[feature][experimental] - This is a distributed training wrapper which should be useful on clusters with slow network interconnects (eg Ethernet). This improves on performance as compared to Distributed Data Parallel in such clusters. [#378]

## [0.4.1] - 2021-09-17
### Fixed
- FSDP: We don't attach post backward hooks for params that don't require grad. However in the hook
        triggered after the post backward hook, we assert on the POST_BACKWARD state which can only
        be set in the post backward hook. Modified the assert to account for the fact that the root
        FSDP module can have child modules with params that require grad and it can contain params
        that don't require grad and hence can fail the previous assert. [#761]
- FSDP: Fixed a bug when multiple backward pass is called within an iteration, parameters' sharding
        state might be incorrect. [#775]
- activation checkpoint: Ensure outputs of checkpointed modules only require grad if either
                         the input requires grad or if the parameters require grad. [#787]

- OSS: fix the broadcast_fp16 option, broken after a refactor, this flag was doing nothing (bugfix).[#795]
- OSS: update default device when refreshing the params, meaning that moving the model to GPU after
       the OSS wrap will not trigger warnings and slow the jobs (ease of use). [#786]

### Added
- FSDP: Added support for returning the original names of parameters when `named_parameters` is called on
        the module. To retrieve the orginal names of the parameters along with the params, you need to
        call `named_parameters` under the `summon_full_params` context when using flattened params or original
        params. If you are using original params (i.e flatten_params=False), calling `named_parameters` outside
        of the `summon_full_params` context will still return the original param names along with the local shards. [#755]
- FSDP: Ensure gradient reduction accumulates into the unsharded gradient tensor
        within a backwards pass. This matters when an FSDP module is called
        multiple times within a forward pass, and reduction is not deferred
        using activation checkpoint forward counters, bucketing or some other
        mechanism. [#784]
- activation checkpoint: Added a context manager to disable checkpoint in case the same wrapped module
                         needs to be checkpointed and not checkpointed in different parts of
                         the module forward pass. [#772]
- FSDP: Added a toggle with an environment variable ENABLE_NCCL_BASE_COLLECTIVES=[0,1] to allow users
        enable/disable using new nccl base collecectives. By default, using new nccl base collectives
        is enabled. [#801]

## [0.4.0] - 2021-07-31
### Fixed
- FSDP: fixed final backward callback in certain activation checkpointed cases. Before this fix,
        if a model is activation checkpointed in a certain way, the final backward
        callback can fire incorrectly. That's due to autograd and reentrant backward
        graphs. With this fix, the final callback is always registered on the outer
        most root FSDP instance (i.e. the outer most backward graph), which result
        in reliably firing. This makes FSDP much more robust with respect to different
        models and activation checkpoints. [#753]

### Added
- FSDP: support gradient accumulation without the `no_sync` context. This is useful
        in training with smaller number of GPU with same overall batch size as large
        number of GPUs. Compared with the `no_sync` context, this mode consumes less
        GPU memory but uses more networking bandwidth. [#752]


## [0.3.9] - 2021-07-26
### Fixed
- FSDP: fixed metadata saving and shard consolidation for MoE cases. When a model has
        shared parameters or mixture of expert layers, the handling of state dict
        metadata was broken. This release fixes that. [#746]
- OSS: fixed the buckets which would stay in fp16 if `broadcast fp16` was required [#751]

### Added
- FSDP: better performance; use `_allgather_base` and `_reduce_scatter_base` when they are
        available from pytorch nightly version (will be in 1.10 releases) [#729]
- FSDP: prepared FSDP internals for supporting multiple groups of flatten parameters (to support more general optimization) [#746]

## [0.3.8] - 2021-07-12
### Fixed
- checkpointing: Use dummy tensor to ensure backward pass is called. [#701]
- checkpointing: Ensure internal fwd counter is not incremented in eval mode. [#709]
- checkpointing: Use non-blocking CPU transfer to improve perf. [#719]
- FSDP: Fixed bug where buffers returned in `state_dict()` could still be half precision when `mixed_precision` is set to `True`. [#705]
- FSDP: Ensure requires_grad of FlatParameter is consistent with requires_grad of the original parameters. [#721]
- doc: Thoroughly improved the doc for FSDP. [#711]
- cleanup: Remove examples/ doc from the repo. [#712]
- cleanup: Future proof storage size test. [#735]
- cleanup: Migrate away from legacy torchtext iterators. [#713]
- chore: Updated torch 1.9 to release version. [#717]

### Added
- FSDP: supporting multiple flatten parameter groups [#708] [#711]
- chore: Add the latest numpy version to requirements-test.txt to prevent mypy errors on certain PR commits [#732]

## [0.3.7] - 2021-05-17
### Fixed
- setup.py: hide CUDA extensions behind `BUILD_CUDA_EXTENSIONS` envvar [#634]
- checkpointing: rename and move the `checkpoint_activations` wrapper [#654]
- FSDP: fix `local_state_dict` potentially called child class's `state_dict` [#574]
- FSDP: fix extra process groups being created by default. Old behavior can cause excessive GPU memory usage [#678] [#681]
- FSDP: fix forward pass not overlapping compute and allgather [#671]
- FSDP: improved frozen weight support [#657]
- FSDP: workaround AMP autocast cache issue with `clear_autocast_cache` flag [#650]
- FSDP: Rename API arg `cpu_offload` to `move_params_to_cpu` to better reflect functionality. We will deprecate `cpu_offload` in an upcoming release [#676]
- MoE: several fixes [#666] [#667] [#668]
- SDP: re-expose the module property [#647]
- wrap: support wrapping based on `wrapper_config` [#685]

### Added
- FSDP: added `force_input_to_fp32` flag for SyncBatchNorm [#659]
- FSDP: better memory usage for reduce bucket [#633]
- FSDP: added `local_metadata_dict` to save sharding relating information [#683]
- FSDP: added `consolidate_shard_weights` to reconstruct the consolidated (non-sharded) model weights from saved sharded weights and metadata on the disk [#683]
- Experimental SyncBatchNorm [#662] [#680]

## [0.3.6] - 2021-04-26
### Added
- FSDP: Consolidate cpu\_adam optimizer state dict ([#607](https://github.com/facebookresearch/fairscale/pull/607))

### Fixed
- FSDP: handle model with multiple forward pass and checkpoint ([#621](https://github.com/facebookresearch/fairscale/pull/621))
- FSDP & SDP: check before calling `_specify_ddp_gpu_num` ([#626](https://github.com/facebookresearch/fairscale/pull/626))
- FSDP: relax checking root condition ([#620](https://github.com/facebookresearch/fairscale/pull/620))
- SDP: removing an assert which does not seem always accurate ([#625](https://github.com/facebookresearch/fairscale/pull/625))
- FSDP: changing FSDP init to by pass pg validation ([#619](https://github.com/facebookresearch/fairscale/pull/619))
- OSS: to 100% coverage ([#618](https://github.com/facebookresearch/fairscale/pull/618))

## [0.3.5] - 2021-04-19
### Added
- [offload] Add API, tutorial and smaller doc string changes. ([#576](https://github.com/facebookresearch/fairscale/pull/576))

### Fixed
- FSDP: fixing training with freezing weights ([#614](https://github.com/facebookresearch/fairscale/pull/614))
- SDP: privatizing all the things ([#611](https://github.com/facebookresearch/fairscale/pull/611))
- FSDP: Make `_get_default_cuda_device` more robust to modules without params ([#606](https://github.com/facebookresearch/fairscale/pull/606))
- OffloadModel: Add prev codepath of using OffloadModel without activation checkpointing ([#608](https://github.com/facebookresearch/fairscale/pull/608))

## [0.3.4] - 2021-04-13
### Added
- FSDP: Add no broadcast optim state option ([#560](https://github.com/facebookresearch/fairscale/pull/560))

### Fixed
- ShardedDDP: Properly handle .eval() mode ([#587](https://github.com/facebookresearch/fairscale/pull/587))
- ShardedDDP: Handle model being moved back to CPU prior to state consolidation ([#573](https://github.com/facebookresearch/fairscale/pull/573))
- FSDP: much faster state consolidation ([#595](https://github.com/facebookresearch/fairscale/pull/595))
- FSDP: Add gradient pre-dedivide to prevent overflow with large world sizes ([#565](https://github.com/facebookresearch/fairscale/pull/565))
- Offload: (experimental) Fix activation offloading to CPU ([#588]((https://github.com/facebookresearch/fairscale/pull/588) )

## [0.3.3] - 2021-04-1
### Added
- FSDP: changed `auto_wrap_bn` utility function so that single FSDP group is optional ([#556](https://github.com/facebookresearch/fairscale/pull/556))
- FSDP: optimizer state load/save ([#537](https://github.com/facebookresearch/fairscale/pull/537))
- FSDP: fix weight init when using apply() ([#543](https://github.com/facebookresearch/fairscale/pull/543))
- Multiprocess Pipe: retired old implementation
- Experimental: xpipe

### Fixed
- ShardedDDP deferred init ([#558](https://github.com/facebookresearch/fairscale/pull/558))

## [0.3.2] - 2021-03-18
### Added
- Experimental: Add spectrain support ([#372](https://github.com/facebookresearch/fairscale/issues/372))
- FSDP: enabled pytorch SyncBN (no asserting) ([#527](https://github.com/facebookresearch/fairscale/issues/527))
- FSDP: added `auto_wrap_bn` utility function ([#531](https://github.com/facebookresearch/fairscale/pull/531))

### Fixed
- OSS: fix a compatibily problem with lightning wrt optimizer state dict ([#510](https://github.com/facebookresearch/fairscale/issues/510))
- FSDP: fixed a bug when part of autograd graph is traversed multiple times in mixed precision mode ([#513](https://github.com/facebookresearch/fairscale/pull/513))

## [0.3.1] - 2021-03-09
### Added
- FSDP docs ([#455](https://github.com/facebookresearch/fairscale/issues/455))
- `enable_wrap` and `auto_wrap` APIs ([#446](https://github.com/facebookresearch/fairscale/issues/446))
- Added experimental.nn.OffloadModel API for training large models on a single GPU.([#432](https://github.com/facebookresearch/fairscale/issues/432))

### Fixed
- OSS: fix a broken state dict when using non contiguous param groups
- Several SDP fixes around performance and corner cases
- Many FSDP fixes
- AdaScale & SDP/FSDP test added but not officially supported

## [0.3.0] - 2021-02-22
### Added
- FullyShardedDataParallel (FSDP) ([#413](https://github.com/facebookresearch/fairscale/issues/413))
- ShardedDDP fp16 grad reduction option ([#402](https://github.com/facebookresearch/fairscale/issues/402))
- Expose experimental algorithms within the pip package ([#410](https://github.com/facebookresearch/fairscale/pull/410))

### Fixed
- Catch corner case when the model is too small with respect to the world size, and shards are empty ([#406](https://github.com/facebookresearch/fairscale/pull/406))
- Memory leak in `checkpoint_wrapper` ([#412](https://github.com/facebookresearch/fairscale/pull/412))

## [0.1.7] - 2021-02-19
### Fixed
- ShardedDDP and OSS handle model trainability changes during training ([#369](https://github.com/facebookresearch/fairscale/issues/369))
- ShardedDDP state dict load/save bug ([#386](https://github.com/facebookresearch/fairscale/issues/386))
- ShardedDDP handle train/eval modes ([#393](https://github.com/facebookresearch/fairscale/issues/393))
- AdaScale handling custom scaling factors ([#401](https://github.com/facebookresearch/fairscale/issues/401))

### Added
- ShardedDDP manual reduce option for checkpointing ([#389](https://github.com/facebookresearch/fairscale/issues/389))

## [0.1.6] - 2021-02-10
### Added
- Checkpointing model wrapper (#376)
- Faster OSS, flatbuffers (#371)
- Small speedup in OSS clipgradnorm (#363)

### Fixed
- Bug in ShardedDDP with 0.1.5 depending the init (KeyError / OSS)
- Much refactoring in Pipe (#357, #358, #360, #362, #370, #373)
- Better pip integration / resident pytorch (#375)

## [0.1.5] - 2021-02-03
### Added
- Pytorch compatibility for OSS checkpoints (#310)
- Elastic checkpoints for OSS, world size can vary in between save and loads (#310)
- Tensor views for OSS bucketing, reduced CPU use (#300)
- Bucket calls in ShardedDDP, for faster inter node communications (#327)
- FlattenParamWrapper, which flattens module parameters into a single tensor seamlessly (#317)
- AMPnet experimental support (#304)

### Fixed
- ShardedDDP properly handles device changes via `.to()` (#353)
- Add a new interface for AdaScale, AdaScaleWrapper, which makes it compatible with OSS (#347)


## [0.1.4] - 2021-01-07
### Fixed
- Missing cu files in the pip package


## [0.1.3] - 2021-01-04
### Fixed
- Release numbering within python and from pypi

## [0.1.2] - 2021-01-04
### Added
- AdaScale:
  . Added gradient accumulation feature (#202)
  . Added support of `torch.lr_scheduler` (#229)
  . Added support for `add_param_groups` (#266)
  . Added support for `scale != world_size` (#266)

### Fixed
- AdaScale: smoothing factor value fixed when using gradient accumulation (#235)
- Pipe: documentation on balancing functions (#243)
- ShardedDDP: handle typical NLP models
- ShardedDDP: better partitioning when finetuning


## [0.1.1] - 2020-12-01
### Fixed
- make sure pip package includes header files (#221)

## [0.1.0] - 2020-12-01
### Added
- ShardedDataParallel with autoreduce (#157)
- cpu support for Pipe (#188)
- ShardedOptim: Distributed Grad Scaler (for torch AMP)  (#182)
- OSS-aware clip grads, bridge sharded states (#167)
- oss: add `rank_local_state_dict` staticmethod (#174)
- support for PyTorch 1.7.0 (#171)
- Add implementation of AdaScale (#139)

### Fixed
- pip package install (#196, #200)

## [0.0.3] - 2020-10-14
### Added
- multi-process pipe

### Fixed
- multiple OSS fixes
- MegaTron+OSS DDP fix

## [0.0.2] - 2020-08-28
### Added
- add ddp that works with oss with `reduce()` not `all_reduce()` (#19)
- support for PyTorch v1.6
- add mixed precision Adam (#40)
- Adam optimizer state scaling (#44)

### Fixed
- properly restore a sharded optim state (#39)
- OSS restore state to proper device (#46)
- optim/oss: support optimizers with additional step kwargs (#53)
- optim/oss: fix state cast (#56)
- fix eval for `oss_ddp` (#55)
- optim/oss: work correctly with LRScheduler (#58)

## [0.0.1] - 2020-07-31
- Initial release.
