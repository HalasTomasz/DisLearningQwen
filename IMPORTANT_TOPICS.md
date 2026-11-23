# Important Topics

- Ring Attention.
- Fully Sharded Data Parallel (FSDP): `mp_policy` and `reshard_after_forward`.
- Distributed Data Parallel (DDP): `ddp_optimizer` and `bucket_cap_mb`.
- `ScheduleZBVZeroBubble`: stages, looping versus V-shaped placement,
  SPMD-style PT-D techniques, pipeline schedules, and one-forward-one-backward (1F1B).
- Pipeline parallelism (PP) versus FSDP.
- Activation checkpointing.
- Mixed precision training: loss scaling to prevent gradient underflow,
  weight updates in FP32, and selected arithmetic operations in FP32.
- Profiling: the warmup phase.

## HSDP versus FSDP

With `dp_shard=4` and `dp_replicate=1`, FSDP distributes one copy of the model
across four GPUs.

With `dp_shard=4` and `dp_replicate=2`, Hybrid Sharded Data Parallel (HSDP)
maintains two copies of the model, each distributed across four GPUs,
for a total of eight GPUs.

FSDP2 uses `DTensor` and the `fully_shard()` function.
