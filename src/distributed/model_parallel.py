from typing import Any, cast

import torch
from loguru import logger
from torch import nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed._composable.replicate import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

"""disable_fsdp_gradient_division(model) sets set_gradient_divide_factor(1.0) for all FSDP modules, 
so gradient synchronization produces the sum of gradients instead of their average — for example, 
with two processes, g0 + g1 instead of (g0 + g1) / 2."""


def disable_fsdp_gradient_division(model: nn.Module) -> None:
    """Disable FSDP's automatic gradient division for all FSDP modules.

    Sets gradient_divide_factor=1.0 so gradient scaling can be handled manually (e.g., by normalizing loss with global token count).
    """
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)


# DP PARALISM ?
# Distributed Data Parallel
def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
    enable_compiled_autograd: bool,
):

    if enable_compile:
        if enable_compiled_autograd:
            torch._dynamo.config.optimize_ddp = "python_reducer_without_compiled_forward"
            # Therefore, PyTorch uses a Python reducer, allowing gradient communication to be
            # represented in the captured backward graph and potentially optimized alongside computation.
        else:
            # ddp_optimizer -> Splits compiled model graphs around DDP gradient bucket boundaries to overlap
            # computation and communication
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"
            """
            This overlap helps improve training performance. 
            However, compiling a large backward graph can delay the gradient-ready hooks until the compiled region finishes.
            ddp_optimizer addresses this by partitioning the compiled graph along DDP bucket boundaries, allowing synchronization to overlap with later backward computation
            """
    # data parallelism using PyTorch's composable replicate() AP vs DistributedDataParallel (DDP)
    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logger.info("Applied DDP to the model")


# FSDP = Fully Sharded Data Parallel in PyTorch.

# It's another technique for distributed training, similar to DDP, but with one major difference:
# - DDP: Every GPU stores a complete copy of the model's parameters, gradients, and optimizer states.
# - FSDP: The model's parameters, gradients, and optimizer states are sharded (split) across GPUs to reduce memory consumption.


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    dp_replicate_efsdp_mesh: DeviceMesh | None = None,
):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            # For PP, by default do not reshard after forward to avoid per-microbatch all-gathers, which can be expensive and non-overlapped
            reshard_after_forward = not pp_enabled

        #### HERE IS IMPORTANT TOPIC Trade off FSDP VS PP -> we can save communication time by spliiting pipeline
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    ## PP model splitting or creation
    tok_embeddings: Any = model.tok_embeddings
    layers = cast(nn.ModuleDict, model.layers)
    norm: Any = model.norm
    output: Any = model.output

    if tok_embeddings is not None:
        fully_shard(
            tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    for transformer_block in layers.values():
        if transformer_block.moe_enabled and ep_degree > 1:
            raise NotImplementedError("TO DO")

        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # As an optimization, do not reshard_after_forward the last layers by default since FSDP would prefetch them immediately after the forward pass
    if norm is not None and output is not None:
        fully_shard(
            [norm, output],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    fully_shard(model, **fsdp_config)
    disable_fsdp_gradient_division(model)

    # NOTE: set up explicit prefetching when EP is enabled, as D2H syncs in EP could interfere with implicit prefetching in FSDP
    if ep_degree == 1:
        return

    ### FORWARD
    transformer_blocks: list[Any] = list(layers.values())
    next_transformer_blocks = transformer_blocks[1:] + [None]

    if tok_embeddings is not None and len(layers) > 0:
        tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])  # ty:ignore[call-non-callable, unresolved-attribute]

    for transformer_block, next_transformer_block in zip(
        transformer_blocks, next_transformer_blocks
    ):
        if next_transformer_block is not None:
            if next_transformer_block.moe_enabled:
                transformer_block.set_modules_to_forward_prefetch(
                    [next_transformer_block, next_transformer_block.moe.experts]
                )
            else:
                transformer_block.set_modules_to_forward_prefetch([next_transformer_block])

        elif norm is not None and output is not None:
            transformer_block.set_modules_to_forward_prefetch([norm, output])

    ### BACKWARD -> the same logic as forward but in reverse

    reversed_transformer_blocks: list[Any] = list(reversed(list(layers.values())))
    prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

    if norm is not None and output is not None and len(layers) > 0:
        output.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])  # ty:ignore[call-non-callable, unresolved-attribute]

    for transformer_block, prev_transformer_block in zip(
        reversed_transformer_blocks, prev_transformer_blocks
    ):
        if prev_transformer_block is not None:
            if prev_transformer_block.moe_enabled:
                transformer_block.set_modules_to_backward_prefetch(
                    [prev_transformer_block, prev_transformer_block.moe.experts]
                )
            else:
                transformer_block.set_modules_to_backward_prefetch([prev_transformer_block])
        elif tok_embeddings is not None:
            transformer_block.set_modules_to_backward_prefetch([tok_embeddings])
