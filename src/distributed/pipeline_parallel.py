import copy
import math
from collections.abc import Callable

import torch
from loguru import logger
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    ScheduleZBVZeroBubble,
    _PipelineSchedule,
    get_schedule_class,
)

from src.components.loss import LossFunction
from src.config import JobConfig
from src.distributed import ParallelDims

__all__ = [
    "build_pipeline_schedule",
    "generate_llm_fqn_per_model_part",
    "pipeline_llm",
    "pipeline_module_split",
]


def build_pipeline_schedule(
    job_config: JobConfig, stages: list[PipelineStage], loss_fn: Callable
) -> _PipelineSchedule:

    schedule_class = get_schedule_class(job_config.parallelism.pipeline_parallel_schedule)

    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)

    # micro batches
    microbatch_size = job_config.parallelism.pipeline_parallel_microbatch_size
    batch_size = job_config.training.local_batch_size
    assert batch_size % microbatch_size == 0, (batch_size, microbatch_size)
    n_microbatches = batch_size // microbatch_size

    # We expect that the number of local stages (`len(stages)`) is the same across all ranks
    # - ScheduleZBVZeroBubble (requires exactly two stages per rank)
    # 1 Rank can contain multiple stages,
    # pipeline_parallel_degree means the number of ranks participating in one pipeline-parallel group.
    num_total_stages = job_config.parallelism.pipeline_parallel_degree * len(stages)
    if n_microbatches < num_total_stages:
        logger.warning(
            f"Number of microbatches ({n_microbatches}) is less than the total number "
            f"of stages ({num_total_stages}) which may result in a bubble in the pipeline."
        )

    if looped_schedule:
        schedule = schedule_class(
            stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            scale_grads=False,
        )
    else:
        # while single-stage schedules expect exactly one stage per rank.
        schedule = schedule_class(
            stages[0],
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            scale_grads=False,
        )

    logger.info(
        f"Using pipeline schedule {job_config.parallelism.pipeline_parallel_schedule} "
        f"with {n_microbatches} microbatches and {num_total_stages} stages."
    )

    return schedule


def _build_stage_from_modules(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    stage_idx: int,
    module_names: list[str],
    num_stages: int,
    device: torch.device,
) -> tuple[PipelineStage, nn.Module]:

    # make a copy of the model so we prune away stuff
    model = copy.deepcopy(whole_model)

    # Create a set of modules to keep for faster lookup
    modules_to_keep = set(module_names)

    for module_name, module_value in model.named_modules():
        if isinstance(module_value, (nn.ModuleDict, nn.ModuleList)):
            layers_to_keep = {
                name.split(".", 1)[1]
                for name in modules_to_keep
                if name.startswith(f"{module_name}.")
            }
            # layers_to_keep = {"0" "1"}

            if layers_to_keep:
                if isinstance(module_value, nn.ModuleDict):
                    for layer_name in list(module_value.keys()):
                        if layer_name not in layers_to_keep:
                            del module_value[layer_name]
                elif isinstance(module_value, nn.ModuleList):
                    indices_to_keep = {
                        int(idx) for idx in layers_to_keep if idx.isdigit()
                    }  # numbers

                    new_layers = nn.ModuleList(
                        [layer for i, layer in enumerate(module_value) if i in indices_to_keep]
                    )
                    setattr(model, module_name, new_layers)

            else:
                # No layers from this structure needed, set to empty structure
                if isinstance(module_value, nn.ModuleDict):
                    setattr(model, module_name, nn.ModuleDict())
                elif isinstance(module_value, nn.ModuleList):
                    setattr(model, module_name, nn.ModuleList())

        elif module_name not in modules_to_keep:
            # Handle simple module attributes (e.g., "linear", "norm")
            # Replace with None
            setattr(model, module_name, None)

    stage = PipelineStage(
        model,
        stage_idx,
        num_stages,
        device,
        group=pp_mesh.get_group("pp"),
    )
    return stage, model


def pipeline_module_split(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    pp_schedule: str,
    device: torch.device,
    module_names_per_stage: list[list[str]],
) -> tuple[list[PipelineStage], list[nn.Module]]:

    pp_rank = pp_mesh.get_local_rank()
    pp_degree = pp_mesh.size()

    num_stages = len(module_names_per_stage)
    stages = []
    models = []

    schedule_class = get_schedule_class(pp_schedule)
    style = "v" if schedule_class is ScheduleZBVZeroBubble else "loop"

    # how many pipeline partitions the model is split into
    assert num_stages % pp_degree == 0, (num_stages, pp_degree)
    stages_per_rank = num_stages // pp_degree

    if style == "v":
        # V-shaped placement used by ZBVZeroBubble. Each rank owns exactly two stages: rank i gets stage i from the first half and stage num_stages - 1 - i from the second half.
        # With 8 stages and 4 ranks, the rank path is 0, 1, 2, 3, 3, 2, 1, 0; rank 0 owns stages (0, 7), rank 1 owns (1, 6), and so on.
        assert stages_per_rank == 2, (stages_per_rank,)

        # This is a “V-shaped” assignment:
        # Example [(0, 7), (1, 6), (2, 5), (3, 4)]

        stage_v_pairs = list(zip(range(pp_degree), range(num_stages - 1, pp_degree - 1, -1)))
        stage_indices = stage_v_pairs[pp_rank]
    elif style == "loop":
        stage_indices = tuple(pp_rank + s * pp_degree for s in range(stages_per_rank))
        # rank 0: stages 0 and 4
        # rank 1: stages 1 and 5
        # rank 2: stages 2 and 6
        # rank 3: stages 3 and 7
    else:
        raise ValueError(f"Unknown pipeline schedule style: {style}")

    for stage_idx in stage_indices:
        module_names = module_names_per_stage[stage_idx]
        stage, model_chunk = _build_stage_from_modules(
            whole_model, pp_mesh, stage_idx, module_names, num_stages, device
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx} with modules {module_names}"
        )
        stages.append(stage)
        models.append(model_chunk)

    return stages, models


def generate_llm_fqn_per_model_part(
    num_stages: int,
    num_layers: int,
    input_weight: int = 1,
    output_weight: int = 1,
) -> list[list[str]]:

    assert num_stages >= 1, num_stages

    if num_stages == 1:
        layer_names = [f"layers.{i}" for i in range(num_layers)]
        return [["tok_embeddings"] + layer_names + ["norm", "output"]]

    # Calculate the number of effective layers, including input and output weights
    num_effective_layers = num_layers + input_weight + output_weight

    assert num_stages <= num_effective_layers, (num_stages, num_effective_layers)

    # Calculate layers per stage (distribute evenly)
    layers_per_stage = num_effective_layers // num_stages  # most stages get X number of layers
    extra_layers = num_effective_layers % num_stages  # How many stages get X extra layers

    assert layers_per_stage > 0, (layers_per_stage, num_stages, num_effective_layers)
    assert input_weight <= layers_per_stage, (input_weight, layers_per_stage)
    assert output_weight <= layers_per_stage, (output_weight, layers_per_stage)

    module_names_per_stage = []
    current_layer = 0

    for stage_idx in range(num_stages):
        # The first `extra_layers` stages get an additional layer to account for any remainder
        effective_layers_for_stage = layers_per_stage + (1 if stage_idx < extra_layers else 0)

        # Subtract weight for special modules on first/last stages
        num_transformer_layers = effective_layers_for_stage
        if stage_idx == 0:
            num_transformer_layers -= input_weight
        if stage_idx == num_stages - 1:
            num_transformer_layers -= output_weight

        stage_modules = []
        if stage_idx == 0:
            stage_modules.append("tok_embeddings")

        for _ in range(num_transformer_layers):
            if current_layer < num_layers:
                stage_modules.append(f"layers.{current_layer}")
                current_layer += 1

        if stage_idx == num_stages - 1:
            stage_modules.extend(["norm", "output"])

        module_names_per_stage.append(stage_modules)

    return module_names_per_stage


def pipeline_llm(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: torch.device,
    num_layers: int,
    parallelize_fn: Callable[[nn.Module, ParallelDims, JobConfig, torch.device], nn.Module],
    loss_fn: LossFunction,
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:

    pp_mesh = parallel_dims.get_mesh("pp")

    # Determine the number of virtual stages based on schedule type
    schedule_class = get_schedule_class(job_config.parallelism.pipeline_parallel_schedule)

    # Single stage: each rank holds exactly one stage (e.g. GPipe, 1F1B)
    # Multi stage: each rank can hold multiple stages (e.g. Interleaved 1F1B). This means rank 0 may hold stages 0, 4 rank 1 may hold stages 1, 5, etc.
    #  Interleaved 1F1B Better than GPIPE in terms of making bubble smaller

    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)
    layers_per_stage = job_config.parallelism.pipeline_parallel_layers_per_stage

    # How many "layers" is the embedding layer equivalent to?
    # Embedding layer is not a real layer, but we assume to treat it as such
    input_weight = job_config.parallelism.pipeline_parallel_first_stage_less_layers

    # How many "layers" is the output layer (norm + output) equivalent to?
    output_weight = job_config.parallelism.pipeline_parallel_last_stage_less_layers

    if layers_per_stage is not None:  # we  split
        # Calculate number of virtual stages needed (using ceiling division)
        # This allows for unequal distribution where stages can differ by at most 1 layer

        num_virtual_stages = math.ceil(
            (num_layers + input_weight + output_weight) / layers_per_stage
        )

        assert num_virtual_stages % parallel_dims.pp == 0, (
            num_virtual_stages,
            parallel_dims.pp,
        )

        stages_per_rank = num_virtual_stages // parallel_dims.pp

        assert not is_single_stage_schedule or stages_per_rank == 1, (
            is_single_stage_schedule,
            stages_per_rank,
        )
        assert is_single_stage_schedule or stages_per_rank >= 2, (
            is_single_stage_schedule,
            stages_per_rank,
        )

    else:  # we dont split
        stages_per_rank = 1 if is_single_stage_schedule else 2
        num_virtual_stages = parallel_dims.pp * stages_per_rank

    # module_fqns_per_model_part is a config value that tells the program which model submodules belong to each pipeline-parallel partition/stage
    module_names_per_stage = job_config.parallelism.module_fqns_per_model_part

    if module_names_per_stage is None:
        module_names_per_stage = generate_llm_fqn_per_model_part(
            num_virtual_stages, num_layers, input_weight, output_weight
        )

    for i, stage_ms in enumerate(module_names_per_stage):
        logger.debug(f"Stage {i}: {stage_ms}")

    # here we receive the full model and the stage-to-module mapping
    stages, model_parts = pipeline_module_split(
        model,
        pp_mesh,
        job_config.parallelism.pipeline_parallel_schedule,
        device,
        module_names_per_stage,
    )

    # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
    # We need to iterate through model_parts to apply SPMD parallelisms, compilation, optimizer, and checkpointing

    for i, m in enumerate(model_parts):
        # apply SPMD-style PT-D techniques
        m = parallelize_fn(m, parallel_dims, job_config)
        model_parts[i] = m
        # NOTE: this is to update the model in the stage in case the model is modified e.g. by torch.compile
        stages[i].submod = m

    # iterate over every stage model chunk on this rank
    # apply the model-parallelization function to that chunk
    # this is where tensor parallel / sharding / compile / other PT-D style transformations happen
    # then replace the original chunk with the transformed one
    # then update the corresponding PipelineStage object to use that new chunk

    pp_schedule = build_pipeline_schedule(job_config, stages, loss_fn)

    # This is used in the train loop to determine whether to pass in the input_ids and labels
    has_first_stage = False
    has_last_stage = False
    for stage in stages:
        if stage.is_first:
            has_first_stage = True
        if stage.is_last:
            has_last_stage = True

    return pp_schedule, model_parts, has_first_stage, has_last_stage
