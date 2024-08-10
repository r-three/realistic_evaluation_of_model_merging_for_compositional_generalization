import logging
import os
import re
from typing import Any, Callable, Dict, List, DefaultDict, Dict, Iterable
from itertools import chain
from collections import defaultdict, abc as container_abcs

import torch
import torch.optim as optim
from transformers import Adafactor
from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from src.train.TrainingConfig import TrainingConfig
from src.utils.io import *


def construct_optimizer(
    trainable_parameters: Dict[str, torch.Tensor], training_config: TrainingConfig
) -> Any:
    """
    Args:
        trainable_parameters:
        training_config:

    Raises:
        ValueError: Invalid optimizer name

    Returns:
        optimizer
    """
    optimizer_name = training_config.optimizer
    learning_rate = training_config.lr
    weight_decay = training_config.weight_decay
    weight_decay = training_config.weight_decay
    if weight_decay is None:
        weight_decay = 0.0

    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(
            trainable_parameters, lr=learning_rate, weight_decay=weight_decay
        )

    elif optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(
            trainable_parameters, lr=learning_rate, weight_decay=weight_decay, eps=1e-8
        )

    elif optimizer_name.lower() == "adafactor":
        optimizer = Adafactor(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            decay_rate=0,
            relative_step=False,
        )

    else:
        raise ValueError(f"Optimizer {optimizer_name} not implemented yet ")

    return optimizer


def construct_scheduler(optimizer: Any, training_config: TrainingConfig) -> Any:
    """

    Args:
        optimizer:
        training_config:

    Raises:
        ValueError: Invalid scheduler name

    Returns:
        scheduler
    """
    scheduler_name = training_config.scheduler
    num_batches = training_config.num_batches
    warmup_ratio = training_config.warmup_ratio

    num_warmup_steps = num_batches * warmup_ratio

    if scheduler_name == "polynomial_decay_with_warmup":
        return get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps, num_batches
        )

    elif scheduler_name == "exponential_decay":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer)

    elif scheduler_name == "linear_decay_with_warmup":
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_batches)

    elif scheduler_name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_batches)

    else:
        raise ValueError(f"scheduler {scheduler_name} not implemented")


def memoryEfficient_OptimizerLoadStateDict(optimizer, state_dict):
    # deepcopy, to be consistent with module API
    # state_dict = deepcopy(state_dict)
    # Validate the state_dict
    groups = optimizer.param_groups
    saved_groups = state_dict["param_groups"]

    if len(groups) != len(saved_groups):
        raise ValueError(
            "loaded state dict has a different number of " "parameter groups"
        )
    param_lens = (len(g["params"]) for g in groups)
    saved_lens = (len(g["params"]) for g in saved_groups)
    if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
        raise ValueError(
            "loaded state dict contains a parameter group "
            "that doesn't match the size of optimizer's group"
        )

    # Update the state
    id_map = {
        old_id: p
        for old_id, p in zip(
            chain.from_iterable((g["params"] for g in saved_groups)),
            chain.from_iterable((g["params"] for g in groups)),
        )
    }

    def cast(param, value, key=None):
        r"""Make a deep copy of value, casting all tensors to device of param."""
        if isinstance(value, torch.Tensor):
            # Floating-point types are a bit special here. They are the only ones
            # that are assumed to always match the type of params.
            # Make sure state['step'] is not casted https://github.com/pytorch/pytorch/issues/74424
            if key != "step":
                if param.is_floating_point():
                    value = value.to(param.dtype)
                value = value.to(param.device)
            return value
        elif isinstance(value, dict):
            return {k: cast(param, v, key=k) for k, v in value.items()}
        elif isinstance(value, container_abcs.Iterable):
            return type(value)(cast(param, v) for v in value)
        else:
            return value

    # Copy state assigned to params (and cast tensors to appropriate types).
    # State that is not assigned to params is copied as is (needed for
    # backward compatibility).
    state = defaultdict(dict)
    for k, v in state_dict["state"].items():
        if k in id_map:
            param = id_map[k]
            state[param] = cast(param, v)
        else:
            state[k] = v

    # Update parameter groups, setting their 'params' value
    def update_group(group, new_group):
        new_group["params"] = group["params"]
        return new_group

    param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
    optimizer.__setstate__({"state": state, "param_groups": param_groups})


def load_trainingStateToResumeFrom(
    training_state_directory_to_resume_training: str, model, optimizer, scheduler
):
    """

    Args:
        training_state_directory_to_resume_training:
        model:
        optimizer:
        scheduler:

    Returns:

    """

    checkpoint_toResumeTraining = getFile_inDirectory(
        training_state_directory_to_resume_training, "training_state"
    )

    resumeTraining_dict = torch.load(
        os.path.join(
            training_state_directory_to_resume_training, checkpoint_toResumeTraining
        )
    )

    memoryEfficient_OptimizerLoadStateDict(optimizer, resumeTraining_dict["optimizer"])

    if scheduler is not None:
        scheduler.load_state_dict(resumeTraining_dict["scheduler"], strict=False)
    batchIdx_toResumeFrom = resumeTraining_dict["num_batches"]

    if not set(resumeTraining_dict["model"].keys()).issubset(
        set(model.state_dict().keys())
    ):
        import ipdb

        ipdb.set_trace()

    model.load_state_dict(resumeTraining_dict["model"], strict=False)

    return model, optimizer, scheduler, batchIdx_toResumeFrom
