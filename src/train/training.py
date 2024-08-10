import argparse
import logging
import os
from typing import Dict

import torch
from tqdm import tqdm

from src.data.batches import getMultipleEpochs_ofBatches
from src.data.Dataset import LanguageDataset, VisionDataset
from src.data.dataset_reader.DatasetReader import DatasetReader
from src.data.readers import get_datasetReader
from src.eval.evaluation import evaluate_onDatasets
from src.eval.utils import *
from src.model.Checkpointer import Checkpointer
from src.model.load_model import *
from src.model.utils import get_modelParameters
from src.train.TrainingConfig import TrainingConfig
from src.train.utils import *
from src.train.utils import construct_optimizer, construct_scheduler
from src.utils.Config import *
from src.utils.config_utils import *
from src.utils.stats import *
from src.utils.utils import *


def checkpointing(
    model,
    trainable_parameters: Dict[str, torch.Tensor],
    optimizer,
    scheduler,
    checkpointer: Checkpointer,
    training_config: TrainingConfig,
    cached_datasetReaders: Dict[str, DatasetReader],
    batch_idx: int,
) -> (int, Dict[str, DatasetReader]):
    logging.info(f"Evaluating checkpoint")

    evaluation_scores = {}
    prediction_dir = os.path.join(
        training_config.experiment_dir, "predictions", f"batch_{batch_idx}"
    )
    if training_config.should_eval_train:
        train_scores, _, cached_datasetReaders = evaluate_onDatasets(
            model,
            training_config.get_modelConfig(),
            update_evaluationConfig(
                training_config.get_evaluationConfig(), {"split": "train"}, {}
            ),
            prediction_dir,
            cached_datasetReaders,
            device,
        )
        evaluation_scores.update(train_scores)
        evaluation_scores["score_to_select_checkpoint"] = train_scores["train"][
            "average"
        ]
    if training_config.should_eval_validation:
        validation_scores, _, cached_datasetReaders = evaluate_onDatasets(
            model,
            training_config.get_modelConfig(),
            update_evaluationConfig(
                training_config.get_evaluationConfig(), {"split": "validation"}, {}
            ),
            prediction_dir,
            cached_datasetReaders,
            device,
        )

        evaluation_scores.update(validation_scores)

        if training_config.get_evaluationConfig().task_mixture is None:
            score_toSelectCheckpoint = validation_scores["validation"]["average"]
        else:
            score_toSelectCheckpoint = validation_scores["average"]

        evaluation_scores["score_to_select_checkpoint"] = score_toSelectCheckpoint

    numCheckpoints_sinceBestCheckpoint = checkpointer.checkpoint(
        trainable_parameters, optimizer, scheduler, evaluation_scores, batch_idx
    )

    return numCheckpoints_sinceBestCheckpoint, cached_datasetReaders


def train(device, training_config: TrainingConfig):

    set_seeds(training_config.seed)

    if training_config.get_modelConfig().language_or_vision == "vision":
        classifier_head, cached_models = get_classifierHeadAndCLIPModel(
            training_config.get_modelConfig(), training_config, None, {}
        )
    else:
        classifier_head = None
        cached_models = {}

    model, _ = load_model(
        training_config.get_modelConfig(),
        classifier_head,
        cached_models,
        device=device,
    )

    trainable_parameters = get_modelParameters(model)

    optimizer = construct_optimizer(trainable_parameters.values(), training_config)

    scheduler = None
    if training_config.scheduler is not None:
        scheduler = construct_scheduler(optimizer, training_config)

    dataset_reader, cached_singleDatasetReaders = get_datasetReader(
        training_config.train_task_mixture,
        training_config.mixture_subset_size,
        training_config.mixture_subset_id,
        training_config.get_datasetConfig(),
        cached_singleDatasetReaders={},
    )

    if training_config.get_modelConfig().language_or_vision == "language":
        dataset = LanguageDataset(
            dataset_reader.get_dataset("train"),
            training_config.get_datasetConfig(),
            model.tokenize_fn,
            "train",
            device=device,
        )
    else:
        dataset = VisionDataset(
            dataset_reader.get_dataset("train"),
            training_config.get_datasetConfig(),
            model.preprocess_fn,
            "train",
            device=device,
        )

    train_iterator = getMultipleEpochs_ofBatches(
        dataset, training_config.micro_train_batch_size, should_shuffle=True
    )

    # Load training state if resuming training
    if training_config.training_state_directory_to_resume_training:
        (
            model,
            optimizer,
            scheduler,
            batchIdx_toResumeFrom,
        ) = load_trainingStateToResumeFrom(
            training_config.training_state_directory_to_resume_training,
            model,
            optimizer,
            scheduler,
        )
    else:
        if training_config.checkpoint_to_initialize_training:
            initial_checkpoint = torch.load(
                training_config.checkpoint_to_initialize_training
            )

            if not set(initial_checkpoint.keys()).issubset(
                set(model.state_dict().keys())
            ):
                import ipdb

                ipdb.set_trace()

            model.load_state_dict(initial_checkpoint, strict=False)

        batchIdx_toResumeFrom = 0

    checkpointer = Checkpointer(training_config, batchIdx_toResumeFrom)

    if training_config.should_eval_before_training:
        logging.info(f"Evaluating before training")
        _, cached_singleDatasetReaders = checkpointing(
            model,
            trainable_parameters,
            optimizer,
            scheduler,
            checkpointer,
            training_config,
            cached_singleDatasetReaders,
            batch_idx=batchIdx_toResumeFrom,
            device=device,
        )

    if (
        training_config.use_bfloat16_during_training
        or training_config.use_fp16_during_training
    ):
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    assert (
        training_config.train_batch_size % (training_config.micro_train_batch_size) == 0
    )
    gradient_accumulation_factor = training_config.train_batch_size // (
        training_config.micro_train_batch_size
    )

    effectiveBatch_metrics = {}

    for i in tqdm(range(training_config.num_batches * gradient_accumulation_factor)):
        batch_idx = i // (gradient_accumulation_factor)

        # Skip the batches we have already seen during training
        if batchIdx_toResumeFrom != 0 and batch_idx <= batchIdx_toResumeFrom:
            continue

        model.train()

        train_batch = next(train_iterator)
        if training_config.use_bfloat16_during_training:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, batch_metrics = model(train_batch)
                loss = loss / gradient_accumulation_factor
            scaler.scale(loss).backward()
        elif training_config.use_fp16_during_training:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, batch_metrics = model(train_batch)
                loss = loss / gradient_accumulation_factor
            scaler.scale(loss).backward()
        else:
            loss, batch_metrics = model(train_batch)
            loss = loss / gradient_accumulation_factor
            loss.backward()

        effectiveBatch_metrics = addValues_inDict(effectiveBatch_metrics, batch_metrics)

        if (i + 1) % gradient_accumulation_factor == 0:
            # Log effective batch metrics and reset
            checkpointer.update_runningSumOfMetrics(effectiveBatch_metrics)
            effectiveBatch_metrics = {}

            # Clip norm of gradient
            if training_config.norm_to_clip_gradient is not None:
                # Unscale gradient if using bfloat16 so clipping can be correct magnitude
                if (
                    training_config.use_bfloat16_during_training
                    or training_config.use_fp16_during_training
                ):
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters.values(),
                    training_config.norm_to_clip_gradient,
                )

            # Take a gradient step
            # Unscale gradient for bfloat16 training
            if (
                training_config.use_bfloat16_during_training
                or training_config.use_fp16_during_training
            ):
                scaler.step(optimizer)
                scaler.update()
                if training_config.scheduler is not None:
                    scheduler.step()
            else:
                optimizer.step()
                if training_config.scheduler is not None:
                    scheduler.step()

            # Reset optimizer
            optimizer.zero_grad()

            # Checkpoint and evaluate if necessary
            if (batch_idx + 1) % training_config.checkpoint_frequency == 0:
                (
                    numCheckpoints_sinceBestCheckpoint,
                    cached_singleDatasetReaders,
                ) = checkpointing(
                    model,
                    trainable_parameters,
                    optimizer,
                    scheduler,
                    checkpointer,
                    training_config,
                    cached_singleDatasetReaders,
                    batch_idx=batch_idx,
                    device=device,
                )

                if training_config.use_early_stopping:
                    if (
                        numCheckpoints_sinceBestCheckpoint
                        >= training_config.early_stopping_num_checkpoints_without_improvement
                    ):
                        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = addConfigArguments_toParser(
        parser,
        add_trainingArguments=True,
        add_inferenceArguments=False,
        add_mergingArguments=False,
    )
    parser.add_argument("--fifo_file", type=str)
    parser.add_argument("--slurm_job_id")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting training")

    training_config, _ = construct_configs(args, "train", is_merging=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(device, training_config)
