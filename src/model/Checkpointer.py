import logging
import os
from typing import Dict

import torch

from src.model.utils import *
from src.utils.io import *
from src.utils.utils import *

METRICS_PRIORITY = ["score_to_select_checkpoint", "loss", "batch_idx"]


class Checkpointer(object):
    def __init__(self, training_config, initialCheckpoint_idx):
        """
        Args:
            training_config:
            initial_checkpoint:
        """
        self.training_config = training_config
        self.initial_checkpoint = initialCheckpoint_idx

        self.runningSum_ofMetrics = {}
        self.numberOfUpdates_sinceLastCheckpoint = 0

        self.current_bestScore = 0
        self.numberOfCheckpoints_sinceBestCheckpoint = 0

        self.log_fp = os.path.join(
            self.training_config.experiment_dir, "training_log.json"
        )

        self._get_bestCheckpoint()

    def _get_bestCheckpoint(self):
        """
        If we are resuming training from a checkpoint, we check what the best checkpoint saved so far was
        """
        if os.path.exists(self.log_fp):
            list_scores = read_jsonl(self.log_fp)

            previous_bestScore = 0
            previous_bestCheckpointIdx = 0
            for score in list_scores:
                if score["score_to_select_checkpoint"] > previous_bestScore:
                    previous_bestScore = score["score_to_select_checkpoint"]
                    previous_bestCheckpointIdx = score["batch_idx"]

            # If we are resuming training, the initial checkpoint to resume training
            # should match the last checkpoint sored in the log
            assert list_scores[-1]["batch_idx"] == self.initial_checkpoint

            self.current_bestScore = previous_bestScore
            print(
                self.initial_checkpoint,
                previous_bestCheckpointIdx,
                self.training_config.checkpoint_frequency,
            )

            if previous_bestCheckpointIdx != 0:
                assert (
                    self.initial_checkpoint - previous_bestCheckpointIdx
                ) % self.training_config.checkpoint_frequency == 0

            self.numberOfCheckpoints_sinceBestCheckpoint = (
                self.initial_checkpoint - previous_bestCheckpointIdx
            ) // self.training_config.checkpoint_frequency

            print(self.current_bestScore, self.numberOfCheckpoints_sinceBestCheckpoint)

    def _is_bestCheckpoint(self, current_log: Dict):
        """

        Args:
            current_log:

        Returns:

        """
        current_score = getValueOfKey_inDictionary(current_log, METRICS_PRIORITY)
        return current_score > self.current_bestScore

    def _update_bestCheckpoint(self, current_log: Dict):
        """
        Args:
            current_log:

        Returns:

        """
        current_score = getValueOfKey_inDictionary(current_log, METRICS_PRIORITY)
        self.current_bestScore = current_score
        self.numberOfCheckpoints_sinceBestCheckpoint = 0

    def _save_checkpoint(
        self, trainable_parameters: Dict[str, torch.Tensor], save_fp: str
    ):
        """
        Args:
            trainable_parameters:
            save_name:
        """
        torch.save(
            trainable_parameters,
            save_fp,
        )

    def _save_trainingState(
        self, trainable_parameters, optimizer, scheduler, batch_idx: int, save_fp: str
    ):
        """
        Args:
            trainable_parameters:
            optimizer:
            scheduler:
            batch_idx:
            save_fp:

        Returns:

        """
        current_stateDict = {
            "num_batches": batch_idx,
            "model": trainable_parameters,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        }
        torch.save(current_stateDict, save_fp)

    def update_runningSumOfMetrics(self, current_metrics: Dict[str, float]):
        """
        Args:
            current_metrics:
        """
        self.runningSum_ofMetrics = addValues_inDict(
            self.runningSum_ofMetrics, current_metrics
        )
        self.numberOfUpdates_sinceLastCheckpoint += 1

    def _get_averageMetrics(self):
        """
        Get average metric per batch since the last time we got the average.

        Note that average is per example, not batch size
        (i.e. every gradient update, not every forward pass).
        """
        average_metric = {}
        for k in self.runningSum_ofMetrics.keys():
            average_metric[k] = float(
                "%.3f"
                % (
                    self.runningSum_ofMetrics[k]
                    / self.numberOfUpdates_sinceLastCheckpoint
                    / self.training_config.train_batch_size
                )
            )

        # Reset running dict_metrics and counter when we take average
        self.runningSum_ofMetrics = {}
        self.numberOfUpdates_sinceLastCheckpoint = 0

        return average_metric

    def _log_metricAndScores(self, batch_idx: int, evaluation_scores: Dict) -> Dict:
        """
        Args:
            batch_idx:
            evaluation_scores:

        Returns:
            _description_
        """
        current_log = {}
        current_log["batch_idx"] = batch_idx
        current_log.update(self._get_averageMetrics())
        current_log.update(evaluation_scores)

        append_json(current_log, self.log_fp, pretty_print=False)

        return current_log

    def checkpoint(
        self,
        trainable_parameters: Dict[str, torch.Tensor],
        optimizer,
        scheduler,
        evaluation_scores: Dict,
        batch_idx: int,
    ):
        """
        Handles checkpointing which means
        1) logging metrics and evaluation_scores
        2) saving the model if needed

        Args:
            trainable_parameters:
            optimizer,
            scheduler,
            evaluation_scores:
            batch_idx:

        Returns:
            current_log
        """
        current_log = self._log_metricAndScores(batch_idx, evaluation_scores)

        self.numberOfCheckpoints_sinceBestCheckpoint += 1

        # Save training state in training state directory
        trainingState_dir = os.path.join(
            self.training_config.experiment_dir, "training_state"
        )
        if not os.path.exists(trainingState_dir):
            os.makedirs(trainingState_dir)

        # Create checkpoint directory
        checkpoint_dir = os.path.join(
            self.training_config.experiment_dir, "checkpoints"
        )
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Save every checkpoint
        if self.training_config.should_save_every_checkpoint:
            self._save_checkpoint(
                trainable_parameters,
                os.path.join(checkpoint_dir, f"checkpoint_{batch_idx}.pt"),
            )

        # Save only best checkpoint
        if self._is_bestCheckpoint(current_log):
            deleteFiles_inDirectory(checkpoint_dir, "best")
            self._save_checkpoint(
                trainable_parameters,
                os.path.join(checkpoint_dir, f"best_checkpoint_{batch_idx}.pt"),
            )

        # Ignore saving the model if we are just evaluating at the beginning
        if batch_idx > 0:
            if self.training_config.should_save_training_state:
                deleteFiles_inDirectory(trainingState_dir, "training_state")
                self._save_trainingState(
                    trainable_parameters,
                    optimizer,
                    scheduler,
                    batch_idx,
                    os.path.join(trainingState_dir, f"training_state_{batch_idx}.pt"),
                )

        if self._is_bestCheckpoint(current_log):
            self._update_bestCheckpoint(current_log)

        logging.info(f"Finished {batch_idx} batches with log {current_log}")

        return self.numberOfCheckpoints_sinceBestCheckpoint
