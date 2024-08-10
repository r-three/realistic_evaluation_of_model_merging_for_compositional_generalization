import json
import os
from typing import Any, Dict, List, Callable

from src.utils.Config import Config


class MergingConfig(Config):
    def __init__(
        self,
        config_filepaths=None,
        update_dict=None,
    ):
        """
        Args:
            configDict_toInitializeFrom:
            update_dict:
        """
        super().__init__()

        self.method = None
        self.merging_lambda = None

        # task mixture to merge
        self.task_mixture = None

        # Whether to merge the PEFT weights (i.e. A and B in LoRA or IA3 into pretrained weight)
        self.merge_peft_weights = None

        # Hyperparameters for computing statistics
        self.use_true_fisher = None
        self.split_to_compute_statistic = None
        self.fisher_approximation = None

        # Hyperparameters for MaTS
        self.checkpoint_for_initalization = None
        self.number_of_iterations = None

        # checkpoint descriptor for loading checkpoints to merge
        self.checkpoint_descriptor = None
        # For DARE Task Arithmetic
        self.dropout_probability = None

        self.save_model = None
        self.experiment_name = None

        # Update config with values from list of files
        if config_filepaths:
            for filename in config_filepaths:
                super()._update_fromDict(
                    json.load(open(filename)),
                    assert_keyInUpdateDict_isValid=True,
                )

        # Update config with values from dict
        if update_dict:
            super()._update_fromDict(
                update_dict,
                assert_keyInUpdateDict_isValid=True,
            )

    def get_fisherArguments(self):
        if self.use_true_fisher:
            fisher_estimate = "true_fixed"
        else:
            fisher_estimate = "empirical"

        return f"{self.fisher_approximation}_{fisher_estimate}_fisher_{self.split_to_compute_statistic}"

    def get_experimentDir(self):

        if self.merge_peft_weights is not None and self.merge_peft_weights:
            experiment_dir = os.path.join("merge_peft_weights", self.method)
        else:
            experiment_dir = self.method

        if (
            self.method == "fisher_merging"
            or self.method == "mats"
            or self.method == "regmean"
        ):
            experiment_dir = os.path.join(experiment_dir, self.get_fisherArguments())

        if self.number_of_iterations is not None:
            experiment_dir = os.path.join(
                experiment_dir, f"{self.number_of_iterations}_iterations"
            )

        if self.checkpoint_for_initalization is not None:
            experiment_dir = os.path.join(
                experiment_dir,
                "_".join(self.checkpoint_for_initalization.split("/")[-3:-2]),
            )

        if self.merging_lambda is not None:
            experiment_dir = os.path.join(
                experiment_dir,
                f"merging_lambda_{self.merging_lambda}",
            )

        if self.dropout_probability is not None:
            experiment_dir = os.path.join(
                experiment_dir,
                f"dropout_probability_{self.dropout_probability}",
            )

        if self.experiment_name is not None:
            experiment_dir = os.path.join(experiment_dir, self.experiment_name)

        return experiment_dir
