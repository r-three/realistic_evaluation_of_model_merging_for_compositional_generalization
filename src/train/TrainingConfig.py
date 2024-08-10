import datetime
import json
import os
from shutil import copytree, ignore_patterns
from typing import Dict, List

from src.data.DatasetConfig import DatasetConfig
from src.eval.EvaluationConfig import EvaluationConfig
from src.model.ModelConfig import ModelConfig
from src.utils.Config import Config
from src.utils.io import *


class TrainingConfig(Config):
    def __init__(
        self,
        model_config: ModelConfig,
        trainingDataset_config: DatasetConfig,
        evaluation_config: EvaluationConfig,
        config_filepaths: List[str] = None,
        update_dict: Dict = None,
    ):
        super().__init__()

        """
        other configs that are part of training run 
        """
        self.model_config = model_config
        self.train_dataset_config = trainingDataset_config
        self.evaluation_config = evaluation_config
        self.train_task_mixture = None
        self.mixture_subset_size = None
        self.mixture_subset_id = None

        """
        training run parameters 
        """
        self.micro_train_batch_size = None
        self.train_batch_size = None
        self.num_batches = None
        self.use_bfloat16_during_training = None
        self.use_fp16_during_training = None
        self.num_epochs = None

        """
        checkpoint parameters 
        """
        self.checkpoint_to_initialize_training = None
        self.checkpoint_frequency = None
        self.use_early_stopping = None
        self.early_stopping_num_checkpoints_without_improvement = None
        self.should_save_every_checkpoint = None
        self.should_save_training_state = None
        self.should_eval_before_training = None
        self.should_eval_validation = None
        self.should_eval_train = None
        self.training_state_directory_to_resume_training = None

        """
        optimization parameters 
        """
        self.lr = None
        self.optimizer = None
        self.scheduler = None
        self.warmup_ratio = None
        self.weight_decay = None
        self.norm_to_clip_gradient = None

        """
        reproducabilty parameters 
        """
        self.seed = None
        self.experiment_name = None
        self.experiment_dir = None
        self.fifo_file = None  # Used to store the experiment_directory name to allow for automatic resubmission of jobs
        self.slurm_job_id = None

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

        if self.experiment_dir is None:
            self.get_experimentDir()
            configs_dir = os.path.join(self.experiment_dir, "configs")
            os.makedirs(configs_dir, exist_ok=True)
            self._save_config(
                os.path.join(configs_dir, "training_run_config.json"),
            )
            self.train_dataset_config._save_config(
                os.path.join(configs_dir, "training_dataset_config.json"),
            )
            self.model_config._save_config(
                os.path.join(configs_dir, "model_config.json"),
            )
            self.evaluation_config._save_config(
                os.path.join(configs_dir, "evaluation_run_config.json"),
            )
            evaluationDataset_config = self.evaluation_config.get_datasetConfig()
            evaluationDataset_config._save_config(
                os.path.join(configs_dir, "evaluation_dataset_config.json"),
            )

    def get_experimentDir(self):
        """
        Create experiment directory and assign it in TrainingConfig
        """
        now = datetime.datetime.now()
        timestamp = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second
        )
        self.experiment_dir = "exp_out"

        if self.train_task_mixture is not None:
            self.experiment_dir = os.path.join(
                self.experiment_dir,
                self.get_datasetConfig().instruction_format,
                self.train_task_mixture,
            )

        self.experiment_dir = os.path.join(
            self.experiment_dir,
            self.train_dataset_config.get_experimentDir(),
            self.model_config.get_experimentDir(),
        )
        if self.experiment_name is not None:
            self.experiment_dir = os.path.join(
                self.experiment_dir, self.experiment_name
            )

        self.experiment_dir = os.path.join(self.experiment_dir, timestamp)
        os.makedirs(self.experiment_dir, exist_ok=True)

        copytree(
            os.path.join(os.environ["REM_ROOT"], "src"),
            os.path.join(self.experiment_dir, "src"),
            ignore=ignore_patterns("*.pyc", "tmp*"),
        )

    def get_modelConfig(self) -> ModelConfig:
        """
        Returns:
            model_config
        """
        return self.model_config

    def get_evaluationConfig(self) -> EvaluationConfig:
        """

        Returns:
            evaluation_config
        """
        return self.evaluation_config

    def get_datasetConfig(self) -> DatasetConfig:
        """

        Returns:
            datset_config
        """
        return self.train_dataset_config
