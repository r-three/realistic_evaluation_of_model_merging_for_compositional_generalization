import json
from typing import Dict, List

from src.data.DatasetConfig import DatasetConfig
from src.utils.Config import Config


class EvaluationConfig(Config):
    def __init__(
        self,
        evaluationDataset_config: DatasetConfig,
        config_filepaths: List[str] = None,
        update_dict: Dict[str, str] = None,
    ):
        """

        Args:
            evaluationDataset_config:
            config_filepaths: Defaults to None.
            update_dict: Defaults to None.
        """
        super().__init__()
        self.evaluationDataset_config = evaluationDataset_config

        # task mixture for inference
        self.task_mixture = None
        self.mixture_subset_size = None
        self.mixture_subset_id = None

        self.max_gen_len = None
        self.sample_tokens = None
        self.eval_batch_size = None

        self.length_normalization = None

        self.overwrite_previous_run = None

        self.num_samples = None

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

    def get_datasetConfig(self) -> DatasetConfig:
        """

        Returns:
            EvaluationDataset_config
        """
        return self.evaluationDataset_config
