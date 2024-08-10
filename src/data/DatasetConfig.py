import json
import os
from typing import Any, Callable, Dict, List

from src.utils.Config import Config


class DatasetConfig(Config):
    def __init__(
        self,
        config_filepaths: List[str] = None,
        update_dict: Dict[str, str] = None,
    ):
        """

        Args:
            config_filepaths: Defaults to None.
            update_dict: Defaults to None.
        """
        super().__init__()

        self.dataset = None
        self.split = None
        self.max_number_of_samples = None

        # For language tasks only
        self.instruction_format = None
        self.template_idx = None
        self.language_code = None
        self.language = None

        # For vision tasks
        self.domain = None
        self.task = None
        self.shift_lbls = None  # Whether to shift labels for the domain

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

    def get_experimentDir(self):
        """

        Returns:

        """
        experiment_dir = ""

        if self.dataset is not None:
            experiment_dir = os.path.join(
                experiment_dir, self.instruction_format, self.dataset
            )

        if self.language_code is not None:
            experiment_dir = os.path.join(experiment_dir, self.language_code)

        if self.language is not None:
            experiment_dir = os.path.join(experiment_dir, self.language)

        if self.domain is not None:
            assert self.task is not None
            experiment_dir = os.path.join(
                experiment_dir, self.domain, f"task_{self.task}"
            )

        return experiment_dir

    def __hash__(self):
        return hash(
            (
                self.instruction_format,
                self.dataset,
                self.language_code,
                self.language,
                self.domain,
                self.task,
                self.split,
                self.template_idx,
                self.max_number_of_samples,
            )
        )

    def __eq__(self, other):
        return hash(
            (
                self.instruction_format,
                self.dataset,
                self.language_code,
                self.language,
                self.domain,
                self.task,
                self.split,
                self.template_idx,
                self.max_number_of_samples,
            )
        ) == hash(
            (
                other.instruction_format,
                other.dataset,
                other.language_code,
                other.language,
                other.domain,
                other.task,
                other.split,
                other.template_idx,
                other.max_number_of_samples,
            )
        )
