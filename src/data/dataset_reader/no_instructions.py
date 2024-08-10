import json
import logging
import os
from typing import Any, Callable, Dict, List

from src.data.dataset_reader.DatasetReader import DatasetReader
from src.data.DatasetConfig import DatasetConfig

kvarithmetic_directory = os.path.join("datasets", "new_key_value_arithmetic")


class TextFileDatasetReader(DatasetReader):
    """
    DatasetReader that constructs dataset from reading a local text file
    """

    def __init__(self, dataset_config: DatasetConfig):
        """
        Args:
            dataset_config:
        """
        self.dataset_config = dataset_config

        # Cached the dataset and temaplates
        self.cached_datasets = {}

        # Dataset attributes
        self.dataset_fp = None

        self._preprocess_fn = None

    def _get_originalData(self) -> List[Any]:
        """
        Get the data in its original format

        Returns:
            load_data:
        """
        assert self.dataset_config.template_idx is None, (
            self.dataset_config.dataset_name
            + "has no instructions but template "
            + self.dataset_config.template_idx
            + " was specified"
        )

        assert self.dataset_config.instruction_format == "no_instructions", (
            self.dataset_config.instruction_format
            + "should be none with no instructions"
        )

        load_data = []
        split_fp = os.path.join(self.dataset_fp, f"{self.dataset_config.split}.json")
        with open(split_fp, "r") as f:
            for idx, line in enumerate(f.readlines()):
                datapoint = json.loads(line)
                load_data.append(self._preprocess_fn(idx, datapoint))
        return load_data

    def get_dataset(self, train_or_eval) -> List[Any]:
        """
        Returns:
            dataset
        """
        if (
            self.dataset_config.split,
            self.dataset_config.max_number_of_samples,
        ) not in self.cached_datasets:
            original_data = self._get_originalData()

            # Trim the original data smaller
            if self.dataset_config.max_number_of_samples != -1:
                assert (
                    self.dataset_config.template_idx != -2
                ), f"Cannot handle max number of samples if doing a cross product of samples and templates "
                original_data = original_data[
                    : self.dataset_config.max_number_of_samples
                ]

            logging.info(
                f"Loaded {self.dataset_config.split} which contains {len(original_data)} datapoints"
            )
            self.cached_datasets[
                (self.dataset_config.split, self.dataset_config.max_number_of_samples)
            ] = original_data
        return self.cached_datasets[
            (self.dataset_config.split, self.dataset_config.max_number_of_samples)
        ]


def KV_ARITHMETIC_PREPROCESS_FN(
    idx: int, datapoint: Dict, input_prompt: str, additionalInput_prompt: str
) -> Dict:
    """
    Args:
        idx:
        example:
        input_prompt

    Returns:
        example
    """
    example = {}
    example["idx"] = idx
    example["id"] = str(idx)
    example["input"] = input_prompt + str(datapoint["input"])
    if additionalInput_prompt is not None:
        example["additional_input"] = additionalInput_prompt
    example["target"] = str(datapoint["target"])
    example["answers"] = {
        "text": [str(datapoint["target"])],
        "answer_start": [0],
    }
    return example


class KVArithmeticReader(TextFileDatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_fp = os.path.join(kvarithmetic_directory, "kv_arithmetic")
        self.input_prompt = (
            "Replace the key with its correct value and solve the following equation: "
        )
        self.additionalInput_prompt = ""
        # self.additionalInput_prompt = " Solve the following equation: "
        self._preprocess_fn = lambda idx, datapoint: KV_ARITHMETIC_PREPROCESS_FN(
            idx, datapoint, self.input_prompt, self.additionalInput_prompt
        )

    def get_datasetMetrics(self):
        return ["arithmetic"]


class KVArithmeticCoTReader(TextFileDatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_fp = os.path.join(kvarithmetic_directory, "kv_arithmetic_cot")
        self.input_prompt = (
            "Replace the key with its correct value and solve the following equation"
        )
        self.additionalInput_prompt = None
        # self.input_prompt = "Replace the key with its correct value: "
        # self.additionalInput_prompt = " Solve the following equation: "
        self._preprocess_fn = lambda idx, datapoint: KV_ARITHMETIC_PREPROCESS_FN(
            idx, datapoint, self.input_prompt, self.additionalInput_prompt
        )

    def get_datasetMetrics(self):
        return ["arithmetic"]


class NumericArithmeticReader(TextFileDatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_fp = os.path.join(kvarithmetic_directory, "numerical_arithmetic")
        self.input_prompt = "Solve the following equation: "
        self.additionalInput_prompt = None
        self._preprocess_fn = lambda idx, datapoint: KV_ARITHMETIC_PREPROCESS_FN(
            idx, datapoint, self.input_prompt, self.additionalInput_prompt
        )

    def get_datasetMetrics(self):
        return ["arithmetic"]


class KVSubstitutionReader(TextFileDatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_fp = os.path.join(kvarithmetic_directory, "kv_substitution")
        self.input_prompt = "Replace the key with its correct value: "
        self.additionalInput_prompt = None
        self._preprocess_fn = lambda idx, datapoint: KV_ARITHMETIC_PREPROCESS_FN(
            idx, datapoint, self.input_prompt, self.additionalInput_prompt
        )

    def get_datasetMetrics(self):
        return ["kv_substitution"]
