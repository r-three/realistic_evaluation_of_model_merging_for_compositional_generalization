import copy
import json
import logging
import os
import random
from typing import Any, Callable, Dict, List

from datasets import load_dataset
from promptsource.templates import DatasetTemplates, Template

from src.data.dataset_reader.DatasetReader import DatasetReader
from src.data.DatasetConfig import DatasetConfig

# We split the train to train/validation and use the validation as the test set
DEFAULT_SPLIT_MAPPING = {"train": "train", "validation": "train", "test": "validation"}
DEFAULT_VALIDATION_SET_SIZE = 1000

NO_SPLIT_MAPPINGS = {"train": "train", "validation": "validation", "test": "test"}

ALL_TRAIN_SPLIT_MAPPINGS = {"train": "train", "validation": "train", "test": "train"}

ONLY_VALIDATION_SPLIT_MAPPINGS = {
    "train": None,
    "validation": "validation",
    "test": "validation",
}

DEFAULT_TEST_SET_SIZE = 2500


def get_datasetTemplates(
    template_stash: (str, str),
    metrics_toUse: List[str],
    templateNames_toIgnore: List[str],
):
    """
    Args:
        template_stash:
        metrics_toUse:
        templateNames_toIgnore:

    Returns:
        _description_
    """
    all_templates = []

    # Get original templates from promptsource
    for template in DatasetTemplates(*template_stash).templates.values():
        # Filter out templates that
        # 1) are not designed for original task
        # 2) have different metrics than we want to use
        # 3) are ones that we want to ignore based on the name
        if template.metadata.original_task:
            should_ignoreTemplate = False

            for metric in template.metadata.metrics:
                if metric not in metrics_toUse:
                    should_ignoreTemplate = True
            for template_name in templateNames_toIgnore:
                if template.name == template_name:
                    should_ignoreTemplate = True

            if not should_ignoreTemplate:
                all_templates.append(template)

    return all_templates


def getData_fromHuggingFace(
    dataset_stash: (str, str),
    split: str,
    real_split: str,
    preprocess_fn: Callable[[int, Dict], Dict],
    filter_fn: Callable[[Dict], bool],
    data_dir: str,
    revision: str,
    offset_trainValidation: int,
    offset_trainTest: int,
    offset_validationTest: int,
) -> List[Dict]:
    """

    Args:
        dataset_stash:
        split: split we want
        real_split: split of the actual data (can differ from split since the test set is the real validation set and train/validation set are the real train set )
        offset_trainValidation: how much of the train set to take to construct the validation set
        offste_trainTest: how much of the train set to take to construct the test set
        offset_validationTest: how much of the validation set to use for test
        preprocess_fn: function that takes in idx, example and preprocesses it correctly
        data_dir: directory of HuggingFace data
        revision: commit number of HuggingFace data to load

    Returns:

    """
    if data_dir is not None:
        huggingFace_data = load_dataset(
            *dataset_stash, split=real_split, data_dir=data_dir
        )
    elif revision is not None:
        huggingFace_data = load_dataset(
            *dataset_stash, split=real_split, revision=revision
        )

    else:
        huggingFace_data = load_dataset(*dataset_stash, split=real_split)

    data = []
    for idx, example in enumerate(huggingFace_data):
        if filter_fn is None or filter_fn(example):
            data.append(preprocess_fn(idx, example))

    # If we use some of the train to construct validation, then adjust the
    # train/validation accordingly
    if offset_trainValidation != 0 and offset_trainTest != 0:
        if split == "train":
            data = data[: -(offset_trainValidation + offset_trainTest)]
        elif split == "validation":
            data = data[
                -(offset_trainValidation + offset_trainTest) : -offset_trainTest
            ]
        elif split == "test":
            data = data[-offset_trainTest:]
    elif offset_trainValidation != 0:
        if split == "train":
            data = data[:-offset_trainValidation]
        elif split == "validation":
            data = data[-offset_trainValidation:]
    elif offset_trainTest != 0:
        if split == "train":
            data = data[:-offset_trainTest]
        elif split == "test":
            data = data[-offset_trainTest:]
    elif offset_validationTest != 0:
        if split == "validation":
            data = data[:-offset_validationTest]
        elif split == "test":
            data = data[-offset_validationTest:]
    return data


def applyTemplate_toData(
    original_data: List[Dict],
    all_templates: List,
    train_or_eval: str,
    template_idx: int,
) -> List[Dict]:
    """
    Args:
        original_data:
        template_idx:
        all_templates:

    Raises:
        ValueError: Invalid template idx

    Returns:

    """
    templated_data = []

    for datapoint_idx, datapoint in enumerate(original_data):
        # Use fixed template across entire dataset
        if template_idx >= 0:
            templateIdx_forDatapoint = template_idx

        # Use all templates across entire dataset, where different datapoints can get
        # different templates. However, a datapoint is always matched with the same template
        elif template_idx == -1:
            templateIdx_forDatapoint = datapoint_idx % len(all_templates)

        else:
            raise ValueError(f"Invalid template idx {templateIdx_forDatapoint}")

        template = all_templates[templateIdx_forDatapoint]

        # Copy datapoint since we might reuse the original data
        new_datapoint = copy.deepcopy(datapoint)

        # Whether to use answer_choices or target
        answer_choices = template.get_answer_choices_list(datapoint)
        if train_or_eval == "eval" and answer_choices is not None:
            new_datapoint["answer_choices"] = answer_choices

        # We apply the template to datapoint instead of new_datapoint since the answer_choices
        # are added in the template function, and so applying the template to new_datapoint
        # will cause an error with the answer_choices key
        input_txt, target_txt = template.apply(datapoint)
        new_datapoint["input"] = input_txt

        # For non-evaluation or tasks where they are no answer_choices, we just add the target (the correct answer_choice)
        if "answer_choices" not in new_datapoint:
            assert isinstance(target_txt, list)
            new_datapoint["target"] = target_txt[0]

        templated_data.append(new_datapoint)

    return templated_data


def DEFAULT_PREPROCESS_FN(idx: int, example: Dict) -> Dict:
    """
    Args:
        idx:
        example:

    Returns:
        example
    """
    example["idx"] = idx
    example["lbl"] = int(example["label"])
    return example


def DEFAULT_FILTER_FN(example: Dict) -> bool:
    return True


class P3_DatasetReader(DatasetReader):
    def __init__(self, dataset_config: DatasetConfig):
        """
        Args:
            dataset_config:
        """
        self.dataset_config = dataset_config

        # Cached the dataset and temaplates
        self.cached_originalData = {}
        self.cached_datasets = {}
        self.templates = None

        """
        Attributes that each dataset must set 
        """
        # Dataset attributes
        self.dataset_stash = None
        self.validationSet_size = None
        self.testSet_size = None
        self.preprocess_fn = None
        self.filter_fn = None
        self.split_mappings = None
        self.data_dir = None
        self.revision = None

        # Template attributes
        self.template_stash = None
        self.templates_toIgnore = None
        self.metrics_toUse = None

    def _get_originalData(self) -> List:
        """
        Args:
            split:

        Returns:

        """
        real_split = self.split_mappings[self.dataset_config.split]

        if self.dataset_config.split not in self.cached_originalData:
            self.cached_originalData[self.dataset_config.split] = (
                getData_fromHuggingFace(
                    self.dataset_stash,
                    self.dataset_config.split,
                    real_split,
                    self.preprocess_fn,
                    self.filter_fn,
                    self.data_dir,
                    self.revision,
                    offset_trainValidation=self.validationSet_size,
                    offset_trainTest=self.testSet_size,
                    offset_validationTest=0,
                )
            )

        return self.cached_originalData[self.dataset_config.split]

    def _get_datasetTemplates(self) -> List:
        """
        Returns:
            templates
        """
        if self.templates is None:
            self.templates = get_datasetTemplates(
                self.template_stash, self.metrics_toUse, self.templates_toIgnore
            )
        return self.templates

    def get_dataset(self, train_or_eval):
        """
        Create dataset that includes the template

        Args:
            train_or_eval: train_or_eval for determing how to format the prompts (not the actual split). For example, to evaluate on train, we would pass eval since we are in eval mode.
        Returns:
            dataset:
        """
        if (
            self.dataset_config.split,
            self.dataset_config.template_idx,
            self.dataset_config.max_number_of_samples,
        ) not in self.cached_datasets:
            # Get original data
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
                f"Loaded {self.dataset_config.split} which contains {len(original_data)} original datapoints"
            )

            all_templates = self._get_datasetTemplates()
            num_templates = len(all_templates)

            # template_idx -2 means we do a cross product of each datapoint with each template
            if self.dataset_config.template_idx == -2:
                dataset = []
                # Get data for all template
                templateIdx_toData = {}
                for iterate_templateIdx in range(num_templates):
                    templateIdx_toData[iterate_templateIdx] = applyTemplate_toData(
                        original_data, all_templates, train_or_eval, iterate_templateIdx
                    )
                # Shuffle data across templates
                for datapoint_idx in range(len(templateIdx_toData[0])):
                    for template_idx in range(len(templateIdx_toData)):
                        dataset.append(templateIdx_toData[template_idx][datapoint_idx])

            # otherwise apply template to dataset
            else:
                dataset = applyTemplate_toData(
                    original_data,
                    all_templates,
                    train_or_eval,
                    self.dataset_config.template_idx,
                )

            logging.info(
                f"Loaded {self.dataset_config.split} which contains {len(dataset)} datapoints with templates"
            )

            self.cached_datasets[
                (
                    self.dataset_config.split,
                    self.dataset_config.template_idx,
                    self.dataset_config.max_number_of_samples,
                )
            ] = dataset

        return self.cached_datasets[
            self.dataset_config.split,
            self.dataset_config.template_idx,
            self.dataset_config.max_number_of_samples,
        ]

    def get_datasetMetrics(self):
        all_templates = self._get_datasetTemplates()
        return all_templates[0].metadata.metrics


class P3_RTEReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("super_glue", "rte")
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = None

        # Template attributes
        self.template_stash = ("super_glue", "rte")
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_MNLIReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("glue", "mnli")
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = "fd8e86499fa5c264fcaad392a8f49ddf58bf4037"

        # Template attributes
        self.template_stash = ("glue", "mnli")
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_QNLIReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("glue", "qnli")
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = None
        self.revision = "fd8e86499fa5c264fcaad392a8f49ddf58bf4037"

        # Template attributes
        self.template_stash = ("glue", "qnli")
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_QQPReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("glue", "qqp")
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = "fd8e86499fa5c264fcaad392a8f49ddf58bf4037"

        # Template attributes
        self.template_stash = ("glue", "qqp")
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_HSwagReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("hellaswag",)
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = None

        # Template attributes
        self.template_stash = ("hellaswag",)
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]

    def _get_datasetTemplates(self) -> List:
        """
        Override default dataset templates

        Returns:
            dataset_templates
        """
        all_templates = get_datasetTemplates(
            self.template_stash,
            ["Accuracy"],
            ["Randomized prompts template"],
        )

        # Add each template from the several templates in the randomized prompt individually
        random_jinjas = [
            (
                "randomized prompt 1",
                "Can you pick the correct ending for the sentence: {{ctx}}|||{{answer_choices [label | int()]}}",
            ),
            (
                "randomized prompt 2",
                "The task is to generate the ending for the sentence: {{ctx}}|||{{answer_choices [label | int()]}}",
            ),
            (
                "randomized prompt 3",
                "How does this sentence end? {{ctx}}|||{{answer_choices [label | int()]}}",
            ),
            (
                "randomized prompt 4",
                "From the list of endings described below, what ending makes the most sense for the sentence {{ctx}}|||{{answer_choices [label | int()]}}",
            ),
        ]

        for name, jinja in random_jinjas:
            all_templates.append(
                Template(
                    name=name,
                    jinja=jinja,
                    reference="",
                    answer_choices='{{endings | join("|||")}}',
                )
            )

        return all_templates


class P3_WiCReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("super_glue", "wic")
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = None

        # Template attributes
        self.template_stash = ("super_glue", "wic")
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_WinograndeReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("winogrande", "winogrande_xl")
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = self._preprocess_fn
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = None

        # Template attributes
        self.template_stash = ("winogrande", "winogrande_xl")
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]

    def _preprocess_fn(self, idx: int, example: Dict) -> Dict:
        """
        Args:
            idx:
            example:

        Returns:
            example
        """
        example["idx"] = idx
        example["lbl"] = int(example["answer"]) - 1
        return example


class P3_CBReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("super_glue", "cb")
        self.validationSet_size = 100
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = None

        # Template attributes
        self.template_stash = ("super_glue", "cb")
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_BoolQReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("super_glue", "boolq")
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = None

        # Template attributes
        self.template_stash = ("super_glue", "boolq")
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_COPAReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("super_glue", "copa")
        self.validationSet_size = 100
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = None

        # Template attributes
        self.template_stash = ("super_glue", "copa")
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]

    def _get_datasetTemplates(self) -> List:
        """
        Returns:
            dataset_templates
        """
        return get_datasetTemplates(
            self.template_stash,
            ["Accuracy"],
            [
                "…which may be caused by",
                "…What could happen next, C1 or C2?",
                "…As a result, C1 or C2?",
                "…why? C1 or C2",
            ],
        )


class P3_MultiRCReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("super_glue", "multirc")
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.data_dir = None
        self.revision = None

        # Template attributes
        self.template_stash = ("super_glue", "multirc")
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_ReCORDReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("super_glue", "record")
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = self._preprocess_fn
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = None

        # Template attributes
        self.template_stash = ("super_glue", "record")
        self.templates_toIgnore = []
        self.metrics_toUse = ["Squad"]

    def _preprocess_fn(self, idx: int, example: Dict) -> Dict:
        """
        Args:
            idx:
            example:

        Returns:
            example
        """
        example["idx"] = idx
        example["id"] = str(idx)
        example["text"] = example["answers"]
        example["answer_start"] = [0] * len(example["answers"])
        return example


class P3_WiCReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("super_glue", "wic")
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = None

        # Template attributes
        self.template_stash = ("super_glue", "wic")
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_WSCReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("super_glue", "wsc.fixed")
        self.validationSet_size = 100
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = None

        # Template attributes
        self.template_stash = ("super_glue", "wsc.fixed")
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_CoLAReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("glue", "cola")
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = "fd8e86499fa5c264fcaad392a8f49ddf58bf4037"

        # Template attributes
        self.template_stash = ("glue", "cola")
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_STSBReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("glue", "stsb")
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = "fd8e86499fa5c264fcaad392a8f49ddf58bf4037"

        # Template attributes
        self.template_stash = ("glue", "stsb")
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_MRPCReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("glue", "mrpc")
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = "fd8e86499fa5c264fcaad392a8f49ddf58bf4037"

        # Template attributes
        self.template_stash = ("glue", "mrpc")
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_SST2Reader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("glue", "sst2")
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = "fd8e86499fa5c264fcaad392a8f49ddf58bf4037"

        # Template attributes
        self.template_stash = ("glue", "sst2")
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_WNLIReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("glue", "wnli")
        self.validationSet_size = 200
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = "fd8e86499fa5c264fcaad392a8f49ddf58bf4037"

        # Template attributes
        self.template_stash = ("glue", "wnli")
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_StoryClozeReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("story_cloze", "2016")
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = self._preprocess_fn
        self.split_mappings = {
            "train": "validation",
            "validation": "validation",
            "test": "test",
        }
        self.data_dir = os.path.join(
            os.environ["HUGGINGFACE_HUB_CACHE"], "datasets", "story_cloze"
        )
        self.revision = None

        # Template attributes
        self.template_stash = ("story_cloze", "2016")
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]

    def _preprocess_fn(self, idx: int, example: Dict) -> Dict:
        """
        Args:
            idx:
            example:

        Returns:
            example
        """
        example["idx"] = idx
        example["lbl"] = int(example["answer_right_ending"]) - 1
        return example


class P3_ANLIR1Reader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("anli",)
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        # Hack so that the real_split includes the round
        self.split_mappings = {
            "train": "train_r1",
            "validation": "train_r1",
            "test": "dev_r1",
        }
        self.revision = "bf206833154d4fcaf5e3b01b8bf17d4d15213cb1"

        # Template attributes
        self.template_stash = ("anli",)
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_ANLIR2Reader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("anli",)
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        # Hack so that the real_split includes the round
        self.split_mappings = {
            "train": "train_r2",
            "validation": "train_r2",
            "test": "dev_r2",
        }
        self.data_dir = None
        self.revision = "bf206833154d4fcaf5e3b01b8bf17d4d15213cb1"

        # Template attributes
        self.template_stash = ("anli",)
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_ANLIR3Reader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("anli",)
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        # Hack so that the real_split includes the round
        self.split_mappings = {
            "train": "train_r3",
            "validation": "train_r3",
            "test": "dev_r3",
        }
        self.data_dir = None
        self.revision = "bf206833154d4fcaf5e3b01b8bf17d4d15213cb1"

        # Template attributes
        self.template_stash = ("anli",)
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_CosmosQAReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("cosmos_qa",)
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = None

        # Template attributes
        self.template_stash = ("cosmos_qa",)
        self.templates_toIgnore = ["xp3longcontext"]
        self.metrics_toUse = ["Accuracy"]


class P3_SocialIQAReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("social_i_qa",)
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = self._preprocess_fn
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = None

        # Template attributes
        self.template_stash = ("social_i_qa",)
        self.templates_toIgnore = ["Check if a random answer is valid or not"]
        self.metrics_toUse = ["Accuracy"]

    def _preprocess_fn(self, idx: int, example: Dict) -> Dict:
        """
        Args:
            idx:
            example:

        Returns:
            example
        """
        example["idx"] = idx
        example["lbl"] = int(example["label"]) - 1
        return example


class P3_PAWSReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("paws", "labeled_final")
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = "cd6b868f3d2d71e9708ed861deee4bbc4d32441e"

        # Template attributes
        self.template_stash = ("paws", "labeled_final")
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_QuAILReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("quail",)
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = self._preprocess_fn
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = "4e1ae2f7abec085a44643e33888ca520c5d2b304"

        # Template attributes
        self.template_stash = ("quail",)
        self.templates_toIgnore = ["xp3longtokenpassage", "xp3longstory"]
        self.metrics_toUse = ["Accuracy"]

    def _preprocess_fn(self, idx: int, example: Dict) -> Dict:
        """
        Args:
            idx:
            example:

        Returns:
            example
        """
        example["idx"] = idx
        example["lbl"] = example["correct_answer_id"]
        return example


class P3_WikiQAReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("wiki_qa",)
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = "4cdcefa3617bf52a562b1c423fd992859b031ee4"

        # Template attributes
        self.template_stash = ("wiki_qa",)
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_QuaRTzReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("quartz",)
        self.validationSet_size = 200
        self.testSet_size = 0
        self.preprocess_fn = self._preprocess_fn
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = "d09d2e1e12c2c44c001da9cfa83ca6192a51f71d"

        # Template attributes
        self.template_stash = ("quartz",)
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]

        # Attributes specific for QuarTZ
        self.string_toLabelIdx = {"A": 0, "B": 1}

    def _preprocess_fn(self, idx: int, example: Dict) -> Dict:
        """
        Args:
            idx:
            example:

        Returns:
            example
        """
        example["idx"] = idx
        example["lbl"] = self.string_toLabelIdx[example["answerKey"]]
        return example


class P3_QASCReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("qasc",)
        self.validationSet_size = 500
        self.testSet_size = 0
        self.preprocess_fn = self._preprocess_fn
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = "3dfa861b842adcec206a3a7b56e8c4a03fbf2e22"

        # Template attributes
        self.template_stash = ("qasc",)
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]

        # Attributes specific for QASC
        self.string_toLabelIdx = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
        }

    def _preprocess_fn(self, idx: int, example: Dict) -> Dict:
        """
        Args:
            idx:
            example:

        Returns:
            example
        """
        example["idx"] = idx
        example["lbl"] = self.string_toLabelIdx[example["answerKey"]]
        return example


class P3_ROPESReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("ropes",)
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = self._preprocess_fn
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = "f952e07f6fd78d17499644cbe698bae223560284"

        # Template attributes
        self.template_stash = ("ropes",)
        self.templates_toIgnore = ["xp3longneedbackground", "xp3longwhatsituation"]
        self.metrics_toUse = ["Squad"]

    def _preprocess_fn(self, idx: int, example: Dict) -> Dict:
        """
        Args:
            idx:
            example:

        Returns:
            example
        """
        example["idx"] = idx
        example["answers"]["answer_start"] = [0]
        return example


class P3_XNLIReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("xnli", self.dataset_config.language_code)
        # validation_set size has 0 examples since there are train/validation/test splits
        # so no validation_set has to be taken from the training set
        self.validationSet_size = 0
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = NO_SPLIT_MAPPINGS
        self.data_dir = None
        self.revision = "1cdcf07be24d81f3d782038a5a0b9c8d62f76e60"

        # Template attributes
        self.template_stash = ("xnli", self.dataset_config.language_code)
        self.templates_toIgnore = ["xp3longneedbackground", "xp3longwhatsituation"]
        self.metrics_toUse = ["Accuracy"]

    def _preprocess_fn(self, idx: int, example: Dict) -> Dict:
        """
        Args:
            idx:
            example:

        Returns:
            example
        """
        example["idx"] = idx
        return example


class P3_PAWSXReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("paws-x", self.dataset_config.language_code)
        # validation_set size has 0 examples since there are train/validation/test splits
        # so no validation_set has to be taken from the training set
        self.validationSet_size = 0
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = NO_SPLIT_MAPPINGS
        self.data_dir = None
        self.revision = "8a04d940a42cd40658986fdd8e3da561533a3646"

        # Template attributes
        self.template_stash = ("paws-x", self.dataset_config.language_code)
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_XLWiCReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)

        # There is no korean for training, only inference. Thus, we load the
        # english for training and korean for inference, but assert that this
        # is during inference only
        if self.dataset_config.language_code == "ko":
            trainingLanguage_code = "en"
            assert self.dataset_config.split != "train"
        else:
            trainingLanguage_code = self.dataset_config.language_code

        # Dataset attributes
        self.dataset_stash = (
            "pasinit/xlwic",
            f"xlwic_{trainingLanguage_code}_{self.dataset_config.language_code}",
        )
        # validation_set size has 0 examples since there are train/validation/test splits
        # so no validation_set has to be taken from the training set
        self.validationSet_size = 0
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = NO_SPLIT_MAPPINGS
        self.data_dir = None
        self.revision = None

        # Template attributes
        self.template_stash = (
            "pasinit/xlwic",
            f"xlwic_{self.dataset_config.language_code}_{self.dataset_config.language_code}",
        )
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]


class P3_XLSumReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = (
            "csebuetnlp/xlsum",
            f"xlwic_{self.dataset_config.language}",
        )
        # validation_set size has 0 examples since there are train/validation/test splits
        # so no validation_set has to be taken from the training set
        self.validationSet_size = 0
        self.testSet_size = 0
        self.preprocess_fn = DEFAULT_PREPROCESS_FN
        self.filter_fn = DEFAULT_FILTER_FN
        self.split_mappings = NO_SPLIT_MAPPINGS
        self.data_dir = None
        self.revision = None

        # Template attributes
        self.template_stash = (
            "csebuetnlp/xlsum",
            f"xlwic_{self.dataset_config.language}",
        )
        self.templates_toIgnore = []
        self.metrics_toUse = ["sp_rouge"]


class P3_WikiLinguaReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = (
            "wiki_lingua",
            f"{self.dataset_config.language}",
        )
        # validation_set size has 0 examples since there are train/validation/test splits
        # so no validation_set has to be taken from the training set
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = DEFAULT_TEST_SET_SIZE
        self.filter_fn = self._filter_fn
        self.preprocess_fn = self._preprocess_fn
        self.split_mappings = ALL_TRAIN_SPLIT_MAPPINGS
        self.data_dir = None
        self.revision = "700647c975386e82e711d45ee801d9385af000b1"

        # Template attributes
        self.template_stash = (
            "wiki_lingua",
            f"{self.dataset_config.language}",
        )
        self.templates_toIgnore = ["xp3longsummarizedas", "xp3longfittingnews"]
        self.metrics_toUse = ["sp_rouge"]

    def _preprocess_fn(self, idx: int, example: Dict) -> Dict:
        """
        Args:
            idx:
            example:

        Returns:
            example
        """
        example["idx"] = idx
        example["document"] = example["article"]["document"][0]
        example["summary"] = example["article"]["summary"][0]
        return example

    def _filter_fn(self, example):
        if len(example["article"]["document"]) > 0:
            return True
        else:
            return False


class P3_TyDiQAReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = (
            "tydiqa",
            f"primary_task",
        )
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.preprocess_fn = self._preprocess_fn
        self.filter_fn = lambda example: self._filter_fn(
            dataset_config.language, example
        )
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = None

        # Template attributes
        self.template_stash = (
            "tydiqa",
            f"primary_task_{dataset_config.language}",
        )
        self.templates_toIgnore = []
        self.metrics_toUse = ["Accuracy"]

        self.string_toLabelIdx = {"YES": 0, "NO": 1, "NONE": 2}

    def _preprocess_fn(self, idx: int, example: Dict) -> Dict:
        """
        Args:
            idx:
            example:

        Returns:
            example
        """
        example["idx"] = idx
        example["lbl"] = self.string_toLabelIdx[
            example["annotations"]["yes_no_answer"][0]
        ]
        return example

    def _filter_fn(self, language_code, example):
        if example["language"] == language_code:
            return True
        else:
            return False


class P3_SQuADReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("squad",)
        self.validationSet_size = DEFAULT_VALIDATION_SET_SIZE
        self.testSet_size = 0
        self.filter_fn = DEFAULT_FILTER_FN
        self.preprocess_fn = self._preprocess_fn
        self.split_mappings = DEFAULT_SPLIT_MAPPING
        self.data_dir = None
        self.revision = "5fe18c4c680f9922d794e3f4dd673a751c74ee37"

        # Template attributes
        self.template_stash = ("squad",)
        self.templates_toIgnore = []
        self.metrics_toUse = ["Squad"]

    def _preprocess_fn(self, idx: int, example: Dict) -> Dict:
        """
        Args:
            idx:
            example:
        Returns:
            example
        """
        example["idx"] = idx
        return example


class P3_XQuADReader(P3_DatasetReader):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        # Dataset attributes
        self.dataset_stash = ("xquad", f"xquad.{dataset_config.language_code}")
        self.validationSet_size = 200
        self.testSet_size = 0
        self.filter_fn = DEFAULT_FILTER_FN
        self.preprocess_fn = self._preprocess_fn
        self.split_mappings = ONLY_VALIDATION_SPLIT_MAPPINGS
        self.data_dir = None
        self.revision = "8c2924a720ea543c2b6346284e21d3b85b1c2996"

        # Template attributes
        self.template_stash = ("xquad", f"xquad.{dataset_config.language_code}")
        self.templates_toIgnore = ["xp3longchar", "xp3longcontext"]
        self.metrics_toUse = ["Squad"]

    def _get_originalData(self) -> List:
        """
        Args:
            split:

        Returns:

        """
        real_split = self.split_mappings[self.dataset_config.split]

        if self.dataset_config.split not in self.cached_originalData:
            self.cached_originalData[self.dataset_config.split] = (
                getData_fromHuggingFace(
                    self.dataset_stash,
                    self.dataset_config.split,
                    real_split,
                    self.preprocess_fn,
                    self.filter_fn,
                    self.data_dir,
                    self.revision,
                    offset_trainValidation=0,
                    offset_trainTest=self.testSet_size,
                    offset_validationTest=self.validationSet_size,
                )
            )

        return self.cached_originalData[self.dataset_config.split]

    def _preprocess_fn(self, idx: int, example: Dict) -> Dict:
        """
        Args:
            idx:
            example:
        Returns:
            example
        """

        example["idx"] = idx
        return example


if __name__ == "__main__":
    import ipdb

    ipdb.set_trace()
    data = load_dataset(*("wiki_lingua", "italian"), split="train")
