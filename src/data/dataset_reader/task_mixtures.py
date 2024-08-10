import random
import math
from typing import Any, Dict, List, Callable

from src.data.dataset_reader.domainnet import *
from src.utils.config_utils import *
from src.data.dataset_reader.DatasetReader import DatasetReader
from src.data.dataset_reader.dataset_reader import get_singleDatasetReader, DATASETS
from src.data.DatasetConfig import DatasetConfig
from itertools import combinations

domainnet_taskMixtures = {}

domainnet_taskNames = []
for task_idx, task, domain in getPairs_taskWithDomain():
    domainnet_taskNames.append(("domainnet", domain, task))
domainnet_taskMixtures[f"domainnet"] = {
    "dataset_names": domainnet_taskNames,
    "dataset_ratios": [1] * len(domainnet_taskNames),
}

domainnet_crossProductTaskNames = []
for task_idx, task, domain in getCrossProductPairs_taskWithDomain():
    domainnet_crossProductTaskNames.append(("domainnet", domain, task))
domainnet_taskMixtures[f"cp_domainnet"] = {
    "dataset_names": domainnet_crossProductTaskNames,
    "dataset_ratios": [1] * len(domainnet_crossProductTaskNames),
}

domainnet_generalizationTasknames = []
for task_idx, task, domain in getMissingPairs_taskWithDomain():
    domainnet_generalizationTasknames.append(("domainnet", domain, task))
domainnet_taskMixtures[f"generalization_domainnet"] = {
    "dataset_names": domainnet_generalizationTasknames,
    "dataset_ratios": [1] * len(domainnet_generalizationTasknames),
}


TASK_MIXTURES = {
    "multitask_multilingual": {
        "dataset_names": [
            "squad",
            ("xnli", "ar"),
            ("wiki_lingua", "thai"),
            ("xlwic", "de"),
            ("tydiqa", "korean"),
        ],
        "dataset_ratios": [1, 1, 1, 1, 1],
    },
    "generalization_multitask_multilingual": {
        "dataset_names": [
            ("xquad", "ar"),
            ("xquad", "de"),
            ("xquad", "th"),
            ("xnli", "th"),
            ("xnli", "en"),
            ("xnli", "de"),
            ("wiki_lingua", "arabic"),
            ("wiki_lingua", "english"),
            ("wiki_lingua", "korean"),
            ("wiki_lingua", "german"),
            "wic",
            ("xlwic", "ko"),
            ("tydiqa", "english"),
            ("tydiqa", "thai"),
            ("tydiqa", "arabic"),
        ],
        "dataset_ratios": [1] * 17,
    },
}


TASK_MIXTURES.update(domainnet_taskMixtures)


def get_allDatasets():
    """
    Get all datasets using the datasets which are mapped to a reader
    """
    all_datasets = set()
    for _, dataset_andReader in DATASETS.items():
        all_datasets.update(set(dataset_andReader.keys()))
    return list(all_datasets)


def checkAllPreviousSubsets_differ(sampled_permuation, permutations):

    for permutation in permutations:
        assert len(sampled_permuation) == len(permutation)

        for idx in range(2, len(permutation)):
            if sampled_permuation[:idx] == permutation[:idx]:
                return False

    return True


# Returns permutations which differ and which every possible subset also differ
def generate_permutations(list_size, number_of_permutations):
    random.seed(0)

    indices = list(range(list_size))

    permutations = []
    while len(permutations) < number_of_permutations:

        indices_copy = indices.copy()
        random.shuffle(indices_copy)

        if checkAllPreviousSubsets_differ(indices_copy, permutations):
            permutations.append(indices_copy)
    return permutations


def sample_subset(
    list_toSampleFrom: list,
    numberOfElements_toSample: int,
    subset_id: int,
):
    """

    Args:
        list_toSampleFrom:
        numberOfElements_toSample:
    """
    if len(list_toSampleFrom) > 20:
        number_ofPermutations = 20
    else:
        number_ofPermutations = 10
    permutation = generate_permutations(len(list_toSampleFrom), number_ofPermutations)[
        subset_id
    ]

    sampled_elements = [list_toSampleFrom[i] for i in permutation][
        :numberOfElements_toSample
    ]

    return sampled_elements


def getTasks_inMixture(task_mixture: str | List) -> List[str]:
    """
    Args:
        task_mixture:

    Returns:
        task_mixture
    """
    all_datasets = get_allDatasets()
    # If task_mixture is a list, then check that each dataset in list is valid.
    if isinstance(task_mixture, list):
        for dataset in task_mixture:
            assert dataset in all_datasets
            actual_taskMixture = task_mixture

        return actual_taskMixture
    # If task_mixture is a string, then we look up the dataset mixture.
    elif task_mixture in TASK_MIXTURES.keys():

        actual_taskMixture = TASK_MIXTURES[task_mixture]["dataset_names"]

        return actual_taskMixture
    # task_mixture might be just one dataset
    else:
        assert task_mixture in all_datasets
        return [task_mixture]


def getTaskRatios_inMixture(task_mixture: str | List) -> List[int]:
    """
    Args:
        task_mixture:
        use_firstNTasks:
        seed_forShufflingTasks:

    Returns:
        task_mixture_ratios
    """
    all_datasets = get_allDatasets()
    # If task_mixture is a list, then check that each dataset in list is valid.
    if isinstance(task_mixture, list):
        for dataset in task_mixture:
            assert dataset in all_datasets

        actual_taskMixture = task_mixture
        return [1] * len(actual_taskMixture)

    # If task_mixture is a string, then we look up the dataset mixture.
    elif task_mixture in TASK_MIXTURES.keys():

        actual_taskRatios = TASK_MIXTURES[task_mixture]["dataset_ratios"]

        return actual_taskRatios

    # task_mixture might be just one dataset
    else:
        assert task_mixture in all_datasets
        return [1.0]


class TaskMixtureReader(object):
    """
    A task consists of a dataset and possible other attributes.
    For example, in cross lingual generalization, XNLI is a dataset and the language is the attribute that forms the task

    Args:
        object:
    """

    def __init__(
        self,
        task_mixture: str,
        mixture_subsetSize: int | None,
        mixtureSubset_id: int | None,
        dataset_config: DatasetConfig,
        cached_singleDatasetReaders: Dict[str, DatasetReader],
    ):
        """

        Args:
            task_mixture:
            mixture_subset_size:
            mixtureSubset_id:
            dataset_config:
            cached_singleDatasetReaders:
        """
        self.task_mixture = task_mixture
        self.mixture_subsetSize = mixture_subsetSize
        self.mixtureSubset_id = mixtureSubset_id
        self.dataset_config = dataset_config

        assert (
            self.dataset_config.dataset is None
        ), f"Dataset Config should not have dataset, but has dataset {self.dataset_config.dataset}"

        self.cached_singleDatasetReaders = cached_singleDatasetReaders

        # Store the individual dataset readers so we don't have to rely on the cache to store it,
        # even if the cache also has the dataset readers
        self.mixture_ofDatasetReaders = []

        # Add dataset_readers to the cache of single dataset readers
        for dataset in getTasks_inMixture(
            self.task_mixture, self.mixture_subsetSize, self.mixtureSubset_id
        ):
            newDataset_config = update_datasetConfig(
                self.dataset_config, getDatasetUpdateDict_fromTask(dataset)
            )
            dataset_reader, self.cached_singleDatasetReaders = get_singleDatasetReader(
                newDataset_config, self.cached_singleDatasetReaders
            )
            self.mixture_ofDatasetReaders.append(dataset_reader)

    def get_cacheSingleDatasetReaders(self) -> Dict[str, DatasetReader]:
        """

        Returns:
            cached_singleDatasetReaders
        """
        return self.cached_singleDatasetReaders

    def get_dataset(self, train_or_eval) -> List:
        """
        Get the dataset for the mixture by combining all the individual datasets

        Args:
            train_or_eval:

        Returns:
            datset
        """
        dataset = []

        for task_name, task_ratio in zip(
            getTasks_inMixture(
                self.task_mixture, self.mixture_subsetSize, self.mixtureSubset_id
            ),
            getTaskRatios_inMixture(
                self.task_mixture, self.mixture_subsetSize, self.mixtureSubset_id
            ),
        ):
            dataset_updateDict = getDatasetUpdateDict_fromTask(task_name)
            newDataset_config = update_datasetConfig(
                self.dataset_config, dataset_updateDict
            )
            # DatasetMixtureReaders always uses train mode since it will always add targets only, never the answer_choices since different datasets can have different number of answer_choices, causing issues when batching
            single_dataset = self.cached_singleDatasetReaders[
                newDataset_config
            ].get_dataset("train")
            dataset.extend(single_dataset * task_ratio)
        return dataset

    def get_datasetMetrics(self):
        raise ValueError("Cannot get metrics for mixture of datasets")


def get_taskMixtureReader(
    task_mixture: str,
    mixtureSubset_size: int | None,
    mixtureSubset_id: int | None,
    dataset_config: DatasetConfig,
    cached_singleDatasetReaders: Dict[str, DatasetReader],
) -> (TaskMixtureReader, Dict[str, DatasetReader]):
    """

    Args:
        task_mixture:
        use_firstNTasks:
        seed_forShufflingTasks:
        dataset_config:
        cached_singleDatasetReaders:

    Returns:
        datasetMixture_reader, cached_singleDatasetReader
    """

    taskMixture_reader = TaskMixtureReader(
        task_mixture,
        mixtureSubset_size,
        mixtureSubset_id,
        dataset_config,
        cached_singleDatasetReaders,
    )
    return taskMixture_reader, taskMixture_reader.get_cacheSingleDatasetReaders()


def getDatasetUpdateDict_fromTask(task) -> Dict:
    """
    Get update dict for a datast config that corresponds to a task.
    This is because for task_mixtures, the language can be combined with the dataset name to form the dataset, and this has to be parsed when updating the dataset config.

    Args:
        dataset:

    Returns:

    """

    # Normal dataset
    if isinstance(task, str):
        dataset_updateDict = {"dataset": task}
    else:
        # Add domain
        if task[0] == "domainnet":
            dataset_updateDict = {
                "dataset": task[0],
                "domain": task[1],
                "task": task[2],
            }
        else:
            if task[0] == "xnli" or task[0] == "xlwic" or task[0] == "xquad":
                dataset_updateDict = {
                    "dataset": task[0],
                    "language_code": task[1],
                }
            else:
                assert task[0] == "wiki_lingua" or task[0] == "tydiqa"
                dataset_updateDict = {
                    "dataset": task[0],
                    "language": task[1],
                }

    return dataset_updateDict


def getFormattedTask_fromTask(task) -> str:
    """

    Args:
        task:

    Returns:
        formmated_task:
    """
    # If task is not string, join the string to form task
    if not isinstance(task, str):
        formmated_task = "-".join(list(map(lambda x: str(x), task)))
    else:
        formmated_task = task

    return formmated_task


def getFormattedTask_fromEvaluationConfigWithSingleTask(
    evaluation_config: EvaluationConfig,
) -> str:
    """

    Args:
        evaluation_config:

    Returns:
        formmated_task:
    """
    # Account for cross-lingual dataset where we add the language code to the task to construct dataset name to look up checkpoint
    formatted_tasks = evaluation_config.get_datasetConfig().dataset
    language_code = evaluation_config.get_datasetConfig().language_code
    if language_code is not None:
        formatted_tasks = formatted_tasks + "-" + language_code
    # Add language
    language = evaluation_config.get_datasetConfig().language
    if language is not None:
        formatted_tasks = formatted_tasks + "-" + language
    # Account for different domains dataset where we add the domain to the task to construct dataset name to look up checkpoint
    domain = evaluation_config.get_datasetConfig().domain
    if domain is not None:
        formatted_tasks = formatted_tasks + "-" + domain + "-" + str(formatted_tasks)
    return formatted_tasks


def getFormattedTask_fromEvaluationConfig(evaluation_config) -> List[str] | str:
    """

    Args:
        evaluation_config:

    Returns:

    """

    # If task mixture is specified, we can easily extra tasks from the mixture
    if evaluation_config.task_mixture is not None:
        tasks = getTasks_inMixture(evaluation_config.task_mixture)
        formatted_tasks = []
        for task in tasks:
            formatted_tasks.append(getFormattedTask_fromTask(task))
        return formatted_tasks
    else:
        return getFormattedTask_fromEvaluationConfigWithSingleTask(evaluation_config)
