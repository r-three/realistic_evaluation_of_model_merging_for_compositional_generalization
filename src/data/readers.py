import json
import logging
import os

from src.data.dataset_reader.task_mixtures import *
from src.data.dataset_reader.DatasetReader import DatasetReader
from src.data.dataset_reader.dataset_reader import *
from src.data.dataset_reader.no_instructions import *
from src.data.dataset_reader.prompt_source import *
from src.data.DatasetConfig import DatasetConfig


def get_datasetReader(
    task_mixture: str,
    mixtureSubset_size: str | None,
    mixtureSubset_id: str | None,
    dataset_config: DatasetConfig,
    cached_singleDatasetReaders: Dict[str, DatasetReader],
) -> ((DatasetReader | TaskMixtureReader), Dict[str, DatasetReader]):
    """
    Args:
        task_mixture:
        dataset_config:
        cached_singleDatasetReaders:

    Returns:
        dataset_reader
        cached_singleDatasetReaders
    """
    if task_mixture is None:
        return get_singleDatasetReader(dataset_config, cached_singleDatasetReaders)
    else:
        return get_taskMixtureReader(
            task_mixture,
            mixtureSubset_size,
            mixtureSubset_id,
            dataset_config,
            cached_singleDatasetReaders,
        )
