import json
import logging
import os

from src.data.dataset_reader.DatasetReader import DatasetReader
from src.data.dataset_reader.no_instructions import *
from src.data.dataset_reader.prompt_source import *
from src.data.dataset_reader.domainnet import *
from src.data.DatasetConfig import DatasetConfig

# Maps instruction format to dataset name to dataset reader
DATASETS = {
    "p3": {
        "squad": P3_SQuADReader,
        "xnli": P3_XNLIReader,
        "paws-x": P3_PAWSXReader,
        "xlwic": P3_XLWiCReader,
        "wiki_lingua": P3_WikiLinguaReader,
        "tydiqa": P3_TyDiQAReader,
        "xquad": P3_XQuADReader,
    },
    "no_instructions": {
        "domainnet": DomainNetReader,
    },
}


def get_singleDatasetReader(
    dataset_config: DatasetConfig,
    cached_singleDatasetReaders: dict[(str, str), DatasetReader],
) -> tuple[DatasetReader, dict[tuple[str, str], DatasetReader]]:
    """Get a dataset reader for a single dataset

    Args:
        dataset_config:
        cached_singleDatasetReaders:


    Returns:
        dataset_reader:
        cached_singleDatasetReaders:
    """
    if dataset_config not in cached_singleDatasetReaders:
        cached_singleDatasetReaders[dataset_config] = DATASETS[
            dataset_config.instruction_format
        ][dataset_config.dataset](dataset_config)

    return (
        cached_singleDatasetReaders[dataset_config],
        cached_singleDatasetReaders,
    )
