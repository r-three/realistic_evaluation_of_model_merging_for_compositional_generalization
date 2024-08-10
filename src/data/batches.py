from typing import Any, Callable, Dict, List

import torch
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler

from src.data.Dataset import LanguageDataset
from src.utils.distributed import is_distributedSetup


def _create_dataLoader(
    pytorch_dataset: LanguageDataset,
    batch_size: int,
    should_shuffle: bool,
):
    """
    Use the dataset and collate_fn from the pytorch_dataset to construct a data_loader

    Args:
        pytorch_dataset:
        batch_size:
        should_shuffle:
        device:

    Returns:
        data_loader:
    """
    data_loader = data.DataLoader(
        pytorch_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=should_shuffle,
        collate_fn=pytorch_dataset.collate_fn,
    )

    return None, data_loader


def getMultipleEpochs_ofBatches(
    pytorch_dataset: LanguageDataset,
    batch_size: int,
    should_shuffle: bool,
):
    """
    Iterator that loops through a dataset multiple times as needed

    Args:
        pytorch_dataset:
        batch_size:
        should_shuffle:
        world_size:
        device:

    Yields:
        batch of data
    """
    data_loader = _create_dataLoader(pytorch_dataset, batch_size, should_shuffle)

    while True:
        for x in data_loader:
            yield x


def getSingleEpoch_OfBatches(pytorch_dataset: LanguageDataset, batch_size: int):
    """
    Args:
        pytorch_dataset:
        batch_size:

    Yields:

    """
    _, data_loader = _create_dataLoader(pytorch_dataset, batch_size, False)

    for x in data_loader:
        yield x
