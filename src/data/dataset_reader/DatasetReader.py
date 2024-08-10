import json
import logging
import os
from typing import Any, Callable, Dict, List

from src.data.DatasetConfig import DatasetConfig


class DatasetReader(object):
    def __init__(self, dataset_config: DatasetConfig):
        """

        Args:
            dataset_config:
        """
        self.dataset_config = dataset_config
        self.cached_origData = {}
        self.cached_datasets = {}

    def _get_originalData(self):
        raise NotImplementedError

    def get_dataset(self, train_or_eval):
        raise NotImplementedError

    def get_datasetMetrics(self):
        raise NotImplementedError
