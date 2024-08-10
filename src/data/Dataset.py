import copy
from typing import Any, Callable, Dict, List

import open_clip
import skimage.io as io
import torch
from PIL import Image
from torch.utils import data

from src.data.DatasetConfig import DatasetConfig
from src.utils.utils import flatten_list

NULL_ANSWER_CHOICE = "NULL_ANSWER"


class LanguageDataset(data.Dataset):
    def __init__(
        self,
        dataset: List[Any],
        dataset_config: DatasetConfig,
        tokenize_fn: Callable,
        train_or_eval: str,
        device,
    ):
        self.dataset = dataset
        self.dataset_config = dataset_config
        self.tokenize_fn = tokenize_fn
        self.train_or_eval = train_or_eval
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, get_idx: int):
        return copy.deepcopy(self.dataset[get_idx])

    def collate_fn(self, batch_ofDatapoints: List[Dict]) -> Dict[Any, List]:
        """
        Convert a batch of datapoints into a datapoint that is batched.  This is meant to override the default collate function in pytorch.

        Args:
            batch_ofDatapoints:

        Returns:

        """
        # tokenize_fn is defined by the model and assumes a generic structure
        # of input and targets

        datapoint_batched = self.tokenize_fn(
            batch_ofDatapoints, self.train_or_eval, self.device
        )
        if "answer_choices" in datapoint_batched:
            # Datasets which have different numbers of answer_choices for
            # different examples add a NULL answer choice so that each example
            # has the same number of choices. These NULL answer choices have to
            # be ignored later on
            listof_nonNullAnswerChoices = []
            for list_answerChoices in datapoint_batched["answer_choices"]:
                nonNull_answerChoices = []
                for answer_choice in list_answerChoices:
                    if answer_choice == NULL_ANSWER_CHOICE:
                        nonNull_answerChoices.append(0)
                    else:
                        nonNull_answerChoices.append(1)
                listof_nonNullAnswerChoices.append(torch.tensor(nonNull_answerChoices))
            datapoint_batched["non_null_answer_choices"] = torch.stack(
                listof_nonNullAnswerChoices, dim=0
            ).to(self.device)

        # Convert lbl to tensor
        if "lbl" in datapoint_batched:
            datapoint_batched["lbl"] = torch.tensor(datapoint_batched["lbl"]).to(
                self.device
            )

        return datapoint_batched


class VisionDataset(data.Dataset):
    def __init__(
        self,
        dataset: List[Any],
        dataset_config: DatasetConfig,
        preprocess_fn,
        train_or_eval: str,
        device,
    ):
        self.dataset = dataset
        self.dataset_config = dataset_config
        self.preprocess_fn = preprocess_fn
        self.device = device

    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, get_idx: int):
        datapoint = self.dataset[get_idx]

        image = self.preprocess_fn(Image.fromarray(io.imread(datapoint["image_path"])))

        datapoint.update(
            {
                "idx": get_idx,
                "image": image,
            }
        )
        return datapoint

    def collate_fn(self, batch_ofDatapoints: List[Dict]) -> Dict[Any, List]:
        """
        Convert a batch of datapoints into a datapoint that is batched.  This is meant to override the default collate function in pytorch.

        Args:
            batch_ofDatapoints:

        Returns:

        """
        datapoint_batched = {}
        for datapoint in batch_ofDatapoints:
            # Gather together all the values per key
            for key, value in datapoint.items():
                if key in datapoint_batched:
                    datapoint_batched[key].append(value)
                else:
                    datapoint_batched[key] = [value]

        if self.device is not None:
            datapoint_batched["image"] = torch.stack(
                datapoint_batched["image"], dim=0
            ).to(self.device)
            datapoint_batched["lbl"] = torch.tensor(datapoint_batched["lbl"]).to(
                self.device
            )

        return datapoint_batched
