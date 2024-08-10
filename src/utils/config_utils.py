import argparse
import ast
import copy
import json
from typing import Any, Callable, Dict, List

from src.data.DatasetConfig import DatasetConfig
from src.eval.EvaluationConfig import EvaluationConfig
from src.model.ModelConfig import ModelConfig
from src.train.TrainingConfig import TrainingConfig
from src.merging.MergingConfig import MergingConfig


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


def parse_configFilepaths(
    config_filepaths: List[str], config_types: List[str]
) -> Dict[str, List[str]]:
    """
    Groups filepaths based on type (training_run, dataset, ...)

    Args:
        config_filepaths:
        config_types:

    Raises:
        ValueError: Invalid config type

    Returns:
        parsed_filepaths: maps config_types to their filepaths
    """
    parsed_filepaths = {}
    for type in config_types:
        parsed_filepaths[type] = []

    for filepath in config_filepaths:
        found_filepathType = False
        for config_type in parsed_filepaths.keys():
            for configType_prefix in [
                "/" + config_type + "/",
                config_type.strip("s") + "_config.json",
            ]:
                if configType_prefix in filepath:
                    parsed_filepaths[config_type].append(filepath)
                    found_filepathType = True
                    break

        if not found_filepathType:
            raise ValueError(f"Cannot find type of filepath {filepath}")

    return parsed_filepaths


def construct_configs(args, train_or_eval: str, is_merging: bool):
    """

    Args:
        args:
        train_or_eval:
        is_merging:

    Returns:
        training_config (if train) or model_config, evaluation_config (if eval)
    """
    if train_or_eval == "train":
        config_types = [
            "model",
            "training_dataset",
            "training_run",
            "evaluation_dataset",
            "evaluation_run",
        ]
    elif train_or_eval == "eval":
        config_types = [
            "model",
            "evaluation_dataset",
            "evaluation_run",
        ]
    else:
        assert train_or_eval is None

    if is_merging:
        config_types.append("merging")

    parsed_filepaths = parse_configFilepaths(args.config_filepaths, config_types)
    model_config = ModelConfig(
        config_filepaths=parsed_filepaths["model"], update_dict=args.model_kwargs
    )
    evaluation_dataset_config = DatasetConfig(
        config_filepaths=parsed_filepaths["evaluation_dataset"],
        update_dict=args.evaluation_dataset_kwargs,
    )

    evaluation_config = EvaluationConfig(
        evaluation_dataset_config,
        config_filepaths=parsed_filepaths["evaluation_run"],
        update_dict=args.evaluation_run_kwargs,
    )

    # Load merging config if passed in
    if is_merging:
        merging_config = MergingConfig(
            config_filepaths=parsed_filepaths["merging"],
            update_dict=args.merging_kwargs,
        )
    else:
        merging_config = None

    if train_or_eval == "train":
        training_dataset_config = DatasetConfig(
            config_filepaths=parsed_filepaths["training_dataset"],
            update_dict=args.training_dataset_kwargs,
        )
        training_config = TrainingConfig(
            model_config,
            training_dataset_config,
            evaluation_config,
            config_filepaths=parsed_filepaths["training_run"],
            update_dict=args.training_run_kwargs,
        )
        return training_config, merging_config
    elif train_or_eval == "eval":
        return model_config, evaluation_config, merging_config
    else:
        return model_config, merging_config


def update_datasetConfig(
    dataset_config: DatasetConfig, dataset_updateDict: Dict
) -> DatasetConfig:
    """
    Args:
        dataset_config:
        dataset_args:

    Returns:
        newDataset_config
    """
    newDataset_keyValues = dataset_config.get_key_values()
    newDataset_keyValues.update(dataset_updateDict)
    newDataset_config = DatasetConfig(
        update_dict=newDataset_keyValues,
    )
    return newDataset_config


def update_evaluationConfig(
    evaluation_config: EvaluationConfig,
    dataset_updateDict: Dict,
    evaluation_updateDict: Dict,
) -> EvaluationConfig:
    """
    Update evaluation config

    Args:
        evaluation_config:
        dataset_updateDict:
        evaluation_updateDict:

    Returns:
        new_evaluationConfig
    """
    evaluationDataset_config = evaluation_config.get_datasetConfig()

    newDataset_updateDict = evaluationDataset_config.get_key_values()
    if dataset_updateDict is not None:
        newDataset_updateDict.update(dataset_updateDict)
    new_evaluationDatasetConfig = DatasetConfig(None, update_dict=newDataset_updateDict)

    newEvaluation_updateDict = evaluation_config.get_key_values()
    newEvaluation_updateDict.update(evaluation_updateDict)
    new_evaluationConfig = EvaluationConfig(
        evaluationDataset_config=new_evaluationDatasetConfig,
        update_dict=newEvaluation_updateDict,
    )

    return new_evaluationConfig


def update_modelConfig(model_config: ModelConfig, update_dict: Dict) -> ModelConfig:
    """

    Args:
        model_config:
        update_dict:

    Returns:

    """
    new_updateDict = model_config.get_key_values()
    new_updateDict.update(update_dict)
    new_modelConfig = ModelConfig(None, update_dict=new_updateDict)

    return new_modelConfig


def update_mergingConfig(
    merging_config: MergingConfig, update_dict: Dict
) -> MergingConfig:
    """

    Args:
        merging_config:
        update_dict:

    Returns:

    """
    new_updateDict = merging_config.get_key_values()
    new_updateDict.update(update_dict)
    new_mergingConfig = MergingConfig(None, update_dict=new_updateDict)

    return new_mergingConfig


def update_trainingConfig(
    training_config: TrainingConfig,
    model_updateDict: Dict,
    trainingDataset_updateDict: Dict,
    evaluationDataset_updateDict: Dict,
    evaluation_updateDict: Dict,
    training_updateDict: Dict,
) -> TrainingConfig:
    """
    Args:
        evaluation_config:
        dataset_args:

    Returns:
        newEvaluation_config
    """

    new_modelConfig = update_modelConfig(
        training_config.get_modelConfig(), model_updateDict
    )
    new_trainingDatasetConfig = update_datasetConfig(
        training_config.get_datasetConfig(), trainingDataset_updateDict
    )
    new_evaluationConfig = update_evaluationConfig(
        training_config.get_evaluationConfig(),
        evaluationDataset_updateDict,
        evaluation_updateDict,
    )

    newTraining_config = TrainingConfig(
        model_config=new_modelConfig,
        trainingDataset_config=new_trainingDatasetConfig,
        evaluation_config=new_evaluationConfig,
        update_dict=training_updateDict,
    )
    return newTraining_config


def addConfigArguments_toParser(
    parser,
    add_trainingArguments: bool,
    add_inferenceArguments: bool,
    add_mergingArguments: bool,
):
    """

    Args:
        parser:
        add_trainingArguments:
        add_inferenceArguments:
        add_mergingArguments:

    Returns:
        parser
    """
    parser.add_argument("-c", "--config_filepaths", action="store", type=str, nargs="*")
    parser.add_argument(
        "-m", "--model_kwargs", nargs="*", action=ParseKwargs, default={}
    )
    if add_mergingArguments:
        parser.add_argument(
            "-mm", "--merging_kwargs", nargs="*", action=ParseKwargs, default={}
        )
    if add_trainingArguments:
        parser.add_argument(
            "-td",
            "--training_dataset_kwargs",
            nargs="*",
            action=ParseKwargs,
            default={},
        )
        parser.add_argument(
            "-tr", "--training_run_kwargs", nargs="*", action=ParseKwargs, default={}
        )
    if add_trainingArguments or add_inferenceArguments:
        parser.add_argument(
            "-ed",
            "--evaluation_dataset_kwargs",
            nargs="*",
            action=ParseKwargs,
            default={},
        )
        parser.add_argument(
            "-er", "--evaluation_run_kwargs", nargs="*", action=ParseKwargs, default={}
        )
    return parser
