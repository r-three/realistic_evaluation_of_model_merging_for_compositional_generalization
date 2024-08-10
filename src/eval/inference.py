import argparse
import logging
import os

import torch

from src.data.dataset_reader.dataset_reader import DatasetReader
from src.data.dataset_reader.task_mixtures import *
from src.eval.evaluation import evaluate_onDatasets
from src.eval.EvaluationConfig import EvaluationConfig
from src.eval.utils import *
from src.eval.utils import saveResult_acrossTasks
from src.model.load_model import *
from src.model.ModelConfig import ModelConfig
from src.utils.config_utils import *
from src.utils.distributed import *
from src.utils.io import *
from src.utils.utils import *


def inference(
    device,
    model_config: ModelConfig,
    evaluation_config: EvaluationConfig,
    experiment_dir: str,
    title: str,
    cached_models: Dict[str, Any],
    cached_singleDatasetReaders: Dict[str, DatasetReader],
):
    """

    Args:
        device:
        world_size:
        port:
        model_config:
        evaluation_config:
        experiment_dir:
        title: title to use when writing out inference scores
        cached_models:
        cached_singleDatasetReaders:

    Returns:
        cached_models:
        cached_singleDatasetReaders
    """

    if model_config.language_or_vision == "vision":
        classifier_head, cached_models = get_classifierHeadAndCLIPModel(
            model_config, None, evaluation_config, cached_models
        )
    else:
        classifier_head = None

    model, cached_models = load_model(
        model_config, classifier_head, cached_models, device=device
    )

    (
        scores,
        evaluation_dirs,
        cached_singleDatasetReaders,
    ) = evaluate_onDatasets(
        model,
        model_config,
        evaluation_config,
        os.path.join(experiment_dir, "predictions", "inference"),
        cached_singleDatasetReaders,
        device,
    )

    inferenceScores_fp = os.path.join(
        experiment_dir, f"{evaluation_config.get_datasetConfig().split}_scores"
    )

    # Check score is not None since for DDP, result will be None except for the node 0
    if scores is not None:
        # Save all the scores in .json file
        append_json(
            {"scores": scores, "runs_dir": evaluation_dirs},
            inferenceScores_fp + ".json",
            pretty_print=True,
        )

        def getScore_fn(dataset_score):
            return str(
                round(
                    dataset_score[evaluation_config.get_datasetConfig().split][
                        "average"
                    ]
                    * 100,
                    1,
                )
            )

        formatted_tasks = getFormattedTask_fromEvaluationConfig(evaluation_config)
        if isinstance(formatted_tasks, list):
            saveAverage_acrossTasks = True
        else:
            saveAverage_acrossTasks = False

        saveResult_acrossTasks(
            formatted_tasks,
            scores,
            lambda dataset_score: getScore_fn(dataset_score),
            title,
            inferenceScores_fp + ".txt",
            saveAverage_acrossTasks=saveAverage_acrossTasks,
        )

    return cached_models, cached_singleDatasetReaders


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = addConfigArguments_toParser(
        parser,
        add_trainingArguments=False,
        add_inferenceArguments=True,
        add_mergingArguments=False,
    )
    parser.add_argument("-e", "--experiment_dir", type=str)
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument("--merged_model", type=str)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting inference")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.experiment_dir is not None:
        checkpoint_dir = os.path.join(args.experiment_dir, "checkpoints")
        checkpoint_suffix = getFile_inDirectory(checkpoint_dir, "best_checkpoint")

        if args.config_filepaths is None:
            args.config_filepaths = []
        args.config_filepaths.extend(
            [
                os.path.join(args.experiment_dir, "configs", "model_config.json"),
                os.path.join(
                    args.experiment_dir, "configs", "evaluation_dataset_config.json"
                ),
                os.path.join(
                    args.experiment_dir, "configs", "evaluation_run_config.json"
                ),
            ]
        )

        if args.model_kwargs is None:
            args.model_kwargs = {}

        # If no checkpoint passed in, use the best checkpoint as a default
        if (
            "filepath_to_load_model" not in args.model_kwargs
            or args.model_kwargs["filepath_to_load_model"] is None
        ):
            args.model_kwargs.update(
                {
                    "filepath_to_load_model": os.path.join(
                        checkpoint_dir,
                        checkpoint_suffix,
                    )
                }
            )

    if args.merged_model is not None:
        if args.model_kwargs is None:
            args.model_kwargs = {}
        args.model_kwargs.update(
            {
                "filepath_to_load_model": args.merged_model,
            }
        )

    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config, evaluation_config, _ = construct_configs(
        args, "eval", is_merging=False
    )

    assert args.output_dir is not None, "Output directory must be specified"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference(
        device,
        model_config,
        evaluation_config,
        args.output_dir,
        title=None,
        cached_models={},
        cached_singleDatasetReaders={},
    )
