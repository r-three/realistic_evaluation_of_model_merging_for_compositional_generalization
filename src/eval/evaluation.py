import logging
from typing import Dict

import torch
from tqdm import tqdm

from src.data.batches import getSingleEpoch_OfBatches
from src.data.Dataset import LanguageDataset, VisionDataset
from src.data.dataset_reader.DatasetReader import DatasetReader
from src.data.dataset_reader.task_mixtures import *
from src.data.readers import get_datasetReader
from src.eval.EvaluationConfig import EvaluationConfig
from src.eval.Evaluator import Evaluator
from src.eval.utils import *
from src.utils.config_utils import *


def evaluate_fromConfig(
    model,
    model_config: ModelConfig,
    evaluation_config: EvaluationConfig,
    prediction_dir: str,
    cached_singleDatasetReaders: Dict[str, DatasetReader],
    device,
) -> (Dict, Dict[str, DatasetReader]):
    """

    Args:
        model:
        evaluation_config:
        prediction_dir:
        cached_singleDatasetReaders:
        device:

    Returns:

    """
    logging.info(f"Evaluating model")

    evaluationDataset_config = evaluation_config.get_datasetConfig()
    dataset_reader, cached_singleDatasetReaders = get_datasetReader(
        task_mixture=None,
        mixtureSubset_size=None,
        mixtureSubset_id=None,
        dataset_config=evaluationDataset_config,
        cached_singleDatasetReaders=cached_singleDatasetReaders,
    )

    model.eval()

    if model_config.language_or_vision == "language":
        dataset = LanguageDataset(
            dataset_reader.get_dataset("eval"),
            evaluationDataset_config,
            model.tokenize_fn,
            "eval",
            device=device,
        )
    else:
        dataset = VisionDataset(
            dataset_reader.get_dataset("eval"),
            evaluationDataset_config,
            model.preprocess_fn,
            "eval",
            device=device,
        )

    evalBatch_iterator = getSingleEpoch_OfBatches(
        dataset, evaluation_config.eval_batch_size
    )
    metrics = dataset_reader.get_datasetMetrics()

    evaluator = Evaluator(
        evaluation_config,
        metrics,
        os.path.join(prediction_dir, evaluation_config.get_datasetConfig().split),
    )

    with torch.no_grad():
        for batch in tqdm(evalBatch_iterator):
            batchOf_evalInfo = prepare_batchOfEvalInfo(batch)

            if "Accuracy" in metrics:
                if model_config.language_or_vision == "language":
                    (
                        predicted_choice,
                        score_ofChoices,
                        logProbs_ofAllChoicesIds,
                        len_allChoices,
                    ) = model.predict_mulChoice(
                        batch, evaluation_config.length_normalization
                    )

                    batchOf_evalInfo.update(
                        {
                            "predicted_choice": predicted_choice,
                            "score_of_choices": score_ofChoices,
                            "log_probs_of_all_choices_ids": logProbs_ofAllChoicesIds,
                            "len_all_choices": len_allChoices,
                        }
                    )
                else:
                    assert model_config.language_or_vision == "vision"
                    (
                        predicted_choice,
                        predicted_logProb,
                    ) = model.predict(batch)

                    batchOf_evalInfo.update(
                        {
                            "predicted_choice": predicted_choice,
                            "predicted_log_prob": predicted_logProb,
                        }
                    )

            if (
                "Squad" in metrics
                or "F1" in metrics
                or "arithmetic" in metrics
                or "kv_substitution" in metrics
                or "kv_substitution_arithmetic" in metrics
                or "sp_rouge" in metrics
            ):
                generated_ids, generated_txt = model.generate(
                    batch,
                    evaluation_config.max_gen_len,
                    evaluation_config.sample_tokens,
                )

                batchOf_evalInfo.update(
                    {"generated_ids": generated_ids, "prediction_text": generated_txt}
                )

                # Append some additional input to the generation and ask the model to continue generating
                # This is needed for KVArithmetic in particular.
                if "additional_input" in batch:
                    batchOf_evalInfo = additionalRound_ofGeneration(
                        model, evaluation_config, batch, batchOf_evalInfo, device
                    )

            evaluator.add_batch(batchOf_evalInfo)

    results = {
        "score": {
            evaluation_config.get_datasetConfig().split: evaluator.get_result(),
        },
        "evaluation_dir": evaluator.get_evaluationRunDir(),
        "evaluation_config": evaluation_config.get_key_values(),
        "dataset_config": evaluation_config.get_datasetConfig().get_key_values(),
    }
    return (
        results,
        cached_singleDatasetReaders,
    )


def evaluate_onDatasets(
    model,
    model_config: ModelConfig,
    evaluation_config: EvaluationConfig,
    prediction_dir: str,
    cached_singleDatasetReaders: Dict[str, DatasetReader],
    device,
) -> (Dict, List[str], Dict[str, DatasetReader]):
    """

    Args:
        model:
        evaluation_config:
        prediction_dir:
        cached_singleDatasetReaders:
        device:

    Returns:
        score:
        runs_dir:
        cached_singleDatasetReaders:
    """
    score = None
    evaluation_dir = None

    # Evaluate each dataset in the mixture separately
    if evaluation_config.task_mixture is not None:
        all_results = []

        for dataset in getTasks_inMixture(evaluation_config.task_mixture):
            dataset_evaluationConfig = update_evaluationConfig(
                evaluation_config, getDatasetUpdateDict_fromTask(dataset), {}
            )
            results, cached_singleDatasetReaders = evaluate_fromConfig(
                model,
                model_config,
                dataset_evaluationConfig,
                prediction_dir,
                cached_singleDatasetReaders,
                device,
            )
            all_results.append(results)

        # Check result is not None since for DDP, result will be None except for the node 0
        if all_results[0] is not None:
            average_score = average_scores(
                all_results, evaluation_config.get_datasetConfig().split
            )

            def getDataset_fn(dataset_score):
                dataset_name = dataset_score["dataset_config"]["dataset"]
                # Account for cross-lingual dataset where we add the language code to the task to construct dataset name to look up checkpoint
                language_code = dataset_score["dataset_config"]["language_code"]
                if language_code is not None:
                    return dataset_name + "-" + language_code
                language = dataset_score["dataset_config"]["language"]
                if language is not None:
                    return dataset_name + "-" + language
                # Account for different domains dataset where we add the domain to the task to construct dataset name to look up checkpoint
                domain = dataset_score["dataset_config"]["domain"]
                if domain is not None:
                    task = dataset_score["dataset_config"]["task"]
                    return dataset_name + "-" + domain + "-" + str(task)

                return dataset_name

            score = concatenate_scores(all_results, getDataset_fn)
            score = deep_update(score, average_score)
            evaluation_dir = get_allRunsDirs(all_results)

    # Evaluate single dataset
    else:
        assert evaluation_config.get_datasetConfig().dataset is not None

        results, cached_singleDatasetReaders = evaluate_fromConfig(
            model,
            model_config,
            evaluation_config,
            prediction_dir,
            cached_singleDatasetReaders,
            device,
        )
        # Check result is not None since for DDP, result will be None except for the node 0
        if results is not None:
            score = results["score"]
            evaluation_dir = results["evaluation_dir"]

    return score, evaluation_dir, cached_singleDatasetReaders
