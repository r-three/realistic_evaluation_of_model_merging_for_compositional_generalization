import copy
from typing import Any, Callable, Dict, List

import torch

from src.eval.EvaluationConfig import EvaluationConfig
from src.utils.io import *
from src.utils.stats import *
from src.utils.utils import *

KEYS_TO_REMOVE = ["ids", "mask", "flatten", "image"]


def prepare_batchOfEvalInfo(batch: Dict) -> Dict:
    """
    Prepare batch for evaluation by removing ids and masks

    Args:
        batch:

    Returns:
        batchOf_evalInfo
    """
    batchOf_evalInfo = copy.deepcopy(batch)

    for key, value in batch.items():
        # Remove keys that should not be written out
        for key_toRemove in KEYS_TO_REMOVE:
            if key_toRemove in key and key in batchOf_evalInfo:
                del batchOf_evalInfo[key]

        # Convert tensors to list
        if key in batchOf_evalInfo and torch.is_tensor(batchOf_evalInfo[key]):
            batchOf_evalInfo[key] = value.cpu().numpy().tolist()

    return batchOf_evalInfo


def tokenize_additionalInput(tokenizer, total_input, device):

    tokenized_dict = tokenizer(
        total_input,
        return_tensors="pt",
        padding="longest",
        truncation="longest_first",
    )

    input_ids = tokenized_dict["input_ids"]
    attention_mask = tokenized_dict["attention_mask"]

    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

    return input_ids, attention_mask


def additionalRound_ofGeneration(
    model,
    evaluation_config: EvaluationConfig,
    batch: Dict,
    batchOf_evalInfo: Dict,
    device,
) -> Dict:
    """

    Args:
        model:
        evaluation_config:
        batch:
        batchOf_evalInfo:
        device:

    Returns:
        batchOf_evalInfo
    """
    total_input = []
    for input, generatation, additional_prompt in zip(
        batch["input"], batchOf_evalInfo["prediction_text"], batch["additional_input"]
    ):
        # New inpt consist of the original input, original model generation, the new prompt, and an equal sign
        total_input.append(input + generatation + additional_prompt + " = ")

    # Tokenize input called additional_input
    input_ids, input_mask = tokenize_additionalInput(
        model.input_tokenizer, total_input, device
    )

    batch["additional_input_ids"] = input_ids
    batch["additional_input_mask"] = input_mask
    generated_ids, generated_txt = model.generate(
        batch,
        evaluation_config.max_gen_len,
        evaluation_config.sample_tokens,
        input_key="additional_",
    )
    batchOf_evalInfo["intermediate_text"] = total_input
    batchOf_evalInfo.update(
        {
            "generated_ids": generated_ids,
            "prediction_text": generated_txt,
        }
    )
    return batchOf_evalInfo


def concatenate_scores(
    list_results: List[Dict], getKey_fn: Callable[[Dict], Any]
) -> Dict[Any, Dict]:
    """Concatenate the score and using the function to get the key

    Args:
        list_results:
        getKey_fn:

    Returns:
        concatenated_scores:
    """
    concatenated_scores = {}
    for result in list_results:
        concatenated_scores[getKey_fn(result)] = result["score"]
    return concatenated_scores


def get_allRunsDirs(all_results: List[Dict]) -> List[Dict]:
    """
    Args:
        all_results:

    Returns:
        all_runDirs
    """
    all_runDirs = []
    for result in all_results:
        all_runDirs.append(result["evaluation_dir"])
    return all_runDirs


def average_scores(list_results: List[Dict], split: str) -> Dict:
    """
    Average the scores. Assumes each result is a dictionary of
    {"score": {"average": ...}}

    Args:
        list_results:
        split:

    Returns:
        average_score
    """
    individual_averageScores = list(
        map(lambda x: x["score"][split]["average"], list_results)
    )
    average_score = get_average(individual_averageScores)

    return {"average": average_score}


def saveResult_acrossTasks(
    tasks: List[str],
    scores: Dict,
    getScore_fn: Callable[[Dict], Dict],
    title: str,
    score_fp: str,
    saveAverage_acrossTasks: bool,
):
    """
    Save the average of the average score for each dataset

    Args:
        tasks:
        scores:
        getScore_fn:
        title:
        score_fp:
        saveAverage_acrossTasks:

    Returns:

    """
    labels_toDisplay = []
    scores_toDisplay = []

    if saveAverage_acrossTasks:
        labels_toDisplay.append("Avg.")
        scores_toDisplay.append(str(round(scores["average"] * 100, 1)))

    if isinstance(tasks, str):
        labels_toDisplay.append(tasks)
        scores_toDisplay.append(getScore_fn(scores))
    else:
        for dataset in tasks:
            labels_toDisplay.append(dataset)
            scores_toDisplay.append(getScore_fn(scores[dataset]))

    label_str = ",".join(labels_toDisplay)
    scores_str = ",".join(scores_toDisplay)

    with open(score_fp, "a+") as f:
        if title is not None:
            f.write(title + "\n")
        f.write(label_str + "\n")
        f.write(scores_str + "\n")
