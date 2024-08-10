import argparse
import os
import re
from functools import partial

import torch
import torch.nn as nn
from open_clip.multihead_attention import MultiheadAttention
from tqdm import tqdm

from src.data.batches import getSingleEpoch_OfBatches
from src.data.Dataset import LanguageDataset, VisionDataset
from src.data.readers import get_datasetReader
from src.eval.utils import *
from src.merging.utils.model_ops import *
from src.merging.utils.model_utils import *
from src.merging.utils.utils import *
from src.model.load_model import *
from src.model.utils import *
from src.utils.config_utils import *
from src.utils.utils import *


def compute_taskVectors(
    invidual_checkpoints: list[dict[str, torch.tensor]],
    pretrained_checkpoint: dict[str, torch.tensor],
):
    """
    Because various methods use task vectors, we make it a static function

    Args:
        task_checkpoints:
        pretrained_model:

    Returns:

    """

    taskVector_models = list(
        map(
            lambda checkpoint: elementWise_subtract(checkpoint, pretrained_checkpoint),
            invidual_checkpoints,
        )
    )
    return taskVector_models


# From https://github.com/prateeky2806/ties-merging/tree/main/src
def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask

    return M * final_mask, final_mask.float().mean(dim=1)


def get_statisticsDir(modelCheckpoint_fp: str) -> (str, str):
    """
    statistics dir is created under experiment directory

    Args:
        modelCheckpoint_fp:

    Returns:
        statistics_dir:
        checkpoint_filename:
    """
    exp_dir = os.path.dirname(modelCheckpoint_fp).replace("/checkpoints", "")
    checkpoint_filename = os.path.basename(modelCheckpoint_fp)
    statistics_dir = os.path.join(exp_dir, "statistics")
    os.makedirs(statistics_dir, exist_ok=True)
    return statistics_dir, checkpoint_filename


def compute_blockwiseFisher(model) -> Dict[str, torch.Tensor]:
    """
    Args:
        model:

    Returns:
        perExample_fisher
    """
    perExample_fisher = {}
    model_parameters = dict((key, value) for key, value in model.named_parameters())
    for parameter_name, parameter in model_parameters.items():
        if parameter.requires_grad:
            flattened_grad = torch.flatten(parameter.grad)
            outer_product = torch.outer(flattened_grad, flattened_grad)

            if parameter_name not in perExample_fisher:
                perExample_fisher[parameter_name] = outer_product.detach().cpu()
            else:
                perExample_fisher[parameter_name] += outer_product.detach().cpu()
    return perExample_fisher


def compute_diagonalFisher(model) -> Dict[str, torch.Tensor]:
    """
    Args:
        model:

    Returns:
        perExample_fisher
    """
    perExample_fisher = {}
    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            if parameter_name not in perExample_fisher:
                perExample_fisher[parameter_name] = torch.zeros_like(parameter.data)
            perExample_fisher[parameter_name] += torch.square(parameter.grad)
    return perExample_fisher


def save_inputActivations(
    saved_activations: Dict[str, torch.tensor],
    module_name: str,
    module,
    input,
    output,
) -> None:
    """
    Saves the input activations. Hook to attach to model

    Args:
        saved_activations:
        module_name:
        module:
        input:
        output:
    """
    saved_activations[module_name] = input[0].float()


def update_gramMatrix(
    module_name: str,
    gram_matrix: Dict[str, torch.tensor],
    runningSum_gramMatrix: Dict[str, torch.tensor],
) -> Dict[str, torch.tensor]:
    """

    Args:
        module_name:
        gram_matrix:
        runningSum_gramMatrix:

    Returns:
        runningSum_gramMatrix
    """
    if module_name not in runningSum_gramMatrix:
        runningSum_gramMatrix[module_name] = gram_matrix
    else:
        runningSum_gramMatrix[module_name] += gram_matrix

    return runningSum_gramMatrix


def computeGramMatrix_forLinearLayer(
    activations: torch.tensor, mask: torch.tensor
) -> Dict[str, torch.tensor]:
    """

    Args:
        matrix:
        mask:

    Returns:
        _description_
    """
    if mask is not None:
        # [batch_size * num_tokens, input_dim]
        masked_activations = (
            activations.flatten(0, 1)
            * mask.flatten(0, 1).to(activations.device)[:, None]
        )
    else:
        masked_activations = activations
    return torch.matmul(masked_activations.T, masked_activations)


def getCheckpointFisher_name(
    modelCheckpoint_fp: str, merging_config: ModelConfig
) -> str:
    """

    Args:
        modelCheckpoint_fp:
        merging_config:

    Returns:
        name_saveFisher
    """
    statistics_dir, checkpoint_filename = get_statisticsDir(modelCheckpoint_fp)

    name_saveFisher = os.path.join(
        statistics_dir,
        checkpoint_filename
        + "-"
        + merging_config.get_fisherArguments().replace("/", "-"),
    )

    if merging_config.merge_peft_weights:
        name_saveFisher += "_merge_peft_weights"

    return name_saveFisher


def getCheckpointTrimmed_name(modelCheckpoint_fp: str, merging_config) -> str:
    """

    Args:
        modelCheckpoint_fp:
        merging_config:

    Returns:
        named_tiesTrimmed
    """
    statistics_dir, checkpoint_filename = get_statisticsDir(modelCheckpoint_fp)

    named_tiesTrimmed = os.path.join(
        statistics_dir, f"{checkpoint_filename}_ties_trimmed"
    )

    if merging_config.merge_peft_weights:
        named_tiesTrimmed += "_merge_peft_weights"

    return named_tiesTrimmed


def normalize_metadata(
    stored_metadata: Dict[str, torch.tensor],
    count: int,
    total_numberOfEncoderTokens: int,
) -> Dict[str, torch.tensor]:
    """
    Divide metadata by count

    Args:
        stored_metadata:
        count:

    Returns:
        normalized_metadata
    """
    normalized_metadata = {}

    # If total number of encoder tokens is 0, we just use the count
    if total_numberOfEncoderTokens == 0:
        for parameter_name, parameter in stored_metadata.items():
            normalized_metadata[parameter_name] = parameter / count
    else:
        for module_name, parameter in stored_metadata.items():
            if re.fullmatch(
                ".*encoder.*|.*decoder.*EncDecAttention.(k|v).*", module_name
            ):
                num_tokens = total_numberOfEncoderTokens
            else:
                num_tokens = count
            normalized_metadata[module_name] = parameter / num_tokens

    return normalized_metadata


def detach_metadata(
    stored_metadata: Dict[str, torch.tensor]
) -> Dict[str, torch.Tensor]:
    """
    Move metadata to CPU

    Args:
        stored_metadata:

    Returns:
        detached_metadata
    """
    detached_metadata = {}
    for parameter_name, parameter in stored_metadata.items():
        detached_metadata[parameter_name] = parameter.detach().contiguous().cpu()
    return detached_metadata


def trim_model(task_vector: torch.tensor):
    """
    Trim the task vector

    Args:
        task_vector:

    Returns:

    """
    updated_checks, *_ = topk_values_mask(task_vector[None, :], K=20, return_mask=False)
    return updated_checks.squeeze(0).contiguous()


def save_trimmedModel(
    device,
    model_config,
    evaluation_config,
    merging_config,
):
    """

    Args:
        device:
        model_config:
        evaluation_config:
        merging_config:
    """

    checkpointStatistic_name = getCheckpointTrimmed_name(
        model_config.filepath_to_load_model, merging_config
    )

    pretrained_checkpoint = load_pretrainedCheckpoint(model_config, device)

    checkpoint = loadCheckpointOrStatistics_fromNames(
        model_config, ["ties"], [model_config.filepath_to_load_model], device
    )["ties"]

    with torch.no_grad():
        task_vector = compute_taskVectors([checkpoint], pretrained_checkpoint)[0]

        del checkpoint

        flattened_taskVector, parameter_sizes = convertCheckpoint_toTensor(task_vector)
        del task_vector
        del pretrained_checkpoint
        trimmed_model = trim_model(flattened_taskVector)
        unflattened_trimmedModel = convertTensor_toCheckpoint(
            trimmed_model, parameter_sizes
        )
        del trimmed_model

    torch.save(unflattened_trimmedModel, checkpointStatistic_name)
    del unflattened_trimmedModel


def save_fisher(
    device,
    model_config: ModelConfig,
    evaluation_config: EvaluationConfig,
    merging_config: MergingConfig,
):
    """

    Args:
        device:
        model_config:
        evaluation_config:
        merging_config:
    """

    checkpointStatistic_name = getCheckpointFisher_name(
        model_config.filepath_to_load_model, merging_config
    )

    if model_config.language_or_vision == "vision":
        classifier_head, _ = get_classifierHeadAndCLIPModel(
            model_config, None, evaluation_config, {}
        )
    else:
        classifier_head = None

    model, _ = load_model(model_config, classifier_head, {}, device=device)

    model.eval()
    evaluationDataset_config = evaluation_config.get_datasetConfig()
    dataset_reader, _ = get_datasetReader(
        task_mixture=None,
        mixtureSubset_size=None,
        mixtureSubset_id=None,
        dataset_config=evaluationDataset_config,
        cached_singleDatasetReaders={},
    )
    metrics = dataset_reader.get_datasetMetrics()

    if model_config.language_or_vision == "language":
        dataset = LanguageDataset(
            dataset_reader.get_dataset("train"),
            evaluationDataset_config,
            model.tokenize_fn,
            "train",
            device=device,
        )
    else:
        dataset = VisionDataset(
            dataset_reader.get_dataset("train"),
            evaluationDataset_config,
            model.preprocess_fn,
            "train",
            device=device,
        )

    iterator = getSingleEpoch_OfBatches(dataset, batch_size=1)
    stored_fisher = {}
    total_numberOfSamples = 0
    total_numberOfEncoderTokens = 0

    # When approximating Fisher with input activations covariance, we add a hook to the module
    if merging_config.fisher_approximation == "input_activations_covariance":
        # For storing input activations covariance, we only count the number of tokens in the target mask for normalization, which assumes a decoder

        stored_inputActivations = {}

        for module_name, module in model.named_modules():

            if isinstance(module, nn.Linear) or isinstance(module, MultiheadAttention):
                module.register_forward_hook(
                    partial(
                        save_inputActivations,
                        stored_inputActivations,
                        module_name,
                    )
                )

    # Normal storing of Fisher
    else:

        def update_storedFisher(perExample_fisher):
            for parameter_name, value in perExample_fisher.items():
                if parameter_name not in stored_fisher:
                    stored_fisher[parameter_name] = value
                else:
                    stored_fisher[parameter_name] += value

    for idx, batch in tqdm(enumerate(iterator)):
        # When computing the true fisher, we have to sample the label from the predicted
        # distribution
        if merging_config.use_true_fisher:
            batch = sample_label(model, batch, evaluation_config, metrics)

        # For input activations covariance approxiation, we only need a forward pass
        # without gradients
        if merging_config.fisher_approximation == "input_activations_covariance":
            number_ofTokens = None  # For decoder, tokens refers to decoder tokens
            number_ofEncoderTokens = None

            with torch.no_grad():
                loss, _ = model(batch)
                for module_name, activations in stored_inputActivations.items():

                    # For encoder_decoder, the number of input activations for a linear layer is the number of encoder tokens for encoder and the number of decoder tokens for decoder
                    if model_config.language_or_vision == "language":
                        if re.fullmatch(
                            ".*encoder.*|.*decoder.*EncDecAttention.(k|v).*",
                            module_name,
                        ):
                            mask = batch["input_mask"]
                            number_ofEncoderTokens = torch.sum(batch["input_mask"])
                        else:
                            mask = batch["target_mask"]
                            number_ofTokens = torch.sum(batch["target_mask"])

                    # For CLIP, the number of input activations can be computed directly without a mask since there is no padding
                    else:
                        model_config.language_or_vision == "vision"
                        if len(activations.shape) > 2:
                            activations = activations.flatten(0, 1)
                        number_ofTokens = activations.shape[0]
                        mask = None
                    stored_fisher = update_gramMatrix(
                        module_name,
                        computeGramMatrix_forLinearLayer(activations, mask),
                        stored_fisher,
                    )

            total_numberOfSamples += number_ofTokens

            if number_ofEncoderTokens is not None:
                total_numberOfEncoderTokens += number_ofEncoderTokens

        else:
            loss, _ = model(batch)
            # Fisher is the gradient of the log likelihood (which is the negative loss of the log prob )
            log_prob = -loss
            log_prob.backward()

            # Compute the per-example Fisher and update the total Fisher
            with torch.no_grad():
                if merging_config.fisher_approximation == "diagonal":
                    perExample_fisher = compute_diagonalFisher(model)
                elif merging_config.fisher_approximation == "blockwise":
                    perExample_fisher = compute_blockwiseFisher(model)
                else:
                    raise ValueError(
                        f"Invalid fisher approximation {merging_config.fisher_approximation}"
                    )
                update_storedFisher(perExample_fisher)
            total_numberOfSamples += 1

        model.zero_grad()

    if merging_config.fisher_approximation == "input_activations_covariance":
        new_storedFisher = {}
        for module_name, gram_matrix in stored_fisher.items():
            new_storedFisher[module_name + ".weight"] = gram_matrix

    with torch.no_grad():
        stored_fisher = normalize_metadata(
            stored_fisher, total_numberOfSamples, total_numberOfEncoderTokens
        )

    print(
        f"Saving fisher computed on {evaluationDataset_config.dataset} with {total_numberOfSamples} samples in {checkpointStatistic_name}"
    )
    # Though the fisher is not a model, it uses the same API of calling torch.save to save
    torch.save(detach_metadata(stored_fisher), checkpointStatistic_name)


def save_statistic(
    device,
    model_config: ModelConfig,
    evaluation_config: EvaluationConfig,
    merging_config: MergingConfig,
):

    if merging_config.method == "ties":
        save_trimmedModel(
            device,
            model_config,
            evaluation_config,
            merging_config,
        )
    else:
        assert merging_config.fisher_approximation is not None
        save_fisher(
            device,
            model_config,
            evaluation_config,
            merging_config,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = addConfigArguments_toParser(
        parser,
        add_trainingArguments=False,
        add_inferenceArguments=True,
        add_mergingArguments=True,
    )
    args = parser.parse_args()

    model_config, evaluation_config, merging_config = construct_configs(
        args, "eval", is_merging=True
    )

    # Load config that will merge the weights
    if merging_config.merge_peft_weights:
        model_config = getNewModelConfig_withMergedWeights(model_config)

    if merging_config.method == "ties":
        # Compute TIES on CPU when saving statistic
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if merging_config.task_mixture is not None:
        tasks, checkpoint_names = getCheckpointNames_inTaskMixture(
            model_config,
            merging_config,
        )
        for task, checkpoint_name in zip(tasks, checkpoint_names):
            print(checkpoint_name)
            newEvaluation_config = update_evaluationConfig(
                evaluation_config,
                getDatasetUpdateDict_fromTask(task),
                {},
            )

            new_modelConfig = update_modelConfig(
                model_config, {"filepath_to_load_model": checkpoint_name}
            )

            save_statistic(
                device,
                new_modelConfig,
                newEvaluation_config,
                merging_config,
            )
    else:
        task = getFormattedTask_fromEvaluationConfigWithSingleTask(evaluation_config)

        checkpoint_name = get_checkpointName(
            model_config,
            merging_config.checkpoint_descriptor,
            task,
        )
        new_modelConfig = update_modelConfig(
            model_config, {"filepath_to_load_model": checkpoint_name}
        )

        save_statistic(
            device,
            new_modelConfig,
            evaluation_config,
            merging_config,
        )
