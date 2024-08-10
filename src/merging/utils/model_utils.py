from typing import Dict

import numpy as np
import torch

from src.eval.EvaluationConfig import EvaluationConfig


def convertCheckpoint_toTensor(
    checkpoint: Dict[str, torch.tensor]
) -> (torch.tensor, list):
    """

    Args:
        checkpoint:

    Returns:
        tensor:
        parameter_sizes
    """
    parameters = []
    parameter_sizes = []
    for parameter_name, parameter_value in checkpoint.items():
        parameters.append(parameter_value.flatten().contiguous())
        torch.cuda.empty_cache()
        parameter_sizes.append((parameter_name, parameter_value.shape))
    tensor = torch.cat(parameters, dim=0).contiguous()
    return tensor, parameter_sizes


def convertTensor_toCheckpoint(
    tensor: torch.tensor, parameter_sizes: list
) -> Dict[str, torch.tensor]:
    """

    Args:
        tensor:
        parameter_sizes:

    Returns:
        checkpoint
    """
    checkpoint = {}
    start_idx = 0
    for parameter_name, parameter_shape in parameter_sizes:
        parameter_size = parameter_shape.numel()
        end_idx = start_idx + parameter_size

        # It was causing memory issues without cloning. Probably because the memory cannot be freed
        # otherwise
        checkpoint[parameter_name] = torch.clone(
            tensor[start_idx:end_idx].reshape(parameter_shape).contiguous()
        )
        start_idx = end_idx
    return checkpoint


def sample_label(model, batch, evaluation_config: EvaluationConfig, metrics: list[str]):
    """
    Sample a label from the log probs of the model

    Args:
        model:
        batch:
        evaluation_config:
        metrics:

    Returns:

    """
    with torch.no_grad():
        if "Accuracy" in metrics:
            (_, score_ofChoices, _, _) = model.predict_mulChoice(
                batch, evaluation_config.length_normalization
            )
            # batch_size must be 1
            assert len(score_ofChoices) == 1
            probs = np.exp(score_ofChoices[0])
            normalized_probs = probs / np.sum(probs)
            # Sample lbl from predicted distribution
            sampled_lbl = np.random.choice(len(normalized_probs), p=normalized_probs)

            assert (
                len(batch["answer_choices"][0]) == batch["answer_choices_ids"].shape[0]
            )

            # Get the answer_ids from the corresponding sample lbl
            max_targetLen = batch["answer_choices_ids"].shape[1]
            target_idx = (
                torch.tensor([sampled_lbl])
                .to(batch["answer_choices_ids"].device)[:, None]
                .repeat((1, max_targetLen))
            )

            target_ids = torch.gather(batch["answer_choices_ids"], 0, target_idx)
            target_mask = torch.gather(batch["answer_choices_mask"], 0, target_idx)
            batch["target_ids"] = target_ids
            batch["target_mask"] = target_mask

        if "Squad" in metrics or "F1" in metrics:
            sampled_ids, _ = model.generate(
                batch, evaluation_config.max_gen_len, sample_tokens=True
            )
            sampled_ids = torch.tensor(sampled_ids).to(batch["input_ids"].device)
            batch["target_ids"] = sampled_ids
            batch["target_mask"] = (
                (sampled_ids != model.get_tokenizer().pad_token_id)
                .int()
                .to(sampled_ids.device)
            )

    return batch


def scale_nonDiagonalElements(matrix: torch.tensor, scale_lambda: float):
    """
    Scale nondiagonal elements of matrix

    Args:
        matrix:
        scale_lambda:

    Returns:
        scaled_matrix
    """
    scaled_matrix = scale_lambda * matrix + (1 - scale_lambda) * torch.diag_embed(
        torch.diagonal(matrix)
    )
    return scaled_matrix


def preprocess_CLIP(loaded_checkpoints, pretrained_checkpoint=None):
    """

    Args:
        loaded_checkpoints:
        pretrained_checkpoint:

    Returns:

    """

    parameters_toDelete = []

    for parameter_name, _ in list(loaded_checkpoints.values())[0].items():
        if (
            "clip.transformer" in parameter_name
            or "clip.ln_final" in parameter_name
            or "clip.token_embedding.weight" in parameter_name
            or "clip.text_projection" in parameter_name
            or "clip.positional_embedding" in parameter_name
            or "clip.logit_scale" in parameter_name
        ):
            parameters_toDelete.append(parameter_name)

    for checkpoint_fp, checkpoint in loaded_checkpoints.items():
        for parameter_name in parameters_toDelete:
            del checkpoint[parameter_name]

    if pretrained_checkpoint is not None:
        for parameter_name in parameters_toDelete:
            del pretrained_checkpoint[parameter_name]

    return (loaded_checkpoints, pretrained_checkpoint)
