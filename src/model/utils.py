import logging
from typing import Any, Dict

import torch

from src.utils.io import *


def computeLogProb_perChoice(
    logProb_ofAnswerChoiceIds: torch.Tensor,
    mask_ofAnswerChoices: torch.Tensor,
    nonNullAnswerChoices_mask: torch.Tensor,
    num_answerChoices: int,
    maxLen_ofAnswerChoices: int,
    length_normalization: bool,
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Get the log probs of each choice

    Args:
        logProb_ofAnswerChoiceIds: [batch_size * num_answerChoices * max_answerChoiceLen]
        answerChoices_mask: [batch_size * num_answerChoices * max_answerChoiceLen, ]
        nonNullAnswerChoices_mask: [batch_size * num_answerChoices]
        num_answerChoices:
        maxLen_ofAnswerChoices:
        length_normalization: whether to sum or aveage the log prob of the ids for each answer

    Returns:
        logProb_ofAnswerChoices:
        logProb_ofAnswerChoiceIds_zeroOutPadIds:
        answerChoices_len:
    """

    # [batch_size, num_answerChoices, max_answerChoiceLen]
    logProb_ofAnswerChoiceIds = logProb_ofAnswerChoiceIds.reshape(
        -1, num_answerChoices, maxLen_ofAnswerChoices
    )

    mask_ofAnswerChoices = mask_ofAnswerChoices.reshape(
        -1, num_answerChoices, maxLen_ofAnswerChoices
    )
    # Zero out padded out tokens so we their log probability is not included
    logProb_ofAnswerChoiceIds_zeroOutPadIds = (
        logProb_ofAnswerChoiceIds * mask_ofAnswerChoices
    )

    # Sum the log_prob across ids per answer choice
    logProb_ofAnswerChoices = torch.sum(logProb_ofAnswerChoiceIds_zeroOutPadIds, dim=2)

    answerChoices_len = torch.sum(mask_ofAnswerChoices, dim=2)

    if length_normalization:
        logProb_ofAnswerChoices = logProb_ofAnswerChoices / answerChoices_len

    nonNullAnswerChoices_mask = nonNullAnswerChoices_mask.reshape(-1, num_answerChoices)

    # For answer choices which are null, we mask them out by setting them to the smallest value
    logProb_ofAnswerChoices = (1 - nonNullAnswerChoices_mask) * torch.finfo(
        logProb_ofAnswerChoices.dtype
    ).min + nonNullAnswerChoices_mask * logProb_ofAnswerChoices

    return (
        logProb_ofAnswerChoices,
        logProb_ofAnswerChoiceIds_zeroOutPadIds,
        answerChoices_len,
    )


def get_modelParameters(model: Any, device: Any = None) -> Dict[str, torch.Tensor]:
    """
    Args:
        model:
        device:

    Returns:

    """
    model_parameters = {}
    parameter_count = 0
    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if device is not None:
                model_parameters[parameter_name] = parameter.to(device)
            else:
                model_parameters[parameter_name] = parameter
            parameter_count += parameter.numel()
    logging.info(f"Parameter count: {parameter_count}")
    return model_parameters


def greedyGeneration_encoderDecoder(
    transformer,
    input_ids,
    input_mask,
    bos_tokenId,
    eos_tokenId,
    pad_tokenId,
    equalSign_tokenId,
    max_generationLength,
):
    """
    Assumes model is encoder_decoder model and caches input first.
    Converts the first eos token to an equal sign when decoding

    Args:
        model:
        input_ids:
        input_mask:
        bos_tokenId:
        eos_tokenId:
        pad_tokenId:
        max_generationLength:

    Returns:
        generated_ids: [batch_size, max_generationLength]
    """
    past_key_values = None
    batch_size = input_ids.shape[0]

    # Decode starting with bos_token_id
    # [batch_size, 1]
    current_decoderInputIds = torch.tensor([bos_tokenId] * batch_size)[:, None].to(
        input_ids.device
    )
    # Decoder mask is fixed to always be 1. We don't need to ignore any tokens in the decoder since we just truncate any token after the eos token
    # [batch_size, 1]
    current_decoderMask = torch.ones((batch_size, 1)).to(input_ids.device)

    encoder_outputs = transformer.get_encoder()(input_ids, input_mask)

    generated_ids = current_decoderInputIds

    hasSequence_hitEOS = torch.zeros(size=(batch_size, 1), dtype=torch.int).to(
        input_ids.device
    )

    generated_scores = []

    for i in range(max_generationLength):
        # attention_mask must be passed in for encoder_decoder models, even if we pass the
        # encoder_outputs, since the attention_mask is used to compute the cross_attention mask
        # for encoder decoder models
        output = transformer(
            attention_mask=input_mask,
            decoder_input_ids=current_decoderInputIds,
            decoder_attention_mask=current_decoderMask,
            encoder_outputs=encoder_outputs,
            use_cache=True,
            past_key_values=past_key_values,
        )

        # Update current key values
        past_key_values = output.past_key_values

        predicted_nextToken = torch.argmax(output.logits, -1)

        generated_scores.append(output.logits)

        # If sequence has hit end, then every token afterwards should be a PAD token
        predicted_nextToken = (
            1 - hasSequence_hitEOS
        ) * predicted_nextToken + hasSequence_hitEOS * pad_tokenId

        # Update whether has sequence has hit end of sequence
        isToken_EOSToken = predicted_nextToken == eos_tokenId

        generated_ids = torch.cat((generated_ids, predicted_nextToken), dim=1)

        hasSequence_hitEOS = torch.bitwise_or(hasSequence_hitEOS, isToken_EOSToken)

        # Exit loop if every sequence has hit EOS
        if torch.sum(hasSequence_hitEOS) == batch_size:
            break

        current_decoderInputIds = predicted_nextToken

    generation_output = {"sequences": generated_ids, "scores": generated_scores}
    return generation_output
