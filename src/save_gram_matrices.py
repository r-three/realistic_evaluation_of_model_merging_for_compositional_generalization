import argparse
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


def get_model(model_config, evaluation_config, merge_peft_weights):

    if model_config.language_or_vision == "vision":
        classifier_head, _ = get_classifierHeadAndCLIPModel(
            model_config, None, evaluation_config, {}
        )
    else:
        classifier_head = None
    model, _ = load_model(model_config, classifier_head, {}, device=device)
    model.eval()

    return model


def get_dataset_iterator(model_config, evaluation_config):
    evaluationDataset_config = evaluation_config.get_datasetConfig()
    dataset_reader, _ = get_datasetReader(
        task_mixture=None,
        mixtureSubset_size=None,
        mixtureSubset_id=None,
        dataset_config=evaluationDataset_config,
        cached_singleDatasetReaders={},
    )
    if model_config.language_or_vision == "language":
        dataset = LanguageDataset(
            dataset_reader.get_dataset(evaluationDataset_config.split),
            evaluationDataset_config,
            model.tokenize_fn,
            "train",
            device=device,
        )
    else:
        dataset = VisionDataset(
            dataset_reader.get_dataset(evaluationDataset_config.split),
            evaluationDataset_config,
            model.preprocess_fn,
            "train",
            device=device,
        )

    dataset_iterator = getSingleEpoch_OfBatches(
        dataset, evaluation_config.eval_batch_size
    )

    return dataset_iterator


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


def compute_gram_matrix(activations: torch.tensor, mask: torch.tensor) -> torch.tensor:
    """
    Compute X^TX

    Args:
        activations:
        mask:

    Returns:
        tensor
    """
    if mask is not None:
        # [num_activations, input_dim]
        masked_activations = (
            activations.flatten(0, 1)
            * mask.flatten(0, 1).to(activations.device)[:, None]
        )
    else:
        masked_activations = activations
    return torch.matmul(masked_activations.T, masked_activations)


def compute_gram_matrices(model, dataset_iterator):
    """
    Compute gram matrix

    Args:
        model:
        dataset_iterator:

    Returns:
        gram_matrix:
    """

    sum_gram_matrices = {}

    total_num_dec_tok = 0
    total_num_enc_tok = 0

    stored_input_activations = {}

    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, MultiheadAttention):
            module.register_forward_hook(
                partial(
                    save_inputActivations,
                    stored_input_activations,
                    module_name,
                )
            )

    # For encoder-decoder models (specifically T5), is the module form the encoder or decoder.  Note that k, v in the encoder decoder attention come from the encoder
    is_encoder_or_decoder = lambda module_name: (
        "encoder"
        if re.fullmatch(
            ".*encoder.*|.*decoder.*EncDecAttention.(k|v).*",
            module_name,
        )
        else "decoder"
    )

    for idx, batch in tqdm(enumerate(dataset_iterator)):

        num_dec_tok = None
        num_enc_tokens = None

        with torch.no_grad():
            # Don't need backward pass in model since only forward pass activations are needed
            loss = model(batch)

            for module_name, activations in stored_input_activations.items():

                # For encoder_decoder, the number of input activations for a linear layer is the number of encoder tokens for encoder and the number of decoder tokens for decoder
                if model_config.language_or_vision == "encoder_decoder":
                    if is_encoder_or_decoder(module_name) == "encoder":
                        mask = batch["input_mask"]
                        num_enc_tokens = torch.sum(batch["input_mask"])
                    else:
                        assert is_encoder_or_decoder(module_name) == "decoder"
                        mask = batch["target_mask"]
                        num_dec_tok = torch.sum(batch["target_mask"])

                # For CLIP, the number of input activations is the actual number of input activations since there is no mask.
                else:
                    assert model_config.language_or_vision == "clip"
                    if len(activations.shape) > 2:
                        activations = activations.flatten(0, 1)
                    num_dec_tok = activations.shape[0]
                    mask = None

                gram_matrix = compute_gram_matrix(activations, mask)

                if module_name not in sum_gram_matrices:
                    sum_gram_matrices[module_name] = gram_matrix
                else:
                    sum_gram_matrices[module_name] += gram_matrix

        total_num_dec_tok += num_dec_tok

        if num_enc_tokens is not None:
            total_num_enc_tok += num_enc_tokens

        model.zero_grad()

    with torch.no_grad():
        final_gram_matrices = {}
        for module_name, gram_matrix in sum_gram_matrices.items():
            # Normalization constant depends on whether it is encoder or decoder models
            if is_encoder_or_decoder(module_name) == "encoder":
                constant = total_num_enc_tok
            else:
                assert is_encoder_or_decoder(module_name) == "decoder"
                constant = total_num_dec_tok

            final_gram_matrices[module_name + ".weight"] = (
                (gram_matrix / constant).detach().contiguous().cpu()
            )

    return final_gram_matrices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=True,
        help="The output path for saving the gram matrix",
    )
    # Use to load the config for loading model and dataset_iterator
    parser = addConfigArguments_toParser(
        parser,
        add_trainingArguments=False,
        add_inferenceArguments=True,
        add_mergingArguments=False,
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    The code below calls lots of other code. To implement saving the gram matrix for your own use case, it is suggested to modify the code and to load the model and dataset_iterator for your own use case. 
    """
    model_config, evaluation_config, _ = construct_configs(
        args, "eval", is_merging=False
    )

    # Load model. This requires a dataset config and relies on lots of other code. A simpler alternative can be replaced with just loading the model. See warning below if using merge_peft_weights
    model = get_model(model_config, evaluation_config, args.merge_peft_weights)

    # Load iterator to loop through dataset. This requires a dataset config and relies on lots of other code. A simpler alternative is to just load your own dataset preprocesor. The iterator should iterate through batches, where each batch is a dictionary passed into the model forward pass.
    dataset_iterator = get_dataset_iterator(model_config, evaluation_config)

    gram_matrix = compute_gram_matrices(
        model,
        dataset_iterator,
    )

    torch.save(
        gram_matrix,
        args.output_path,
    )
