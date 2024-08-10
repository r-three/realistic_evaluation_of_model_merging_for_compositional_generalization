import argparse
from typing import Dict

import open_clip
import torch
from transformers import AutoModelForSeq2SeqLM

"""
This is meant to be a simple self-contained file to compute the trimmed model for TIES-Merging to be compatible with git-theta. 
"""


def convert_checkpoint_to_tensor(
    checkpoint: Dict[str, torch.tensor]
) -> (torch.tensor, list):
    """
    Flatten all the parameters in a checkpoint and then concatenate them into one long vector.
    Returns the parameter name and sizes so that the checkpoint can be reconstructed from the parameters.

    Args:
        checkpoint:

    Returns:
        tensor:
        parameter_sizes:
    """
    parameters = []
    parameter_sizes = []
    for parameter_name, parameter_value in checkpoint.items():
        parameters.append(parameter_value.flatten().contiguous())
        parameter_sizes.append((parameter_name, parameter_value.shape))
    tensor = torch.cat(parameters, dim=0).contiguous()
    return tensor, parameter_sizes


def convert_tensor_to_checkpoint(
    tensor: torch.tensor, parameter_sizes: list
) -> Dict[str, torch.tensor]:
    """
    Convert a tensor into a checkpoint, using the ordering and parameter sizes in parameter_sizes.
    This is the inverse operation of convert_checkpoint_to_tensor

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

        checkpoint[parameter_name] = torch.clone(
            tensor[start_idx:end_idx].reshape(parameter_shape).contiguous()
        )
        start_idx = end_idx
    return checkpoint


def trim(tensor, percent_to_drop=0.8):
    """

    Args:
        tensor:  flattened tensor
        percent_to_drop:

    Returns:
        tensor with only top_k values kept
    """
    assert percent_to_drop > 0 and percent_to_drop < 1

    assert len(tensor.shape) == 1

    dim = tensor.shape[0]
    num_to_drop = int(dim * percent_to_drop)
    num_to_keep = dim - num_to_drop

    magnitudes = tensor.abs()

    # Since kthvalue finds the kth smallest element in each row, we find the kth smallest element in this tensor
    min_value_to_keep, _ = magnitudes[None, :].kthvalue(num_to_keep, dim=1)

    # mask for the top (num_to_keep) parameters
    mask = magnitudes >= min_value_to_keep

    if torch.sum(mask) != num_to_keep:
        print(
            f"WARNING: Mask has {torch.sum(mask).item()} values but wanted to keep {num_to_keep} values"
        )

    return tensor * mask


def compute_trimmed_model(
    pretrained_checkpoint: Dict[str, torch.tensor], checkpoint: Dict[str, torch.tensor]
) -> Dict[str, torch.tensor]:
    """

    Args:
        pretrained_checkpoint:
        checkpoint:

    Returns:
        trimmed_model
    """
    # keys in the pretrained checkpoint and checkpoint must match
    if set(pretrained_checkpoint.keys()) != set(checkpoint.keys()):

        print(
            "Pretrained checkpoint keys not in checkpoint",
            set(pretrained_checkpoint.keys()).difference(set(checkpoint.keys())),
        )
        print(
            "Checkpoint keys not in pre-trained checkpoint",
            set(checkpoint.keys()).difference(set(pretrained_checkpoint.keys())),
        )
        raise ValueError("Keys in checkpoint and pretrained checkpoint do not match ")

    # Compute the trimmed mdoel
    with torch.no_grad():

        task_vector = {
            param_name: checkpoint[param_name] - pretrained_checkpoint[param_name]
            for param_name in checkpoint.keys()
        }

        del pretrained_checkpoint
        del checkpoint

        flattened_task_vector, parameter_sizes = convert_checkpoint_to_tensor(
            task_vector
        )
        del task_vector

        flattened_trimmed_model = trim(flattened_task_vector.cuda(), K=0.8)
        flattened_trimmed_model = flattened_trimmed_model.contiguous()

        trimmed_model = convert_tensor_to_checkpoint(
            flattened_trimmed_model, parameter_sizes
        )
        del flattened_trimmed_model

    return trimmed_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name",
        default=None,
        type=str,
        required=True,
        help="The pretrained model name. This can be filepath or model name from huggingface",
    )
    parser.add_argument(
        "--checkpoint_filepath",
        default=None,
        type=str,
        required=True,
        help="The filepath with the checkpoint to trim",
    )
    parser.add_argument(
        "--model_architecture",
        default="encoder_decoder",
        type=str,
        choices=["encoder_decoder", "decoder", "clip"],
        help="The model architecture. This is needed to determine the base class when loading the pretrained model ",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=True,
        help="The output path for saving the trimmed model",
    )
    args = parser.parse_args()

    """
    WARNINGS: 
    1) For PEFT modules, modification are required to load the pre-trained checkpoint. 
    See src/model/load_model.py for examples 
    2) It is highly recommended to compute the trim on CPU for several reason. For large models, the memory requirement might be large when computing the trimmed model. There will not be memory issues if running on CPU. Also, note that kthvalue is not deterministic on GPU and there can be stability issues when computing kthvalue on GPU (especially for large models) where the kthvalue might not even be close to the kth smallest value.  
    """
    device = torch.device("cpu")

    # One hot fix to account for parameter name mismatches in saved checkpoint
    map_pretrained_param_name = lambda param_name: "transformer." + param_name

    # Load pre-trained checkpoint by loading the backbone model and then loading the model
    if args.model_architecture == "encoder_decoder":
        model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_name)
    else:
        assert args.model_architecture == "clip"
        clip, _, preprocess_fn = open_clip.create_model_and_transforms(
            args.pretrained_model_name, pretrained="laion2b_s34b_b79k"
        )

    pretrained_checkpoint = {}
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            pretrained_checkpoint[map_pretrained_param_name(param_name)] = param.to(
                device
            )

    # Load the checkpoint
    checkpoint = torch.load(args.checkpoint_filepath, map_location=device)

    # Compute trimmed checkpoint by zeroing out all but the top 20% of a task vector
    trimmed_model = compute_trimmed_model(pretrained_checkpoint, checkpoint)

    torch.save(
        trimmed_model,
        args.output_path,
    )
