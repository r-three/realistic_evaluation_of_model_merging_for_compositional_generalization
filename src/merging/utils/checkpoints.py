import os

import torch

from src.data.dataset_reader.task_mixtures import *
from src.merging.utils.model_ops import *
from src.model.load_model import *
from src.model.ModelConfig import ModelConfig, format_modelName
from src.model.utils import *
from src.utils.io import *

MODEL_CHECKPOINTS = {
    "models-google-mt5-xl-lm-adapt": {
        None: {
            "baseline": {
                "tydiqa-korean": "exp_out/p3/tydiqa/korean/models-google-mt5-xl-lm-adapt/full_model/2024-02-06-21-14-42/checkpoints/best_checkpoint_399.pt",
                "wiki_lingua-thai": "exp_out/p3/wiki_lingua/thai/models-google-mt5-xl-lm-adapt/full_model/2024-02-05-12-49-52/checkpoints/best_checkpoint_399.pt",
                "xlwic-de": "exp_out/p3/xlwic/de/models-google-mt5-xl-lm-adapt/full_model/2024-02-05-12-27-53/checkpoints/best_checkpoint_1199.pt",
                "xnli-ar": "exp_out/p3/xnli/ar/models-google-mt5-xl-lm-adapt/full_model/2024-02-07-15-31-30/checkpoints/best_checkpoint_899.pt",
                "squad": "exp_out/p3/squad/models-google-mt5-xl-lm-adapt/full_model/2024-02-06-21-15-12/checkpoints/best_checkpoint_199.pt",
            }
        },
    },
    "ViT-B-32": {
        None: {
            "baseline": {
                "domainnet-clipart-cloth": "exp_out/no_instructions/domainnet/clipart/task_cloth/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-08/checkpoints/best_checkpoint_399.pt",
                "domainnet-clipart-furniture": "exp_out/no_instructions/domainnet/clipart/task_furniture/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-07/checkpoints/best_checkpoint_299.pt",
                "domainnet-clipart-mammal": "exp_out/no_instructions/domainnet/clipart/task_mammal/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-08/checkpoints/best_checkpoint_99.pt",
                "domainnet-clipart-tool": "exp_out/no_instructions/domainnet/clipart/task_tool/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-08/checkpoints/best_checkpoint_699.pt",
                "domainnet-infograph-building": "exp_out/no_instructions/domainnet/infograph/task_building/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-08/checkpoints/best_checkpoint_449.pt",
                "domainnet-infograph-electricity": "exp_out/no_instructions/domainnet/infograph/task_electricity/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-08/checkpoints/best_checkpoint_199.pt",
                "domainnet-infograph-human_body": "exp_out/no_instructions/domainnet/infograph/task_human_body/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-08/checkpoints/best_checkpoint_249.pt",
                "domainnet-infograph-office": "exp_out/no_instructions/domainnet/infograph/task_office/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-08/checkpoints/best_checkpoint_99.pt",
                "domainnet-painting-cold_blooded": "exp_out/no_instructions/domainnet/painting/task_cold_blooded/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-08/checkpoints/best_checkpoint_349.pt",
                "domainnet-painting-food": "exp_out/no_instructions/domainnet/painting/task_food/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-08/checkpoints/best_checkpoint_499.pt",
                "domainnet-painting-nature": "exp_out/no_instructions/domainnet/painting/task_nature/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-08/checkpoints/best_checkpoint_149.pt",
                "domainnet-painting-road_transportation": "exp_out/no_instructions/domainnet/painting/task_road_transportation/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-08/checkpoints/best_checkpoint_149.pt",
                "domainnet-quickdraw-fruit": "exp_out/no_instructions/domainnet/quickdraw/task_fruit/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-09/checkpoints/best_checkpoint_149.pt",
                "domainnet-quickdraw-music": "exp_out/no_instructions/domainnet/quickdraw/task_music/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-08/checkpoints/best_checkpoint_199.pt",
                "domainnet-quickdraw-sport": "exp_out/no_instructions/domainnet/quickdraw/task_sport/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-09/checkpoints/best_checkpoint_149.pt",
                "domainnet-quickdraw-tree": "exp_out/no_instructions/domainnet/quickdraw/task_tree/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-09/checkpoints/best_checkpoint_49.pt",
                "domainnet-real-bird": "exp_out/no_instructions/domainnet/real/task_bird/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-09/checkpoints/best_checkpoint_149.pt",
                "domainnet-real-kitchen": "exp_out/no_instructions/domainnet/real/task_kitchen/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-14/checkpoints/best_checkpoint_99.pt",
                "domainnet-real-shape": "exp_out/no_instructions/domainnet/real/task_shape/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-14/checkpoints/best_checkpoint_599.pt",
                "domainnet-real-vegatable": "exp_out/no_instructions/domainnet/real/task_vegatable/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-14/checkpoints/best_checkpoint_149.pt",
                "domainnet-sketch-insect": "exp_out/no_instructions/domainnet/sketch/task_insect/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-14/checkpoints/best_checkpoint_49.pt",
                "domainnet-sketch-others": "exp_out/no_instructions/domainnet/sketch/task_others/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-14/checkpoints/best_checkpoint_99.pt",
                "domainnet-sketch-sky_transportation": "exp_out/no_instructions/domainnet/sketch/task_sky_transportation/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-14/checkpoints/best_checkpoint_49.pt",
                "domainnet-sketch-water_transportation": "exp_out/no_instructions/domainnet/sketch/task_water_transportation/ViT-B-32/full_model/laion2b_s34b_b79k/2024-01-09-19-59-14/checkpoints/best_checkpoint_349.pt",
            }
        }
    },
}


def get_checkpointName(
    model_config: ModelConfig,
    checkpoint_descriptor: str,
    dataset: str,
) -> str:
    """

    Args:
        model_config:
        checkpoint_descriptor:
        dataset:

    Returns:
        checkpint_filepath
    """
    pretrained_model = format_modelName(model_config.pretrained_model)
    return MODEL_CHECKPOINTS[pretrained_model][model_config.peft_method][
        checkpoint_descriptor
    ][dataset]


def getCheckpointNames_inTaskMixture(
    model_config: ModelConfig,
    merging_config: MergingConfig,
) -> (list[str], list[str]):
    """

    Args:
        model_config:
        merging_config:

    Returns:
        tasks:
        checkpoint_fps:
    """
    checkpoint_fps = []
    tasks = []

    for task in getTasks_inMixture(merging_config.task_mixture):
        formatted_task = getFormattedTask_fromTask(task)

        checkpoint_fps.append(
            get_checkpointName(
                model_config,
                merging_config.checkpoint_descriptor,
                formatted_task,
            )
        )
        tasks.append(task)

    return tasks, checkpoint_fps


def merge_loraWeight(
    A_parameter: torch.Tensor, B_parameter: torch.Tensor
) -> (str, Dict[str, torch.Tensor]):
    """

    Args:
        A_parameterName:
        A_parameter:
        B_parameter:

    Returns:
        merged_lora
    """
    # Multiply out LoRA parameter
    mergedLora_parameter = torch.matmul(B_parameter, A_parameter)
    return mergedLora_parameter


def synchronous_mergeLoraWeights(
    model_config: ModelConfig, checkpoint: Dict[str, torch.tensor]
) -> Dict[str, torch.Tensor]:
    """
    Merge the A and B lora weights together

    Args:
        model_config
        checkpoint:

    Returns:

    """
    # Scaling factor not needed since PEFT module will scale the merged lora
    merged_checkpoint = {}
    for parameter_name, parameter in checkpoint.items():
        assert "lora" in parameter_name

        # Only merge when A is in parameter to not double count the same parameter
        if "A" in parameter_name:
            B_parameterName = parameter_name.replace("A", "B")
            B_parameter = checkpoint[B_parameterName]
            final_parameterName = parameter_name.replace(
                "lora_A.default.", "lora_layer."
            )

            merged_checkpoint[final_parameterName] = torch.matmul(
                B_parameter, checkpoint[parameter_name]
            )

    return merged_checkpoint


def merge_checkpoint(
    model_config: ModelConfig,
    checkpoint: Dict[str, torch.Tensor],
):
    """

    Args:
        model_config:
        checkpoint:
        pretrained_checkpoint:

    Raises:
        NotImplementedError:

    Returns:

    """
    if model_config.merge_lora:
        checkpoint = synchronous_mergeLoraWeights(model_config, checkpoint)
    return checkpoint


def loadCheckpointOrStatistics_fromNames(
    model_config: ModelConfig,
    datasets: List[str],
    checkpoint_names: List[str],
    device,
) -> Dict[str, Dict[str, torch.tensor]]:
    """

    Args:
        model_config:
        checkpoint_fps:
        device:

    Returns:
        checkpoints:
    """

    checkpoints = {}

    for dataset, checkpoint_name in zip(datasets, checkpoint_names):
        checkpoint = torch.load(checkpoint_name, device)
        checkpoint = merge_checkpoint(model_config, checkpoint)
        checkpoints[dataset] = checkpoint

    check_parameterNamesMatch(list(checkpoints.values()))
    return checkpoints


def load_pretrainedCheckpoint(
    model_config: ModelConfig, device
) -> Dict[str, torch.Tensor]:
    """
    Load pretrained checkpoint

    Args:
        model_config:

    Returns:
        pretrained_checkpoint
    """
    if model_config.peft_method is not None:
        if model_config.peft_method == "lora":
            pretrainedCheckpoint_filepath = os.path.join(
                model_config.pretrained_model, "lora.pt"
            )

        if not os.path.exists(pretrainedCheckpoint_filepath):
            raise ValueError(
                f"Requires pre-trained checkpoint to exist at {pretrainedCheckpoint_filepath} but it doesn't"
            )
        pretrained_checkpoint = torch.load(pretrainedCheckpoint_filepath)

        if model_config.merge_lora:
            pretrained_checkpoint = synchronous_mergeLoraWeights(
                model_config, pretrained_checkpoint
            )

    else:
        pretrainedCheckpoint_filepath = model_config.pretrained_model
        pretrained_model = initialize_pretrainedModel(model_config, None, None, device)
        pretrained_checkpoint = get_modelParameters(pretrained_model, device)

    return pretrained_checkpoint
