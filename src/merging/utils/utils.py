import torch
import os
from src.merging.utils.checkpoints import *
from src.merging.utils.model_ops import *
from src.model.load_model import *


def getMergingConfig_withDifferentMixtureSubsetIds(merging_config):
    all_mergingConfig = []
    if merging_config.mixture_subset_max_id is not None:
        assert merging_config.mixture_subset_min_id is not None
        for mixture_subset_id in list(
            range(
                merging_config.mixture_subset_min_id,
                merging_config.mixture_subset_max_id,
            )
        ):
            new_experiment_name = merging_config.experiment_name
            if new_experiment_name is None:
                new_experiment_name = f"mixture_subset_{mixture_subset_id}"
            else:
                new_experiment_name = (
                    new_experiment_name + f"/mixture_subset_{mixture_subset_id}"
                )
            all_mergingConfig.append(
                update_mergingConfig(
                    merging_config,
                    {
                        "mixture_subset_id": mixture_subset_id,
                        "experiment_name": new_experiment_name,
                    },
                )
            )
    else:
        assert merging_config.mixture_subset_min_id is None

        if merging_config.mixture_subset_id is not None:
            new_experiment_name = merging_config.experiment_name
            if new_experiment_name is None:
                new_experiment_name = (
                    f"mixture_subset_{merging_config.mixture_subset_id}"
                )
            else:
                new_experiment_name = (
                    new_experiment_name
                    + f"/mixture_subset_{merging_config.mixture_subset_id}"
                )
            all_mergingConfig.append(
                update_mergingConfig(
                    merging_config,
                    {
                        "mixture_subset_id": merging_config.mixture_subset_id,
                        "experiment_name": new_experiment_name,
                    },
                )
            )
        else:
            all_mergingConfig.append(merging_config)
    return all_mergingConfig


def iterate11Values_from0To1() -> List[float]:
    """

    Returns:

    """
    all_mergingLambda = []
    for i in range(0, 11):
        all_mergingLambda.append(i / 10.0)
    return all_mergingLambda


def iterate10Values_from10To100() -> List[int]:
    """

    Args:
        number_of_iterations:

    Returns:

    """
    all_numberOfIterations = []
    for i in range(10, 110, 10):
        all_numberOfIterations.append(i)

    return all_numberOfIterations


def getMerging_experimentDir(
    model_config: ModelConfig,
    instruction_format: str,
    merging_config,
):
    """

    Args:
        model_config:
        instruction_format:
        merging_config

    Returns:
        str
    """
    datasetMixtureToMerge_str = (
        merging_config.task_mixture
        if isinstance(merging_config.task_mixture, str)
        else "_".join(merging_config.task_mixture)
    )

    if merging_config.mixture_subset_size is not None:
        numTasksToMerge_str = f"{merging_config.mixture_subset_size}_tasks"

    else:
        numTasksToMerge_str = "all_tasks"

    if model_config.peft_method is None:
        experiment_dir = os.path.join(
            "exp_out",
            "merging",
            instruction_format,
            datasetMixtureToMerge_str,
            numTasksToMerge_str,
            format_modelName(model_config.pretrained_model),
            merging_config.checkpoint_descriptor,
            merging_config.get_experimentDir(),
        )

    else:
        experiment_dir = os.path.join(
            "exp_out",
            "merging",
            instruction_format,
            datasetMixtureToMerge_str,
            numTasksToMerge_str,
            format_modelName(model_config.pretrained_model),
            model_config.peft_method,
            merging_config.checkpoint_descriptor,
            merging_config.get_experimentDir(),
        )

    return experiment_dir


def getNewModelConfig_withMergedWeights(model_config):
    """

    Args:
        model_config:

    Raises:
        ValueError:
    """
    assert model_config.peft_method is not None

    if model_config.peft_method == "ia3":
        model_updateDict = {"merge_ia3": True}
    elif model_config.peft_method == "lora":
        model_updateDict = {"merge_lora": True}
    else:
        raise ValueError(f"Invalid peft method {model_config.peft_method}")

    model_config = update_modelConfig(model_config, model_updateDict)
    return model_config


def getNewModelConfig_withLoadedWeights(
    model_config, merge_peft_weights, model_updateDict
):
    """

    Args:
        model_config:
        merge_peft_weights:
        model_updateDict

    Raises:
        ValueError:
    """
    if merge_peft_weights:
        # For LoRA, since we merge the weights, we load the pre-trained model and then load the merged weights into the pretrained model
        if model_config.peft_method == "lora":
            assert model_config.merge_lora
            model_updateDict.update({"load_merged_lora": True, "merge_lora": None})
        # For IA3, since we merge the weights, we load the pre-trained model and then load the merged weights into the pretrained model
        elif model_config.peft_method == "ia3":
            assert model_config.merge_ia3
            model_updateDict.update({"load_merged_ia3": True, "merge_ia3": None})
        else:
            raise ValueError(f"Invalid peft method {model_config.peft_method}")

    model_config = update_modelConfig(model_config, model_updateDict)
    return model_config


def getParameterNames_toIgnore(model_config):
    # ViT implies there is a classification head that has to be merged
    if "ViT" in model_config.pretrained_model:
        parameters_toIgnore = [
            ".*clip.transformer.*",
            ".*clip.ln_final.*",
            ".*clip.token_embedding.weight.*",
            ".*clip.text_projection.*",
            ".*clip.positional_embedding.*",
            ".*clip.logit_scale.*",
        ]
    else:
        parameters_toIgnore = []

    return parameters_toIgnore
