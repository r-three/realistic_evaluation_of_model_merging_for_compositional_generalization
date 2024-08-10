from typing import Any, Dict

import open_clip
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.data.dataset_reader.domainnet import *
from src.data.dataset_reader.task_mixtures import *
from src.model.CLIPWrapper import CLIPWrapper
from src.model.EncoderDecoderWrapper import EncoderDecoderWrapper
from src.model.ModelConfig import ModelConfig
from src.model.utils import *


def initialize_languageModel(model_config: ModelConfig, device) -> Any:
    """
    Args:
        model_config,
        device

    Returns:
        model:
    """
    transformer = AutoModelForSeq2SeqLM.from_pretrained(
        model_config.pretrained_model
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.pretrained_model,
        model_max_length=model_config.max_seq_len,
        legacy=False,
    )

    model = EncoderDecoderWrapper(transformer, tokenizer, model_config).to(device)

    return model


def initialize_visionModel(
    model_config: ModelConfig, classifier_head, cached_models, device
):
    assert "ViT" in model_config.pretrained_model

    if cached_models is not None and "clip" in cached_models:
        clip = cached_models["clip"]
    else:
        clip = None

    if cached_models is not None and "preprocess_fn" in cached_models:
        preprocess_fn = cached_models["preprocess_fn"]
    else:
        preprocess_fn = None
    model = CLIPWrapper(model_config, classifier_head, clip, preprocess_fn, device).to(
        device
    )

    return model


def initialize_pretrainedModel(
    model_config: ModelConfig, classifier_head, cached_models, device
) -> Any:
    """
    Initialize pretrained model

    Args:
        model_config:
        num_classes:
        device:

    Returns:
        model
    """
    if model_config.language_or_vision == "language":
        model = initialize_languageModel(model_config, device)
    else:
        assert model_config.language_or_vision == "vision"
        model = initialize_visionModel(
            model_config, classifier_head, cached_models, device
        )
    return model


def loadParameters_intoModel(
    model_config: ModelConfig, model: Any, cached_models
) -> Any:
    """
    Args:
        filepath_to_load_model:
        model:

    Returns:
        model:
    """
    if model_config.filepath_to_load_model is not None:
        if model_config.filepath_to_load_model in cached_models:
            print("Using cached merged model")
            parameters = cached_models[model_config.filepath_to_load_model]
        else:
            parameters = torch.load(model_config.filepath_to_load_model)
        # Check parameters to load are subset of model and will be loaded
        modelParameters_names = set(parameters.keys())

        # Load classifcation head from different model
        if model_config.clip_load_classification_head is not None:
            classification_head = torch.load(model_config.clip_load_classification_head)
            parameters["classification_layer.weight"] = classification_head[
                "classification_layer.weight"
            ]
            parameters["classification_layer.bias"] = classification_head[
                "classification_layer.bias"
            ]

        modelStateDict_keys = set(model.state_dict().keys())

        if not modelParameters_names.issubset(modelStateDict_keys):
            import ipdb

            ipdb.set_trace()
        model.load_state_dict(parameters, strict=False)

    return model


def load_model(
    model_config: ModelConfig, classifier_head, cached_models, device
) -> (Any, Dict[str, Any]):
    """
    Args:
        model_config:
        classifier_head:
        cached_models,
        device:

    Returns:
        model:
        cached_models:
    """

    if "pretrained_model" in cached_models:
        model = cached_models["pretrained_model"]
        print(
            "Warning: Using cached pre-trained model, which loaded a previous checkpoint, and now used again to load a new checkpoint. This only works if the checkpoints have the same parameters "
        )
    else:
        model = initialize_pretrainedModel(
            model_config,
            classifier_head,
            cached_models,
            device=device,
        )
        cached_models["pretrained_model"] = model

    model = loadParameters_intoModel(model_config, model, cached_models)

    return model, cached_models


def getClassIdsAndLbls_forTask(tokenizer, task) -> (List, List[int]):
    """
    Get class ids and lbls for a task

    Args:
        tokenizer:
        task:

    Returns:
        prompts:
        lbls:
    """
    lbls = []
    prompts = []
    for class_name, lbl in CATEGORIES[task].items():
        prompt = f"A photo of a {class_name}"
        prompts.append(prompt)
        lbls.append(CATEGORY_GLOBAL_IDX[task] + lbl)
    return prompts, lbls


def get_classifierHeadAndCLIPModel(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    evaluation_config,
    cached_models: Dict[str, Any],
):
    """
    Get classifier head for CLIP using the open vocabulary where the representations are from the prompts of the classes

    Args:
        model_config:
        training_config:
        evaluation_config:
        cached_models:

    Returns:

    """
    assert training_config is None or evaluation_config is None

    task_mixture = None

    dataset_config = (
        training_config.get_datasetConfig()
        if training_config is not None
        else evaluation_config.get_datasetConfig()
    )

    if ("clip" not in cached_models) or ("preprocess_fn" not in cached_models):
        clip, _, preprocess_fn = open_clip.create_model_and_transforms(
            model_config.pretrained_model, pretrained=model_config.pretraining_mixture
        )
        cached_models["clip"] = clip
        cached_models["preprocess_fn"] = preprocess_fn
    tokenizer = open_clip.get_tokenizer(model_config.pretrained_model)

    if dataset_config.shift_lbls and "classifier_head_shift_lbls" in cached_models:
        classifier_head = cached_models["classifier_head_shift_lbls"]
    else:
        # For multiple datasets, get the prompts and labels for all the datasets in the dataset mixture
        if dataset_config.shift_lbls:
            if task_mixture is None:
                task_mixture = "domainnet"
            else:
                assert "domainnet" in task_mixture
            all_prompts = []
            all_lbls = []
            seen_tasks = set()
            # mixture_subset_size and seed for random tasks is fixed to None since we always load the full classifier head, even if we are only interested in a subset of the tasks
            for dataset in getTasks_inMixture(task_mixture):
                newDataset_config = update_datasetConfig(
                    dataset_config, getDatasetUpdateDict_fromTask(dataset)
                )
                if newDataset_config.task not in seen_tasks:
                    prompts, lbls = getClassIdsAndLbls_forTask(
                        tokenizer, newDataset_config.task
                    )
                    all_prompts.extend(prompts)
                    all_lbls.extend(lbls)
                    seen_tasks.add(newDataset_config.task)
        # Get the prompts and labels for a single dataset
        else:
            all_prompts, all_lbls = getClassIdsAndLbls_forTask(
                tokenizer, dataset_config.task
            )

        # Sort the prompts to match the labels, since the label ids represent the index of the vector in the classifier head
        prompts = list(zip(all_prompts, all_lbls))
        sortedClassIds_byLbls = sorted(prompts, key=lambda x: x[1])
        sorted_prompts, sorted_lbls = zip(*sortedClassIds_byLbls)

        prompt_ids = tokenizer(sorted_prompts)

        # Encode prompts to get actualy classifier head
        with torch.no_grad():
            class_emb = clip.encode_text(prompt_ids)

        classifier_head = torch.nn.Linear(
            class_emb.shape[1], class_emb.shape[0], bias=False
        )
        classifier_head.weight.data = class_emb

        cached_models["classifier_head_shift_lbls"] = classifier_head

    return classifier_head, cached_models
