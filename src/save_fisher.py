import argparse

import torch
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


def get_model(model_config, evaluation_config):

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

    if evaluation_config.eval_batch_size != 1:
        print(
            "Warning: Fisher uses batch_size of 1, so overriding specified batch size with 1"
        )
    dataset_iterator = getSingleEpoch_OfBatches(dataset, 1)

    return dataset_iterator


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


def compute_fisher(model, dataset_iterator):
    """

    Args:
        device:
        world_size:
        model_config:
        evaluation_config:
        merging_config:
        slurm_job_id:
    """

    stored_fisher = {}
    number_of_samples = 0

    for idx, batch in tqdm(enumerate(dataset_iterator)):

        # Compute empirical Fisher
        loss, _ = model(batch)
        # Fisher is the gradient of the log likelihood (which is the negative loss of the log prob )
        log_prob = -loss
        log_prob.backward()

        # Compute the per-example Fisher and update the total Fisher
        with torch.no_grad():
            perExample_fisher = compute_diagonalFisher(model)

            for parameter_name, value in perExample_fisher.items():
                if parameter_name not in stored_fisher:
                    stored_fisher[parameter_name] = value
                else:
                    stored_fisher[parameter_name] += value

        number_of_samples += 1

        model.zero_grad()

    with torch.no_grad():
        final_fisher = {}
        for module_name, fisher in stored_fisher.items():
            final_fisher[module_name] = (
                (fisher / number_of_samples).detach().contiguous().cpu()
            )

    return final_fisher


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

    # Load model. This requires a dataset config and relies on lots of other code. A simpler alternative can be replaced with just loading the model.
    model = get_model(model_config, evaluation_config)

    # Load iterator to loop through dataset. This requires a dataset config and relies on lots of other code. A simpler alternative is to just load your own dataset preprocesor. The iterator should iterate through batches, where each batch is a dictionary passed into the model forward pass.
    dataset_iterator = get_dataset_iterator(model_config, evaluation_config)

    fisher = compute_fisher(model, dataset_iterator)

    torch.save(
        fisher,
        args.output_path,
    )
