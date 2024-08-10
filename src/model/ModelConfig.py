import json
import os

from src.utils.Config import Config


def format_modelName(model_name: str) -> str:
    """
    Removes any directory prefix in model_name and replace / with -

    Args:
        model_name:

    Returns:

    """
    return model_name.replace("/fruitbasket/models/", "").replace("/", "-")


class ModelConfig(Config):
    def __init__(
        self,
        config_filepaths=None,
        update_dict=None,
    ):
        """
        Args:
            configDict_toInitializeFrom:
            update_dict:
        """
        super().__init__()

        self.pretraining_mixture = None
        self.pretrained_model = None
        self.max_seq_len = None

        self.filepath_to_load_model = None

        self.freeze_backbone = None
        self.language_or_vision = None

        self.clip_load_classification_head = None

        # Update config with values from list of files
        if config_filepaths:
            for filename in config_filepaths:
                super()._update_fromDict(
                    json.load(open(filename)),
                    assert_keyInUpdateDict_isValid=True,
                )

        # Update config with values from dict
        if update_dict:
            super()._update_fromDict(
                update_dict,
                assert_keyInUpdateDict_isValid=True,
            )

    def get_experimentDir(self):
        # Required for language
        if self.pretrained_model is not None:
            experiment_dir = format_modelName(self.pretrained_model)
        # Required for vision
        else:
            assert self.architecture is not None
            experiment_dir = self.architecture
        if self.peft_method is not None:
            experiment_dir = os.path.join(
                experiment_dir, format_modelName(self.peft_method)
            )
        else:
            experiment_dir = os.path.join(experiment_dir, "full_model")

        if self.pretraining_mixture is not None:
            experiment_dir = os.path.join(experiment_dir, self.pretraining_mixture)

        if self.freeze_backbone:
            experiment_dir = os.path.join(experiment_dir, "freze_backbone")

        return experiment_dir
