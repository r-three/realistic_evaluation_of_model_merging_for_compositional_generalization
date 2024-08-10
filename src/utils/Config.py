import ast
import json
from typing import Any, Dict


class Config(object):
    def get_key_values(self):
        """
        Get the key, value pairs in the config if the value is not another Config

        Returns:
            key_values
        """
        key_values = {}
        for key, value in self.__dict__.items():
            if not isinstance(value, Config):
                key_values[key] = value
        return key_values

    def _update_fromDict(
        self, dict_toUpdateFrom: Dict[str, Any], assert_keyInUpdateDict_isValid: bool
    ):
        """
        Update the config using the dictionary

        Args:
            dict_toUpdateFrom:
            assert_keyInUpdateDict_isValid: whether to check each key in the dict_toUpdateFrom is in the Config

        Raises:
            ValueError: Cannot parse the value
        """
        for k, v in dict_toUpdateFrom.items():
            try:
                # For strings that are actually filepaths or regex, literal eval will fail so we have to ignore strings which are filepaths. We check a string is a filepath if a "/" is in string or "*" is in string.
                if not (isinstance(v, str) and ("/" in v or "*" in v)):
                    v = ast.literal_eval(v)
            except ValueError:
                v = v

            if hasattr(self, k):
                setattr(self, k, v)

            else:
                if assert_keyInUpdateDict_isValid:
                    raise ValueError(f"{k} is not in the config")

    def _save_config(self, config_fp: str):
        """
        Save config at filename

        Args:
            filename:

        Returns:

        """
        with open(config_fp, "w+") as f:
            f.write(json.dumps(self.get_key_values(), indent=4, sort_keys=True))
            f.write("\n")
