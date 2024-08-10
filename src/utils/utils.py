import argparse
import os
import random
import re
import sys
from typing import Any, Callable, Dict, List

import numpy as np
import torch


def set_seeds(seed: int):
    """
    Args:
        seed:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def addValues_inDict(original_dict: Dict, new_dict: Dict) -> Dict:
    """
    Update a dict with a new dict by adding the values

    Args:
        original_dict:
        new_dict:

    Returns:
        original_dict
    """
    for k in new_dict.keys():
        if k not in original_dict:
            original_dict[k] = new_dict[k]
        else:
            original_dict[k] += new_dict[k]
    return original_dict


def getValueOfKey_inDictionary(dictionary_toSearch: Dict, keys_toSearchFor: Any):
    """
    Check if key or path of key exists in dictionary and return the value correspoding to the key

    Args:
        dictionary_toSearch:
        keys_toSearchFor: returns the value of the first key that is found in dictionary

    Returns:

    """

    for full_key in keys_toSearchFor:
        # Full key can be path in nested dictionary
        if isinstance(full_key, tuple):
            for key in full_key:
                # If key exists in dictionary, keep searching deeper
                if key in dictionary_toSearch:
                    dictionary_toSearch = dictionary_toSearch[key]

                    # If found value, return it
                    if not isinstance(dictionary_toSearch, dict):
                        return dictionary_toSearch
                    # Continue searching children dictionary_toSearch
                    else:
                        continue
                # Else skip to next key
                else:
                    continue

        else:
            # If key exists in dictionary, return it
            if full_key in dictionary_toSearch:
                dictionary_toSearch = dictionary_toSearch[full_key]

                # If found value, return it
                if not isinstance(dictionary_toSearch, dict):
                    return dictionary_toSearch
                else:
                    raise ValueError(
                        "Key specifies dictionary not value", dictionary_toSearch
                    )
            # Else skip to next key
            else:
                continue

    raise ValueError("None of the keys found", dictionary_toSearch)


def getValueFromKey_matchingRegex(
    dict_ofRegexKeyToValue: Dict, key_toMatch: Any
) -> Any:
    """
    Args:
        dict_regex_keyToValue:
        key_toMatch:

    Returns:

    """
    matching_value = None
    for regex_key, value in dict_ofRegexKeyToValue.items():
        if re.search(regex_key, key_toMatch) is not None:
            matching_value = value
    return matching_value


def flatten_list(list_ofList: List[List]) -> List[Any]:
    """
    Args:
        list_ofList:

    Returns:

    """
    return [item for list in list_ofList for item in list]


def convert_listOfDict_toDictOfList(list_ofDict: List[Dict]) -> Dict[Any, List[Any]]:
    """
    Args:
        list_ofDict:

    Returns:
        dict_ofList
    """
    dict_ofList = {}

    for single_dict in list_ofDict:
        for k, v in single_dict.items():
            if k in dict_ofList:
                dict_ofList[k].append(v)
            else:
                dict_ofList[k] = [v]

    return dict_ofList


# From https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
def convert_dictOfLists_to_listOfDicts(dict_ofLists: Dict) -> List[Dict]:
    """
    Args:
        dictOfLists:

    Returns:
        list_ofDict
    """
    list_ofDicts = []
    for datapoint_values in zip(*dict_ofLists.values()):
        list_ofDicts.append(dict(zip(dict_ofLists, datapoint_values)))
    return list_ofDicts


def map_forDictionaries(my_dict: Dict[Any, Any], map_fn: Callable[[Any], Any]):
    """

    Args:
        my_dict:
        map_fn:

    Returns:
        mapped_dict
    """
    mapped_dict = {}
    for k, v in my_dict.items():
        mapped_dict[k] = map_fn(v)
    return mapped_dict


def print_memUsage(location: str):
    """

    Args:
        loc: location in code to print memory usage
    """
    print(
        "%s mem usage: %.3f GB, %.3f GB, %.3f GB"
        % (
            location,
            float(torch.cuda.memory_allocated() / 1e9),
            float(torch.cuda.memory_reserved() / 1e9),
            float(torch.cuda.max_memory_allocated() / 1e9),
        )
    )
    sys.stdout.flush()


# From https://github.com/pydantic/pydantic/blob/fd2991fe6a73819b48c906e3c3274e8e47d0f761/pydantic/utils.py#L200
def deep_update(mapping: Dict, *updating_mappings):
    """
    Update one dictionary with another dictionary

    Args:
        mapping:
        updating_mappings:

    Returns:
        updated_mappings
    """
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def map_forDictionaries(my_dict: Dict, map_fn: Callable) -> Dict:
    """

    Args:
        my_dict:
        map_fn:

    Returns:
        mapped_dict
    """
    mapped_dict = {}
    for k, v in my_dict.items():
        mapped_dict[k] = map_fn(v)
    return mapped_dict
