import copy
import json
import re
from typing import Any, Callable, Dict, List

from _ctypes import PyObj_FromPtr  # see https://stackoverflow.com/a/15012814/355230


class NoIndent(object):
    """Value wrapper."""

    def __init__(self, value):
        if not isinstance(value, (list, dict)):
            raise TypeError("Only lists and dictionaries can be wrapped")
        self.value = value


# From https://stackoverflow.com/questions/42710879/write-two-dimensional-list-to-json-file
class NoIndentEncoder(json.JSONEncoder):
    FORMAT_SPEC = "@@{}@@"  # Unique string pattern of NoIndent object ids.
    regex = re.compile(FORMAT_SPEC.format(r"(\d+)"))  # compile(r'@@(\d+)@@')

    def __init__(self, **kwargs):
        # Keyword arguments to ignore when encoding NoIndent wrapped values.
        ignore = {"cls", "indent"}

        # Save copy of any keyword argument values needed for use here.
        self._kwargs = {k: v for k, v in kwargs.items() if k not in ignore}
        super(NoIndentEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (
            self.FORMAT_SPEC.format(id(obj))
            if isinstance(obj, NoIndent)
            else super(NoIndentEncoder, self).default(obj)
        )

    def iterencode(self, obj, **kwargs):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.

        # Replace any marked-up NoIndent wrapped values in the JSON repr
        # with the json.dumps() of the corresponding wrapped Python object.
        for encoded in super(NoIndentEncoder, self).iterencode(obj, **kwargs):
            match = self.regex.search(encoded)
            if match:
                id = int(match.group(1))
                no_indent = PyObj_FromPtr(id)
                json_repr = json.dumps(no_indent.value, **self._kwargs)
                # Replace the matched id string with json formatted representation
                # of the corresponding Python object.
                encoded = encoded.replace(
                    '"{}"'.format(format_spec.format(id)), json_repr
                )

            yield encoded


def isDictOrList_onFirstLevel(obj: object) -> bool:
    """

    Args:
        obj:

    Returns:
        on_first_level:
    """
    on_first_level = True

    if isinstance(object, dict):
        iterator = list(obj.values())

    elif isinstance(object, list):
        iterator = object
    else:
        return False

    for child in iterator:
        if isinstance(child, dict) or isinstance(child, list):
            on_first_level = False

    return on_first_level


def noIndent_dictOrList_onFirstLevel(dict_toCheck: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Returns dictionary where there is no indentation for dictionaries and list on the
    first level. This is to help with readibility.

    Args:
        dict_toCheck:

    Returns:
        dict with properties specified above
    """
    copy_dict = copy.deepcopy(dict_toCheck)

    for key, value in copy_dict.items():
        if isDictOrList_onFirstLevel(value):
            copy_dict[key] = NoIndent(value)

        else:
            if isinstance(value, dict):
                copy_dict[key] = noIndent_dictOrList_onFirstLevel(value)

    return copy_dict
