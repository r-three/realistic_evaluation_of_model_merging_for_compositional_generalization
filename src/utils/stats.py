import statistics
from statistics import mean
from typing import Any, Callable, Dict, List
import statistics
from scipy.stats import iqr


def get_median(numbers: List[float]) -> float:
    """

    Args:
        numbers:

    Returns:
        median:
    """
    return round(statistics.median(numbers), 3)


def get_standardDeviation(numbers: List[float]) -> float:
    return statistics.stdev(numbers)


def get_interquartileRange(numbers: List[float]) -> float:
    """
    Args:
        numbers:

    Returns:
        IQR range
    """
    return round(iqr(numbers), 3)


def get_average(numbers: List[float]) -> float:
    """
    Args:
        numbers:

    Returns:
        average
    """
    return round(mean(numbers), 4)


def round_list(my_list: List[float], significant_figures: int) -> List[float]:
    """

    Args:
        my_list:
        significant_figures:

    Returns:
        rounded_list
    """
    rounded_list = []

    for number in my_list:
        rounded_list.append(round(number, significant_figures))

    return rounded_list


def round_nestedList(nested_list: List[List[float]], significant_figures: int):
    """
    Round nested list of numbers where list can be any depth

    Args:
        nested_list:
        significant_figures:

    Returns:
        round_nestedList
    """
    rounded_nestedList = []
    for sublist in nested_list:
        if isinstance(sublist[0], list):
            rounded_sublist = round_nestedList(sublist, significant_figures)
        else:
            rounded_sublist = round_list(sublist, significant_figures)

        rounded_nestedList.append(rounded_sublist)

    return rounded_nestedList
