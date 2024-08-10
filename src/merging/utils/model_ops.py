import copy
import math
import time
from typing import Any, Callable, Dict, List

import torch
import torch.nn.functional as F


def check_parameterNamesMatch(checkpoints: list[Dict[str, torch.Tensor]]):
    """
    Check that the parameter names are the same for all checkpoints

    Args:
        checkpoints:

    Returns:

    """
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) < 2:
        return True

    for checkpoint in checkpoints[1:]:
        current_parameterNames = set(checkpoint.keys())
        if current_parameterNames != parameter_names:
            raise ValueError(
                "Differing parameter names in models. "
                f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
            )


def reduce_modelParameters(
    model_parameters: Dict[str, torch.Tensor],
    reduceValue_fn: Callable[[torch.Tensor], torch.Tensor],
    reduceCheckpoint_fn: Callable[[torch.Tensor], torch.Tensor],
):
    """
    Reduce checkpoint into a single value

    Args:
        model_parameters:
        reduceValue_fn: Function to reduce parameter block into a single value.
        reduceCheckpoint_fn: Function to reduce values from each parameter block into a single value.

    Returns:
        reduced_value
    """
    newModel_parameters = {}
    for parameter_name, parameter_values in model_parameters.items():
        newModel_parameters[parameter_name] = reduceValue_fn(parameter_values)

    return reduceCheckpoint_fn(list(newModel_parameters.values()))


def reduceAll_modelParameters(
    allModels_parameters: list[Dict[str, torch.Tensor]],
    reduce_fn: Callable[[torch.Tensor], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Reduce a list of checkpoints into a single checkpoint

    Args:
        allModels_parameters:
        reduce_fn: Takes a tensor where the first dimension iterates over checkpoints
    Returns:
        Model: dictionary
    """
    check_parameterNamesMatch(allModels_parameters)
    # Returns list of list of parameters where the outer list is the parameter names,
    # and inner list is the models.
    all_parameterValues = zip(*list(map(lambda x: x.values(), allModels_parameters)))

    # All models must have the same parameters
    all_parameterNames = allModels_parameters[0].keys()

    newModel_parameters = {}
    for parameter_name, parameter_values in zip(
        *[all_parameterNames, all_parameterValues]
    ):
        newModel_parameters[parameter_name] = reduce_fn(
            torch.stack(list(parameter_values), dim=0).contiguous()
        )
    return newModel_parameters


def efficientReduceSum_modelParameters(
    allModels_parameters: list[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Reduce a list of checkpoints into a single checkpoint

    Args:
        allModels_parameters:
        reduce_fn: Takes a tensor where the first dimension iterates over checkpoints
    Returns:
        Model: dictionary
    """
    check_parameterNamesMatch(allModels_parameters)
    # Returns list of list of parameters where the outer list is the parameter names,
    # and inner list is the models.
    all_parameterValues = zip(*list(map(lambda x: x.values(), allModels_parameters)))

    # All models must have the same parameters
    all_parameterNames = allModels_parameters[0].keys()

    newModel_parameters = {}
    for parameter_name, parameter_values in zip(
        *[all_parameterNames, all_parameterValues]
    ):
        new_parameter = None
        for parameter in parameter_values:
            if new_parameter is None:
                new_parameter = torch.clone(parameter)
            else:
                new_parameter += parameter
        newModel_parameters[parameter_name] = new_parameter
    return newModel_parameters


def pairwiseMap_modelParameters(
    modelOne_parameters: Dict[str, torch.Tensor],
    modelTwo_parameters: Dict[str, torch.Tensor],
    map_fn: Callable[[torch.Tensor], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """

    Args:
        modelOne_parameters:
        modelTwo_parameters:
        map_fn:

    Returns:
        newModel_parameters:
    """
    # All models must have the same parameters
    check_parameterNamesMatch([modelOne_parameters, modelTwo_parameters])
    all_parameterNames = modelOne_parameters.keys()

    newModel_parameters = {}
    for parameter_name in all_parameterNames:
        newModel_parameters[parameter_name] = map_fn(
            modelOne_parameters[parameter_name], modelTwo_parameters[parameter_name]
        )

    return newModel_parameters


def map_modelParameters(
    model_parameters: Dict[str, torch.Tensor],
    map_fn: Callable[[torch.Tensor], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """

    Args:
        model_parameters:
        map_fn:

    Returns:
        newModel_parameters
    """
    newModel_parameters = {}
    for parameter_name, parameter_value in model_parameters.items():
        newModel_parameters[parameter_name] = map_fn(parameter_value)
    return newModel_parameters


def elementWise_add(
    modelOne_parameters: Dict[str, torch.Tensor],
    modelTwo_parameters: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Add the parameters of two models.

    Args:
        modelOne_parameters:
        modelTwo_parameters:

    Returns:
        newModel_parameters
    """
    add_fn = lambda x, y: x + y
    newModel_parameters = pairwiseMap_modelParameters(
        modelOne_parameters, modelTwo_parameters, add_fn
    )
    return newModel_parameters


def element_wise_multiply(
    modelOne_parameters: Dict[str, torch.Tensor],
    modelTwo_parameters: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Element wise multiply the parameters.

    Args:
        modelOne_parameters:
        modelTwo_parameters:

    Returns:
        newModel_parameters
    """
    elementWiseMul = lambda x, y: torch.mul(x, y)
    newModel_parameters = pairwiseMap_modelParameters(
        modelOne_parameters, modelTwo_parameters, elementWiseMul
    )
    return newModel_parameters


def matrix_multiply(
    modelOne_parameters: Dict[str, torch.Tensor],
    modelTwo_parameters: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Matrix multiply the parameters.

    Args:
        modelOne_parameters:
        modelTwo_parameters:

    Returns:
        newModel_parameters
    """
    matmul_fn = lambda x, y: torch.matmul(x, y)
    newModel_parameters = pairwiseMap_modelParameters(
        modelOne_parameters, modelTwo_parameters, matmul_fn
    )
    return newModel_parameters


def elementWise_subtract(
    modelOne_parameters: Dict[str, torch.Tensor],
    modelTwo_parameters: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Subtract the parameters of modelTwo from the parameters of modelOne.

    Args:
        modelOne_parameters:
        modelTwo_parameters:

    Returns:
        newModel_parameters
    """
    subtract_fn = lambda x, y: x - y
    newModel_parameters = pairwiseMap_modelParameters(
        modelOne_parameters, modelTwo_parameters, subtract_fn
    )
    return newModel_parameters


def elementWise_divide(
    modelOne_parameters: Dict[str, torch.Tensor],
    modelTwo_parameters: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Subtract the parameters of modelTwo from the parameters of modelOne.

    Args:
        modelOne_parameters:
        modelTwo_parameters:

    Returns:
        newModel_parameters
    """
    divide_fn = lambda x, y: x / torch.where(y == 0, torch.ones_like(y), y)
    newModel_parameters = pairwiseMap_modelParameters(
        modelOne_parameters, modelTwo_parameters, divide_fn
    )
    return newModel_parameters


def elementWise_scale(
    model_parameters: Dict[str, torch.Tensor], scaler: float
) -> Dict[str, torch.Tensor]:
    """
    Multiply model parameters by scaler.

    Args:
        model_parameters:
        scaler:

    Returns:
        newModel_parameters
    """
    scale_fn = lambda x: x * scaler
    newModel_parameters = map_modelParameters(model_parameters, scale_fn)
    return newModel_parameters


def elementWise_scaleAndSum(
    allModels_parameters: List[Dict[str, torch.Tensor]], model_lambda: float
) -> Dict[str, torch.Tensor]:
    """
    Scale up a list of model parameters, and then sum them.

    Args:
        allModels_parameters:
        model_lambda:

    Returns:
        newModel_parameters
    """
    sum_fn = lambda parameters: torch.sum(parameters * model_lambda, dim=0)
    newModel_parameters = reduceAll_modelParameters(allModels_parameters, sum_fn)
    return newModel_parameters


def set_minimum(
    model_parameters: Dict[str, torch.Tensor], epsilon: float
) -> Dict[str, torch.Tensor]:
    """
    Set the minimum of the parameters to be epsilon. For any value less than epsilon,
    replace with epsilon.
    Note this assumes all parmeters are positive

    Args:
        model_parameters:
        epsilon: minimum value of model parameter

    Returns:
        new_modelParameters
    """
    new_modelParameters = {}
    for parameter_name, parameter in model_parameters.items():
        new_parameter = parameter.clone()
        new_parameter[new_parameter < epsilon] = epsilon
        new_modelParameters[parameter_name] = new_parameter
    return new_modelParameters


def elementWise_inverse(
    model_parameters: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Element-wise inverse which is equal to 1 over

    Args:
        model_parameters:

    Returns:
        new_modelParameters
    """
    new_modelParameters = map_modelParameters(model_parameters, lambda x: 1 / x)
    return new_modelParameters


def dropout(
    model_parameters: Dict[str, torch.Tensor], dropout_probability
) -> Dict[str, torch.Tensor]:
    new_modelParameters = map_modelParameters(
        model_parameters, lambda x: F.dropout(x, dropout_probability)
    )

    return new_modelParameters


def matrix_inverse(
    model_parameters: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Does a matrix inverse for each parameter block (assumes each block is square )

    Args:
        model_parameters:

    Returns:
        new_modelParameters
    """
    matrixInverse_fn = lambda x: torch.linalg.inv(x)
    new_modelParameters = {}
    for parameter_name, parameter in model_parameters.items():
        try:
            original_device = parameter.device
            matrix_inverse = matrixInverse_fn(parameter.cuda()).to(original_device)
            new_modelParameters[parameter_name] = matrix_inverse
        except:
            # If matrix is not invertible because row/col is 0, then we remove the row/col with 0 and invert the submatrix
            # We then insert the row/col with 0 back afterwards
            nonZero_rowIdx = (torch.sum(parameter, dim=1) != 0).nonzero().squeeze()
            nonZero_colIdx = (torch.sum(parameter, dim=0) != 0).nonzero().squeeze()
            assert (nonZero_colIdx == nonZero_rowIdx).all()
            num_row = parameter.shape[0]
            nonZero_broadcastColIdx = nonZero_colIdx[None, :].repeat((num_row, 1))
            nonZero_broadcastRowIdx = nonZero_rowIdx[:, None].repeat(
                (1, nonZero_broadcastColIdx.shape[1])
            )

            # Get submatrix that is full rank
            fullRank_parameter = torch.gather(parameter, 1, nonZero_broadcastColIdx)
            fullRank_parameter = torch.gather(
                fullRank_parameter, 0, nonZero_broadcastRowIdx
            )

            # Invert submatrix that is full rank
            inverse_fullRankParameter = matrixInverse_fn(fullRank_parameter)
            inverse_parameter = copy.deepcopy(parameter)
            inverse_parameter[nonZero_rowIdx[:, None], nonZero_colIdx] = (
                inverse_fullRankParameter
            )
            new_modelParameters[parameter_name] = inverse_parameter
    return new_modelParameters
