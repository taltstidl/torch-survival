import inspect
from typing import TypeVar

import torch.nn as nn

_TypedDict = TypeVar('_TypedDict')


def make_activation(query: str):
    """
    Resolves the name of an activation function to its PyTorch layer. This method uses inspection and should thus be
    able to resolve any current PyTorch activation function, e.g., `relu` gets mapped to torch.nn.ReLU, and so forth.

    Parameters
    ----------
    query: str
        Name of activation function that should be created

    Returns
    -------
    torch.nn.Module
        PyTorch layer corresponding to query, initialized with default values
    """
    query = query.lower()
    matches = [obj for name, obj in inspect.getmembers(nn) if query == name.lower()]
    if len(matches) == 0:
        raise ValueError('Found no candidate for `{}` activation'.format(query))
    if len(matches) > 1:
        raise ValueError('Found multiple candidates for `{}` activation'.format(query))
    return matches[0]()


def merge_configs(default_config: _TypedDict, user_config: _TypedDict) -> _TypedDict:
    """
    Recursively merges default configuration with user configuration.

    Parameters
    ----------
    default_config: subclass of TypedDict
        Default model configuration
    user_config: subclass of TypedDict
        User-provided model configuration overriding defaults

    Returns
    -------
    subclass of TypedDict
        Merged model configuration
    """
    for k, v in user_config.items():
        if k in default_config:
            if isinstance(default_config[k], dict) and isinstance(v, dict):
                default_config[k] = merge_configs(default_config[k], v)
            else:
                default_config[k] = v
    return default_config

# def sample_params(search_space, trial: optuna.Trial):
#     """
#     Samples hyperparameter search space, allowing for fixed
#
#     Parameters
#     ----------
#     search_space: Mapping[str, ParamConfig]
#         Dictionary of parameter configurations.
#     trial: optuna.Trial
#         Active trial for sampling parameters.
#
#     Returns
#     -------
#     params: dict[str, int | float | str]
#         Sampled parameters.
#     """
#     params = {}
#     for key, value in search_space.items():
#         if type(value) in [int, float, str]:
#             # The value is already fixed and will be passed on as-is
#             params[key] = value
#         elif type(value) is list:
#             # The value is a list of possible categories
#             params[key] = trial.suggest_categorical(key, value)
#         elif type(value) is tuple and type(value[0]) is int:
#             # The value are lower and upper bounds for an integer
#             params[key] = trial.suggest_int(key, *value[:2], log=value[2] if len(value) > 2 else False)
#         elif type(value) is tuple and type(value[0]) is float:
#             # The value are lower and upper bounds for an integer
#             params[key] = trial.suggest_float(key, *value[:2], log=value[2] if len(value) > 2 else False)
#     return params
