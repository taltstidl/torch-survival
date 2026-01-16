from typing import Mapping, Any

import optuna
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch import nn

from torch_survival.config import NetworkConfig, OptimizerConfig
from torch_survival.utils import make_activation


class SimpleNeuralNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs, layers, activation, dropout):
        super().__init__()
        hidden = []
        n_nodes = n_inputs
        for nodes in layers:
            hidden.append(nn.Linear(n_nodes, nodes))
            hidden.append(make_activation(activation))
            hidden.append(nn.Dropout(p=dropout))
            n_nodes = nodes
        self.hidden = nn.Sequential(*hidden)
        self.output = nn.Linear(n_nodes, n_outputs)

    def forward(self, x):
        x = self.hidden(x)
        return self.output(x)


def sample_network(trial: optuna.Trial, config: NetworkConfig, n_inputs: int, n_outputs: int):
    """
    Sample a simple neural network using the provided configuration.

    Parameters
    ----------
    trial: optuna.Trial
        Active or fixed trial used to sample hyperparameters. Final configuration may be obtained by `study.best_trial`.
    config: config.NetworkConfig
        Network configuration specifying architecture and search constraints.
    n_inputs: int
        Number of inputs of the neural network. Typically depends on the data.
    n_outputs: int
        Number of outputs of the neural network. Typically depends on the modeling technique.

    Returns
    -------
    nn.Module
        PyTorch model consisting of linear layers.
    """
    layers = config['layers']
    if not isinstance(layers, list):
        n_layers = trial.suggest_int('layers', 0, layers['max_layers'])
        layers = [trial.suggest_int('nodes_' + str(i + 1), 1, layers['max_nodes_per_layer']) for i in range(n_layers)]
    activation = config['activation']
    if not isinstance(activation, str):
        activation = trial.suggest_categorical('activation', activation)
    dropout = config['dropout']
    if not isinstance(dropout, float):
        dropout = trial.suggest_float('dropout', low=dropout[0], high=dropout[1])
    return SimpleNeuralNetwork(n_inputs, n_outputs, layers, activation, dropout)


def sample_optimizer(trial: optuna.Trial, config: OptimizerConfig, model: nn.Module):
    """
    Sample optimizer and scheduler for training using the provided configuration.

    Parameters
    ----------
    trial: optuna.Trial
        Active or fixed trial used to sample hyperparameters. Final configuration may be obtained by `study.best_trial`.
    config: config.OptimizerConfig
        Optimizer configuration specifying how the neural network should be trained.
    model: nn.Module
        Neural network whose parameters should be optimized.

    Returns
    -------
    torch.optim.Optimizer
        PyTorch optimizer that can be used for training.
    torch.optim.lr_scheduler.LRScheduler or None
        PyTorch scheduler that can be used to update learning rates.
    """
    # Sample optimizer parameters
    lr = config['lr']
    if not isinstance(lr, float):
        lr = trial.suggest_float('lr', *lr, log=True)
    momentum = config['momentum']
    if not isinstance(momentum, float):
        momentum = trial.suggest_float('momentum', *momentum)
    # Initialize optimizer
    optimizer = None
    optimizer_name = config['name']
    if not isinstance(optimizer_name, str):
        optimizer_name = trial.suggest_categorical('optimizer', optimizer_name)
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(momentum, 0.999))
    if optimizer is None:
        raise ValueError('Optimizer with name `{}` is not supported'.format(config['name']))
    # Sample optimizer parameters
    decay = config['decay']
    if not isinstance(decay, float):
        decay = trial.suggest_float('decay', *decay)
    # Initialize scheduler
    scheduler = None
    scheduler_name = config['scheduler']
    if not isinstance(scheduler_name, str):
        scheduler_name = trial.suggest_categorical('scheduler', scheduler_name)
    if scheduler_name == 'inverse_time':
        scheduler = sched.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + epoch * decay))
    # Return both
    return optimizer, scheduler


def sample_int(trial: optuna.Trial, config: Mapping[str, Any], key: str) -> int:
    """
    Helper to sample an integer value from a search space configuration.

    Parameters
    ----------
    trial: optuna.Trial
        Active or fixed trial used to sample hyperparameters. Final configuration may be obtained by `study.best_trial`.
    config: Mapping[str, Any]
        Search space configuration dictionary.
    key: str
        Name of configuration parameter that should be sampled.

    Returns
    -------
    int:
        The sampled value.
    """
    value = config[key]
    if not isinstance(value, int):
        value = trial.suggest_int(key, *value)
    return value


def sample_float(trial: optuna.Trial, config: Mapping[str, Any], key: str) -> float:
    """
    Helper to sample a floating point value from a search space configuration.

    Parameters
    ----------
    trial: optuna.Trial
        Active or fixed trial used to sample hyperparameters. Final configuration may be obtained by `study.best_trial`.
    config: Mapping[str, Any]
        Search space configuration dictionary.
    key: str
        Name of configuration parameter that should be sampled.

    Returns
    -------
    float:
        The sampled value.
    """
    value = config[key]
    if not isinstance(value, float):
        value = trial.suggest_float(key, *value)
    return value
