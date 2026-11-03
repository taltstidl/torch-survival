import functools
import warnings

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.validation import check_is_fitted, validate_data
from sksurv.base import SurvivalAnalysisMixin
from sksurv.functions import StepFunction
from sksurv.util import check_array_survival
from torch.utils.data import TensorDataset, DataLoader

from torch_survival.discrete import time_to_idx, linear_time_points
from torch_survival.earlystop import EarlyStopper
from torch_survival.losses import deephit_loss
from torch_survival.metrics import c_index_antolini
from torch_survival.progress import OptunaProgressCallback
from torch_survival.sample import OptunaSampler, TunedTopology, TunedCategorical, TunedFloat, TunedInt


class DeepHit(SurvivalAnalysisMixin, BaseEstimator):
    r""" Implements the DeepHit model presented by Lee et al. [1]_.

    Uses a deep neural network trained with the negative log likelihood of the survival time probability distribution,
    as estimated over discrete time intervals, combined with a ranking loss.

    .. [1] C. Lee, W. Zame, J. Yoon, and M. Van der Schaar, “DeepHit: A Deep Learning Approach to Survival Analysis
    With Competing Risks,” AAAI, vol. 32, no. 1, Apr. 2018, doi: 10.1609/aaai.v32i1.11842. Available:
    http://dx.doi.org/10.1609/aaai.v32i1.11842
    """

    def __init__(
            self,
            hidden_layer_sizes=TunedTopology(n_layers=4, n_neurons=50),
            activation=TunedCategorical(choices=['relu', 'selu']),
            dropout=TunedFloat(low=0.0, high=0.5),
            optimizer=TunedCategorical(choices=['sgd', 'adam']),
            learning_rate=TunedFloat(low=1e-5, high=1e-2, log=True),
            momentum=TunedFloat(low=0.8, high=0.95),
            scheduler='inverse_time',
            decay=0.0,
            alpha=TunedFloat(low=0.0, high=1.0),
            sigma=TunedFloat(low=1e-1, high=1e1, log=True),
            n_times=TunedInt(low=10, high=500),
            batch_size=256,
            n_epochs=500,
            early_stopping=True,
            patience=10,
            validation_fraction=0.2,
            n_trials=50,
            random_state=None,
            device=None,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.dropout = dropout
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.scheduler = scheduler
        self.decay = decay
        self.alpha = alpha
        self.sigma = sigma
        self.n_times = n_times
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.n_trials = n_trials
        self.random_state = random_state
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _optimize(self, trial: optuna.Trial, X, y_event, y_time):
        # Do 5-fold cross validation
        scores = []
        for train_idx, test_idx in (StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                .split(X, y_event.cpu())):
            model, time_points = self._train(trial, X[train_idx], y_event[train_idx], y_time[train_idx])
            model.eval()
            surv_logits = model(X[test_idx])
            surv_prob = DeepHit.logits_to_prob(surv_logits)
            y_time_idx = time_to_idx(y_event[test_idx], y_time[test_idx], time_points, event_mode='floor')
            c_index = c_index_antolini(surv_prob, y_event[test_idx], y_time[test_idx], y_time_idx)
            scores.append(c_index)
        return sum(scores) / len(scores)

    def _train(self, trial: optuna.Trial, X, y_event, y_time):
        sampler = OptunaSampler(trial)

        # Set up model, optimizer, and scheduler
        n_inputs, n_outputs = X.shape[-1], sampler.sample_int('n_times', self.n_times)
        model = sampler.sample_network(n_inputs, n_outputs, self.hidden_layer_sizes, self.activation, self.dropout)
        optimizer, scheduler = sampler.sample_optimizer(model, self.optimizer, self.learning_rate, self.momentum,
                                                        self.scheduler, self.decay)

        # Split into training and validation for early stopping
        if self.early_stopping:
            X, X_val, y_event, y_event_val, y_time, y_time_val = train_test_split(
                X, y_event, y_time, test_size=self.validation_fraction, stratify=y_event.cpu(),
                random_state=self.random_state)
            early_stopper = EarlyStopper(patience=self.patience)

        # Discretize event times
        time_points = linear_time_points(y_time, n_outputs)
        y_time_idx = time_to_idx(y_event, y_time, time_points)
        if self.early_stopping:
            y_time_idx_val = time_to_idx(y_event_val, y_time_val, time_points, event_mode='floor')

        # Sample batch size and loss parameters
        batch_size = sampler.sample_int('batch_size', self.batch_size)
        alpha = sampler.sample_float('alpha', self.alpha)
        sigma = sampler.sample_float('sigma', self.sigma)

        # Create data loaders
        train_dataset = TensorDataset(X, y_event, y_time_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if self.early_stopping:
            val_dataset = TensorDataset(X_val, y_event_val, y_time_idx_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Train and return model
        model.to(self.device)
        for i in range(self.n_epochs):
            for X, y_event, y_time_idx in train_loader:
                optimizer.zero_grad()
                surv_logits = model(X)
                loss = deephit_loss(surv_logits, y_event, y_time_idx, alpha, sigma)
                loss.backward()
                optimizer.step()
            if scheduler:
                scheduler.step()
            if self.early_stopping:
                model.eval()
                with torch.no_grad():
                    val_loss = torch.tensor(0., device=self.device)
                    for X_val, y_event_val, y_time_idx_val in val_loader:
                        surv_logits = model(X_val)
                        loss = deephit_loss(surv_logits, y_event_val, y_time_idx_val, alpha, sigma)
                        val_loss += loss * X_val.shape[0]
                    if early_stopper.should_stop(model, val_loss):
                        break
                model.train()
        if self.early_stopping:
            early_stopper.restore_best_weights(model)
        return model, time_points

    def fit(self, X, y):
        """ Fit the model to the given survival data.

        Parameters
        ----------
        X: array-like, shape = (n_samples, n_features)
            Data matrix.
        y: structured array, shape = (n_samples,)
            A structured array with two fields. The first field is a boolean where ``True`` indicates an event and
            ``False`` indicates right-censoring. The second field is a float with the time of event or time of
            censoring.

        Returns
        -------
        self
        """
        # Validate and extract data
        X, y = validate_data(self, X, y)
        y_event, y_time = check_array_survival(X, y)

        # Convert data to tensors
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_event = torch.as_tensor(y_event, dtype=torch.bool, device=self.device)
        y_time = torch.as_tensor(y_time.copy(), dtype=torch.float32, device=self.device)

        # Seed PyTorch random number generator
        has_random_state = self.random_state is not None
        if has_random_state:
            torch.manual_seed(self.random_state)

        # Optimize hyperparameters
        optuna.logging.disable_default_handler()
        warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
        with OptunaProgressCallback(model_name='DeepHit', n_trials=self.n_trials) as callback:
            study = optuna.create_study(sampler=TPESampler(seed=self.random_state) if has_random_state else None,
                                        direction='maximize')
            objective = functools.partial(self._optimize, X=X, y_event=y_event, y_time=y_time)
            study.optimize(objective, n_trials=self.n_trials, callbacks=[callback])

        # Train final model with best hyperparameters
        self.model_, self.time_points_ = self._train(study.best_trial, X, y_event, y_time)
        self.model_.eval()

        # Override internal parameters with tuned hyperparameters
        best_params = study.best_params
        if 'n_layers' in best_params:
            best_params['hidden_layer_sizes'] = [best_params.pop('n_neurons_' + str(i + 1)) for i in
                                                 range(best_params.pop('n_layers'))]
        self.set_params(**best_params)

        return self

    def predict_cumulative_hazard_function(self, X, return_array=False):
        r"""Predict cumulative hazard function.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.
        return_array: bool, default=False
            Determines whether to return a 2D array of cumulative hazard values of shape `(n_samples, n_unique_times)`
            (if `True`) or a 1D array of :class:`sksurv.functions.StepFunction` objects (if `False`).

        Returns
        -------
        cum_hazard: ndarray, shape = (n_samples, n_unique_times) or (n_samples,)
            If `return_array` is False, array of `n_samples` :class:`sksurv.functions.StepFunction` objects.
            If `return_array` is True, numeric array of shape `(n_samples, n_unique_times)`, where
            `n_unique_times` is the number of unique event times in the training data.
        """
        check_is_fitted(self)
        X = validate_data(self, X)
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        surv_logits = F.pad(self.model_(X).detach(), (0, 1))
        total_lse = torch.logsumexp(surv_logits, dim=-1, keepdim=True)
        # Need to flip here as we need sum over times k+1:K
        times_lse = torch.flip(torch.logcumsumexp(torch.flip(surv_logits, dims=[-1]), dim=-1), dims=[-1])
        cum_hazard = total_lse - times_lse[..., 1:]
        cum_hazard = cum_hazard.cpu().numpy()
        if return_array:
            return cum_hazard
        n_samples = X.shape[0]
        x = self.time_points_.cpu().numpy()
        funcs = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            funcs[i] = StepFunction(x=x, y=cum_hazard[i])
        return funcs

    @torch.no_grad()
    def predict_survival_function(self, X, return_array=False):
        r"""Predict survival function.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.
        return_array: bool, default=False
            Determines whether to return a 2D array of survival probabilities of shape `(n_samples, n_unique_times)`
            (if `True`) or a 1D array of :class:`sksurv.functions.StepFunction` objects (if `False`).

        Returns
        -------
        survival_prob: ndarray, shape = (n_samples, n_unique_times) or (n_samples,)
            If `return_array` is False, array of `n_samples` :class:`sksurv.functions.StepFunction` objects.
            If `return_array` is True, numeric array of shape `(n_samples, n_unique_times)`, where
            `n_unique_times` is the number of unique event times in the training data.
        """
        check_is_fitted(self)
        X = validate_data(self, X)
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        surv_logits = F.pad(self.model_(X).detach(), (0, 1))
        surv_prob = 1.0 - torch.softmax(surv_logits, dim=-1).cumsum(dim=-1)[..., :-1]
        surv_prob = surv_prob.cpu().numpy()
        if return_array:
            return surv_prob  # just the survival probabilities at the discrete time points
        n_samples = X.shape[0]
        x = self.time_points_.cpu().numpy()
        funcs = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            funcs[i] = StepFunction(x=x, y=surv_prob[i])
        return funcs

    def save(self, path):
        """ Save a trained model to disk.

        This saves the model's parameters (dictionary) and weights (tensors) such that the estimator can be fully
        restored using :meth:`DeepHit.load`. Internally relies on :meth:`torch.save`.

        Parameters
        ----------
        path: str or file-like object or os.PathLike
            Path at which to save the model.
        """
        check_is_fitted(self)
        state = {
            'params': self.get_params(),
            'model': self.model_.state_dict(),
            'time_points': self.time_points_,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path, device=None):
        """ Reload an already trained model.

        Parameters
        ----------
        path: str or file-like object or os.PathLike
            Path from which to load the model. See also :meth:`DeepSurv.save` for additional details.
        device: str or torch.device, default=None
            Device on which tensors will be allocated. If None, uses CUDA if available, else CPU.

        Returns
        -------
        model: DeepHit
            The trained estimator.
        """
        # Note: Device handled manually as users may wish to use a device different from the original when restoring
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path, map_location=device)
        # Reload estimator with initial parameters
        params = state['params']
        params['device'] = device
        estimator = cls(**params)
        # Restore internal model based on weights
        sampler = OptunaSampler(None)
        n_inputs, n_outputs = next(iter(state['model'].values())).shape[-1], state['params']['n_times']
        estimator.model_ = sampler.sample_network(n_inputs, n_outputs, estimator.hidden_layer_sizes,
                                                  estimator.activation, estimator.dropout)
        estimator.model_.load_state_dict(state['model'])
        estimator.model_.to(device)
        estimator.model_.eval()
        # Restore internal time points for discretization
        estimator.time_points_ = state['time_points']
        return estimator

    @staticmethod
    def logits_to_prob(logits):
        return 1.0 - torch.softmax(F.pad(logits, (0, 1)), dim=-1).cumsum(dim=-1)[..., :-1]
