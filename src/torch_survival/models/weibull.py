import functools
import warnings

import numpy as np
import optuna
import torch
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.validation import check_is_fitted, validate_data
from sksurv.base import SurvivalAnalysisMixin
from sksurv.util import check_array_survival
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torch_survival.earlystop import EarlyStopper
from torch_survival.functions import WeibullMixtureFunction
from torch_survival.losses import weibull_neg_log_likelihood, weibull_survival_time
from torch_survival.metrics import concordance_index
from torch_survival.progress import OptunaProgressCallback
from torch_survival.sample import OptunaSampler, TunedTopology, TunedCategorical, TunedFloat, TunedInt


class SparsityLayer(nn.Module):
    def __init__(self, n_dists):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n_dists))
        torch.nn.init.uniform_(self.weight, 0, 1)

    def omega(self):
        w = self.weight.abs()
        return w / w.sum()

    def normalize_weights(self):
        with torch.no_grad():
            self.weight.abs_()
            self.weight /= self.weight.sum()

    def forward(self, alphas):
        return alphas + torch.log(self.omega().clamp_min(1e-12))

    def penalty(self):
        return torch.sqrt(self.weight.abs().clamp_min(1e-12)).sum()


class WeibullNetwork(nn.Module):
    def __init__(self, backbone, n_feats, n_dists, sparse=False):
        super().__init__()
        self.backbone = backbone
        self.shape_layer = nn.Linear(n_feats, n_dists)
        self.scale_layer = nn.Linear(n_feats, n_dists)
        self.weights_layer = None
        if n_dists > 1:
            self.weights_layer = nn.Linear(n_feats, n_dists)
        self.sparsity_layer = None
        if sparse:
            self.sparsity_layer = SparsityLayer(n_dists)

    def normalize_weights(self):
        if self.sparsity_layer:
            self.sparsity_layer.normalize_weights()

    def forward(self, x):
        x = self.backbone(x)
        shape = torch.nn.functional.elu(self.shape_layer(x)) + 2
        scale = torch.nn.functional.elu(self.scale_layer(x)) + 1 + 1e-4
        if not self.weights_layer:  # single Weibull distribution
            return torch.cat((shape, scale), dim=-1)
        else:  # multiple Weibull distributions
            weights = self.weights_layer(x)
            if self.sparsity_layer:
                weights = self.sparsity_layer(weights)
            return torch.cat((weights, shape, scale), dim=-1)


class WeibullBase(SurvivalAnalysisMixin, BaseEstimator):
    _predict_risk_score = False
    _sparse = False

    def __init__(
            self,
            hidden_layer_sizes,
            activation,
            dropout,
            optimizer,
            learning_rate,
            momentum,
            scheduler,
            decay,
            lamb,
            n_dists,
            batch_size,
            n_epochs,
            early_stopping,
            patience,
            validation_fraction,
            n_trials,
            random_state,
            device,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.dropout = dropout
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.scheduler = scheduler
        self.decay = decay
        self.lamb = lamb
        self.n_dists = n_dists
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
            model, time_horizon = self._train(trial, X[train_idx], y_event[train_idx], y_time[train_idx])
            model.eval()
            params = model(X[test_idx])
            times = time_horizon * weibull_survival_time(params)
            c_index = concordance_index(times, y_event[test_idx], y_time[test_idx], mode='time')
            scores.append(c_index)
        return sum(scores) / len(scores)

    def _train(self, trial: optuna.Trial, X, y_event, y_time):
        sampler = OptunaSampler(trial)

        # Sample number of distributions (or maximum if sparse)
        n_dists = sampler.sample_int('n_dists', self.n_dists)

        # Set up model, optimizer, and scheduler
        n_inputs, n_features = X.shape[-1], 16
        backbone = sampler.sample_network(n_inputs, n_features, self.hidden_layer_sizes, self.activation, self.dropout)
        model = WeibullNetwork(backbone, n_features, n_dists, self._sparse)
        optimizer, scheduler = sampler.sample_optimizer(model, self.optimizer, self.learning_rate, self.momentum,
                                                        self.scheduler, self.decay)

        # Split into training and validation for early stopping
        if self.early_stopping:
            X, X_val, y_event, y_event_val, y_time, y_time_val = train_test_split(
                X, y_event, y_time, test_size=self.validation_fraction, stratify=y_event.cpu(),
                random_state=self.random_state)
            early_stopper = EarlyStopper(patience=self.patience)

        # Normalize event times
        time_horizon = y_time.max().item()
        y_time /= time_horizon
        if self.early_stopping:
            y_time_val /= time_horizon

        # Sample batch size and loss parameters
        batch_size = sampler.sample_int('batch_size', self.batch_size)
        if self._sparse:
            lamb = sampler.sample_float('lamb', self.lamb)

        # Create data loaders
        train_dataset = TensorDataset(X, y_event, y_time)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if self.early_stopping:
            val_dataset = TensorDataset(X_val, y_event_val, y_time_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Train and return model
        model.to(self.device)
        for i in range(self.n_epochs):
            for X, y_event, y_time in train_loader:
                optimizer.zero_grad()
                params = model(X).squeeze(-1)
                loss = weibull_neg_log_likelihood(params, y_event, y_time)
                if self._sparse:
                    # Regularize the weights to encourage sparsity
                    loss += lamb * model.sparsity_layer.penalty()
                loss.backward()
                optimizer.step()
                model.normalize_weights()
            if scheduler:
                scheduler.step()
            if self.early_stopping:
                model.eval()
                with (torch.no_grad()):
                    val_loss = torch.tensor(0., device=self.device)
                    for X_val, y_event_val, y_time_val in val_loader:
                        params = model(X_val)
                        loss = weibull_neg_log_likelihood(params, y_event_val, y_time_val)
                        val_loss += loss * X_val.shape[0]
                    if early_stopper.should_stop(model, val_loss):
                        break
                model.train()
        if self.early_stopping:
            early_stopper.restore_best_weights(model)
        return model, time_horizon

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
        with OptunaProgressCallback(model_name='DeepWeiSurv', n_trials=self.n_trials) as callback:
            study = optuna.create_study(sampler=TPESampler(seed=self.random_state) if has_random_state else None,
                                        direction='maximize')
            objective = functools.partial(self._optimize, X=X, y_event=y_event, y_time=y_time)
            study.optimize(objective, n_trials=self.n_trials, callbacks=[callback])

        # Train model
        self.model_, self.time_horizon_ = self._train(study.best_trial, X, y_event, y_time)
        self.model_.eval()

        # Override internal parameters with tuned hyperparameters
        best_params = study.best_params
        if 'n_layers' in best_params:
            best_params['hidden_layer_sizes'] = [best_params.pop('n_neurons_' + str(i + 1)) for i in
                                                 range(best_params.pop('n_layers'))]
        self.set_params(**best_params)

        return self

    @torch.no_grad()
    def predict(self, X):
        """ Predict survival times.

        The survival time is estimated based on the mean of the mixture of Weibull distributions predicted by the
        neural network.

        Parameters
        ----------
        X: array-like, shape = (n_samples, n_features)
            Data matrix.

        Returns
        -------
        survival_time: array, shape = (n_samples,)
            Predicted survival times.
        """
        check_is_fitted(self)
        X = validate_data(self, X)
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        params = self.model_(X).detach()
        times = self.time_horizon_ * weibull_survival_time(params)
        return times.cpu().numpy()

    @torch.no_grad()
    def predict_cumulative_hazard_function(self, X):
        r"""Predict cumulative hazard function.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        Returns
        -------
        cum_hazard: ndarray, shape = (n_samples,)
            Array of `n_samples` :class:`torch_survival.functions.WeibullMixtureFunction` objects.
        """
        check_is_fitted(self)
        X = validate_data(self, X)
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        params = self.model_(X).detach()
        weights, shapes, scales = WeibullBase.extract_params(params)
        n_samples = X.shape[0]
        funcs = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            p = weights[i], shapes[i], scales[i]
            funcs[i] = WeibullMixtureFunction(*p, self.time_horizon_, kind='cumulative_hazard')
        return funcs

    @torch.no_grad()
    def predict_survival_function(self, X):
        r"""Predict survival function.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        Returns
        -------
        survival_prob: ndarray, shape = (n_samples,)
            Array of `n_samples` :class:`torch_survival.functions.WeibullMixtureFunction` objects.
        """
        check_is_fitted(self)
        X = validate_data(self, X)
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        params = self.model_(X).detach()
        weights, shapes, scales = WeibullBase.extract_params(params)
        n_samples = X.shape[0]
        funcs = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            p = weights[i], shapes[i], scales[i]
            funcs[i] = WeibullMixtureFunction(*p, self.time_horizon_, kind='survival')
        return funcs

    def save(self, path):
        """ Save a trained model to disk.

        This saves the model's parameters (dictionary) and weights (tensors) such that the estimator can be fully
        restored using :meth:`WeibullBase.load`. Internally relies on :meth:`torch.save`.

        Parameters
        ----------
        path: str or file-like object or os.PathLike
            Path at which to save the model.
        """
        check_is_fitted(self)
        state = {
            'params': self.get_params(),
            'model': self.model_.state_dict(),
            'time_horizon': self.time_horizon_,
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
        model: WeibullBase
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
        n_inputs, n_features = next(iter(state['model'].values())).shape[-1], 16
        backbone = sampler.sample_network(n_inputs, n_features, estimator.hidden_layer_sizes,
                                          estimator.activation, estimator.dropout)
        estimator.model_ = WeibullNetwork(backbone, n_features, state['params']['n_dists'], cls._sparse)
        estimator.model_.load_state_dict(state['model'])
        estimator.model_.to(device)
        estimator.model_.eval()
        # Restore time horizon for normalization
        estimator.time_horizon_ = state['time_horizon']
        return estimator

    @staticmethod
    def extract_params(params: torch.Tensor):
        # Extract mixture alpha, shape, and scale from params
        if params.shape[-1] == 2 or params.shape[-1] == 3:
            weight, shape, scale = torch.ones_like(params[..., [-1]]), params[..., [-2]], params[..., [-1]]
        elif params.shape[-1] % 3 == 0:
            n_dists = params.shape[-1] // 3
            weight, shape, scale = params[..., 0:n_dists], params[..., n_dists:2 * n_dists], params[..., 2 * n_dists:]
            weight = torch.softmax(weight.double(), dim=-1)
        else:
            raise ValueError('Unexpected number of Weibull parameters: ' + str(params.shape[-1]))
        return weight.cpu().numpy(), shape.cpu().numpy(), scale.cpu().numpy()


class DeepWeiSurv(WeibullBase):
    r""" Implements the DeepWeiSurv model presented by Bennis et al. [1]_.

    Uses a deep neural network trained with the negative log likelihood of the survival time probability distribution,
    as estimated by the parameters of a mixture of Weibull distributions. This implementation tries to stay faithful
    to the original paper, with the following deviations:

    .. [1] A. Bennis, S. Mouysset, and M. Serrurier, “Estimation of Conditional Mixture Weibull Distribution with Right
       Censored Data Using Neural Network for Time-to-Event Analysis,” Lecture Notes in Computer Science. Springer
       International Publishing, pp. 687–698, 2020. doi: 10.1007/978-3-030-47426-3_53. Available:
       http://dx.doi.org/10.1007/978-3-030-47426-3_53
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
            n_dists=TunedInt(low=1, high=10),
            batch_size=256,
            n_epochs=500,
            early_stopping=True,
            patience=10,
            validation_fraction=0.2,
            n_trials=50,
            random_state=None,
            device=None,
    ):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            dropout=dropout,
            optimizer=optimizer,
            learning_rate=learning_rate,
            momentum=momentum,
            scheduler=scheduler,
            decay=decay,
            lamb=1e-4,  # not used
            n_dists=n_dists,
            batch_size=batch_size,
            n_epochs=n_epochs,
            early_stopping=early_stopping,
            patience=patience,
            validation_fraction=validation_fraction,
            n_trials=n_trials,
            random_state=random_state,
            device=device,
        )


class DPWTE(WeibullBase):
    r""" Implements the DPWTE model presented by Bennis et al. [1]_.

    Uses a deep neural network trained with the negative log likelihood of the survival time probability distribution,
    as estimated by the parameters of a mixture of Weibull distributions. This implementation tries to stay faithful
    to the original paper, with the following deviations:

    .. [1]
    """
    _sparse = True

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
            lamb=TunedFloat(low=1e-5, high=1e-3, log=True),
            n_dists_max=10,
            batch_size=256,
            n_epochs=500,
            early_stopping=True,
            patience=10,
            validation_fraction=0.2,
            n_trials=50,
            random_state=None,
            device=None,
    ):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            dropout=dropout,
            optimizer=optimizer,
            learning_rate=learning_rate,
            momentum=momentum,
            scheduler=scheduler,
            decay=decay,
            lamb=lamb,
            n_dists=n_dists_max,
            batch_size=batch_size,
            n_epochs=n_epochs,
            early_stopping=early_stopping,
            patience=patience,
            validation_fraction=validation_fraction,
            n_trials=n_trials,
            random_state=random_state,
            device=device,
        )
