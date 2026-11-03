import torch
import torch.nn.functional as F


def cox_neg_log_likelihood(risk: torch.Tensor, event: torch.Tensor, time: torch.Tensor,
                           sort: bool = True) -> torch.Tensor:
    """ Cox negative log partial likelihood.
    From DeepSurv: https://doi.org/10.1186/s12874-018-0482-1

    Parameters
    ----------
    risk: torch.Tensor, of shape (..., n_samples)
        Risks estimated by model

    event: torch.Tensor, of shape (..., n_samples)
        Event indicator denoting whether time is of observed event or dropout

    time: torch.Tensor, of shape (..., n_samples)
        Time of either observed event or dropout

    sort: bool
        Whether to sort by time, otherwise assumes pre-sorted ground truth times and events

    """
    if sort:
        # Sort risk and events by time
        sort_idx = torch.argsort(time, descending=True)
        risk = torch.gather(risk, dim=-1, index=sort_idx)
        event = torch.gather(event, dim=-1, index=sort_idx)
    # Due to the sorting before, log_risk[i] = log(sum(e^risk[j=0:i]) with time[j] >= time[i]
    log_risk = torch.logcumsumexp(risk, dim=-1)
    likelihood = (risk - log_risk) * event.float()
    return - likelihood.sum(dim=-1) / event.sum(dim=-1)


def weibull_neg_log_likelihood(params: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """ Weibull negative log likelihood.
    From DeepWeiSurv: https://doi.org/10.1007/978-3-030-47426-3_53

    Parameters
    ----------
    params: torch.Tensor, of shape (..., n_samples, 2 | p * 3)
        Parameters of the mixture of p Weibull distributions, namely weights, shapes, and scales

    event: torch.Tensor, of shape (..., n_samples)
        Event indicator denoting whether time is of observed event or dropout

    time: torch.Tensor, of shape (..., n_samples)
        Time of either observed event or dropout
    """
    # Extract mixture alpha, shape, and scale from params
    if params.shape[-1] == 2 or params.shape[-1] == 3:
        n_dists = 1
        weight, shape, scale = None, params[..., -2], params[..., -1]
    elif params.shape[-1] % 3 == 0:
        n_dists = params.shape[-1] // 3
        weight, shape, scale = params[..., 0:n_dists], params[..., n_dists:2 * n_dists], params[..., 2 * n_dists:]
    else:
        raise ValueError('Unexpected number of Weibull parameters: ' + str(params.shape[-1]))

    # Guard against time=0, where log(time) = -inf, since a * event_true = nan which is later disregarded in nansum
    event_true = event & (time != 0)
    event_false = ~event
    time = torch.clamp(time, min=1e-4)

    if n_dists > 1:
        time = time.unsqueeze(-1)  # for proper broadcasting with mixture params

    a = torch.log(shape) - torch.log(scale) + (shape - 1) * (torch.log(time) - torch.log(scale))
    b = -torch.pow(torch.clamp(time / scale, max=1000), shape)
    if n_dists == 1:
        # Simplified equation for single distribution
        return -(torch.nansum(a * event_true.float(), dim=-1) + torch.sum(b, dim=-1)) / event.shape[-1]
    else:
        # Extended equation for multiple distributions
        b += weight
        return -(torch.nansum(torch.logsumexp(a + b, dim=-1) * event_true.float(), dim=-1)
                 + torch.sum(torch.logsumexp(b, dim=-1) * event_false.float(), dim=-1)
                 - torch.sum(torch.logsumexp(weight, dim=-1), dim=-1)) / event.shape[-1]


def weibull_neg_log_likelihood_original(params: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """ Weibull negative log likelihood.
    From DeepWeiSurv: https://doi.org/10.1007/978-3-030-47426-3_53

    Parameters
    ----------
    params: torch.Tensor, of shape (..., n_samples, 2 | p * 3)
        Parameters of the mixture of p Weibull distributions, namely weights, shapes, and scales

    event: torch.Tensor, of shape (..., n_samples)
        Event indicator denoting whether time is of observed event or dropout

    time: torch.Tensor, of shape (..., n_samples)
        Time of either observed event or dropout
    """
    # Extract mixture alpha, shape, and scale from params
    if params.shape[-1] % 3 == 0:
        n_dists = params.shape[-1] // 3
        alphas, shape, scale = params[..., 0:n_dists], params[..., n_dists:2 * n_dists], params[..., 2 * n_dists:]
        alphas, shape, scale = alphas.t(), shape.t(), scale.t()
    else:
        raise ValueError('Unexpected number of Weibull parameters: ' + str(params.shape[-1]))

    # https://github.com/AchrafB2015/pydpwte/blob/133aeb1004adf6bc0fd1e6985b42fc5986a77e02/pydpwte/utils/loss.py#L37-L44
    t_over_eta = torch.div(time, scale)
    h1 = torch.exp(-torch.pow(t_over_eta, shape))
    h1_bis = torch.pow(t_over_eta, shape - 1)
    params_aux = torch.div(torch.mul(alphas, shape), scale)
    return -torch.mean(event * torch.log(torch.sum(torch.mul(torch.mul(params_aux, h1_bis), h1), 0))
                       + (~event) * torch.log(torch.sum(alphas * h1, 0)))


def weibull_survival_time(params: torch.Tensor, softmax: bool = True):
    """ Survival time from mean of mixture of Weibull distributions.
    From DeepWeiSurv: https://doi.org/10.1007/978-3-030-47426-3_53

    Parameters
    ----------
    params: torch.Tensor, of shape (..., n_samples, 2 | p * 3)
        Parameters of the mixture of p Weibull distributions, namely weights, shapes, and scales

    softmax: bool
        Whether to apply a softmax transformation on the weights
    """
    # Extract mixture alpha, shape, and scale from params
    if params.shape[-1] == 2 or params.shape[-1] == 3:
        n_dists = 1
        weight, shape, scale = 1, params[..., [-2]], params[..., [-1]]
    elif params.shape[-1] % 3 == 0:
        n_dists = params.shape[-1] // 3
        weight, shape, scale = params[..., 0:n_dists], params[..., n_dists:2 * n_dists], params[..., 2 * n_dists:]
    else:
        raise ValueError('Unexpected number of Weibull parameters: ' + str(params.shape[-1]))

    # Apply softmax if needed
    if softmax and n_dists > 1:
        weight = torch.softmax(weight, dim=-1)

    # Return weighted mean as survival time estimate
    return torch.sum(weight * scale * torch.exp(torch.lgamma(1 + 1 / shape)), dim=-1)


def mse_with_pairwise_rank(estimate: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """ Extended mean squared error and pairwise ranking loss.
    From RankDeepSurv: https://doi.org/10.1016/j.artmed.2019.06.001

    """
    # First part of the loss function is a simple mean squared error
    error = time - estimate
    loss1 = torch.mean(torch.square(error) * (event | (estimate <= time)).float(), dim=-1)

    # Here, we add extra dimensions to enable pairwise comparisons of (i,j)
    event_i, event_j = event.unsqueeze(-2), event.unsqueeze(-1)
    time_i, time_j = time.unsqueeze(-2), time.unsqueeze(-1)
    # The compatibility matrix specifies which pairs (i,j) can be compared accounting for censoring
    comp = event_i & (event_j | (time_i <= time_j))  # matrix C in report

    # Second part of the loss function encourages correct ranking among compatible pairs based on relative distance
    # To save computations, we can reuse the errors needed for loss1 as
    # (time_j - time_i) - (estimate_j - estimate_i) = (time_j - estimate_j) - (time_i - estimate_i)
    error_i, error_j = error.unsqueeze(-2), error.unsqueeze(-1)
    diff = torch.clamp(error_j - error_i, min=0)  # matrix D in report, fused with condition
    loss2 = torch.sum(diff * comp, dim=(-2, -1)) / event.shape[-1]

    return loss1 + loss2


def deephit_loss(logits: torch.Tensor, event: torch.Tensor, time_idx: torch.Tensor,
                 alpha: float, sigma: float) -> torch.Tensor:
    """ DeepHit loss.
    From DeepHit: https://doi.org/10.1609/aaai.v32i1.11842

    Parameters
    ----------
    logits: torch.Tensor, of shape (..., n_samples, n_times)
        Discrete time probabilities estimated by model, represented as logits

    event: torch.Tensor, of shape (..., n_samples)
        Event indicator denoting whether time is of observed event or dropout

    time_idx: torch.Tensor, of shape (..., n_samples)
        Discrete time indices of either observed event or dropout

    alpha: float
        Weight for balancing the likelihood and ranking loss components

    sigma: float
        Bandwidth of the radial basis function in the ranking loss component
    """
    # Add extra bin to the end for "out of bounds" survival times
    logits = F.pad(logits, (0, 1), mode='constant')

    # For events based on event probability
    logits_t = logits.movedim(-1, 1) if logits.ndim > 2 else logits
    loss_event = F.cross_entropy(logits_t, time_idx, reduction='none')
    # For censoring based on survival probability
    cdf = torch.softmax(logits, dim=-1).cumsum(dim=-1)
    cdf_t = cdf.gather(dim=-1, index=time_idx.unsqueeze(-1)).squeeze(-1)
    loss_censored = -torch.log(torch.clamp(1.0 - cdf_t, min=1e-7))

    nll_loss = torch.where(event.bool(), loss_event, loss_censored).mean(dim=-1)

    # Here, we add extra dimensions to enable pairwise comparisons of (i,j)
    event_i, event_j = event.unsqueeze(-2), event.unsqueeze(-1)
    time_idx_i, time_idx_j = time_idx.unsqueeze(-2), time_idx.unsqueeze(-1)
    # The acceptability matrix specifies which (i, j) can be compared
    acc = event_i & ((time_idx_i < time_idx_j) | ((time_idx_i == time_idx_j) & ~event_j))

    cum_incidence_ti = cdf_t.unsqueeze(-2)
    cum_incidence_tij = cdf.gather(dim=-1, index=time_idx_i.expand(*cdf.shape[:-1], -1))
    rank_loss = torch.mean(torch.exp(-(cum_incidence_ti - cum_incidence_tij) / sigma) * acc, dim=(-2, -1))

    return alpha * nll_loss + (1.0 - alpha) * rank_loss
