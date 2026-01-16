from typing import Literal

import torch


def concordance_index(estimate: torch.Tensor, event: torch.Tensor, time: torch.Tensor,
                      mode: Literal['risk', 'time'] = 'risk') -> torch.Tensor:
    r""" Compute Harrell's concordance index or c-index.

    This essentially computes the ratio of correctly ordered pairs while accounting for censoring. Given estimates
    :math:`\eta`, event indicators :math:`\delta`, and times :math:`t` it is described by

    .. math::

       C = \frac{\sum_{i,j} \mathbb{1}_{\eta_i < \eta_j} \mathbb{1}_{t_i > t_j} \delta_j}
       {\sum_{i,j} \mathbb{1}_{t_i > t_j} \delta_j}.

    If your model directly predicts survival time, rather than risk, you need to negate the estimates as this function
    assumes an inverse relationship between estimates and times, i.e., if one increases the other decreases.

    Note
    ----
    While Harrell's concordance index is easy to interpret, it is known to be biased in the presence of higher amounts
    of censoring [1]_. An alternative is Uno's concordance index, as implemented in
    `sksurv.metrics.concordance_index_ipcw`.

    .. [1] H. Uno, T. Cai, M. J. Pencina, R. B. D’Agostino, and L. J. Wei, “On the C‐statistics for evaluating overall
       adequacy of risk prediction procedures with censored survival data,” Statistics in Medicine, vol. 30, no. 10, pp.
       1105–1117, Jan. 2011, doi: 10.1002/sim.4154. Available: http://dx.doi.org/10.1002/sim.4154


    Parameters
    ----------
    estimate: torch.Tensor, of shape (..., n_samples)
        Risks or times estimated by model

    event: torch.Tensor, of shape (..., n_samples)
        Event indicator denoting whether time is of observed event or dropout

    time: torch.Tensor, of shape (..., n_samples)
        Time of either observed event or dropout

    mode: Literal['risk', 'time']
        Whether the passed estimates are risks or survival times

    Returns
    -------
    torch.Tensor, of shape (...,)
        Concordance index score
    """
    if mode == 'risk':
        estimate_comp = estimate.unsqueeze(-1) < estimate.unsqueeze(-2)
    else:
        estimate_comp = estimate.unsqueeze(-1) > estimate.unsqueeze(-2)
    time_comp = time.unsqueeze(-1) > time.unsqueeze(-2)
    event = event.unsqueeze(-2)
    return torch.sum(estimate_comp & time_comp & event, (-2, -1)) / torch.sum(time_comp & event, (-2, -1))
