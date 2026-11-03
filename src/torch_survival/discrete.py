from typing import Literal

import torch


def linear_time_points(time: torch.Tensor, n_times: int):
    """ Linear time points. """
    t_max, _ = time.max(dim=-1, keepdim=True)
    return torch.linspace(0, 1, steps=n_times, device=t_max.device) * t_max


def time_to_idx(event: torch.Tensor, time: torch.Tensor, time_points: torch.Tensor,
                event_mode: Literal['floor', 'ceil'] = 'ceil', censored_mode: Literal['floor', 'ceil'] = 'floor'):
    """ Discretize continuous event/censoring times.

    Parameters
    ----------
    event: torch.Tensor, of shape (..., n_samples)
    time: torch.Tensor, of shape (..., n_samples)
    time_points: torch.Tensor, of shape (..., n_times)
    event_mode: Literal['floor', 'ceil']
    censored_mode: Literal['floor', 'ceil']

    Returns
    -------
    discrete_time: torch.Tensor, of shape (..., n_samples, n_times)
    """
    idx_event = torch.searchsorted(time_points, time, side='right') - 1 if event_mode == 'floor' \
        else torch.searchsorted(time_points, time, side='left')
    idx_censored = torch.searchsorted(time_points, time, side='right') - 1 if censored_mode == 'floor' \
        else torch.searchsorted(time_points, time, side='left')
    idx = torch.where(event.bool(), idx_event, idx_censored)
    return torch.clamp(idx, 0, time_points.size(-1) - 1)
