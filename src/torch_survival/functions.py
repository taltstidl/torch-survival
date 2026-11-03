from typing import Literal

import numpy as np
from scipy.special import logsumexp

class WeibullMixtureFunction:
    def __init__(self, weights, shapes, scales, horizon=1, kind: Literal['survival', 'cumulative_hazard']='survival'):
        self.w = np.asarray(weights, float)
        self.k = np.asarray(shapes, float)
        self.lam = np.asarray(scales, float)
        self.horizon = horizon
        self.kind = kind

    def __call__(self, t):
        t = np.asarray(t, float)[..., None] / self.horizon
        log_s = logsumexp(-(t / self.lam) ** self.k, b=self.w, axis=-1)
        return np.exp(log_s) if self.kind == 'survival' else -log_s
