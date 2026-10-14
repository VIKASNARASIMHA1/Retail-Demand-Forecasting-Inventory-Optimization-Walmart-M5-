"""Conformalized quantile regression (CQR) on the outer quantile pair (e.g. P10-P90).

Calibrate on the PREVIOUS backtest fold (out-of-sample forecasts of an earlier model), then widen or
narrow the next fold's interval by a scale-normalised correction Q. Targets coverage 1 - alpha.
Strict guarantees need exchangeability, which time series only approximately satisfy.
"""
import numpy as np


def scale_vec(Yh, window=28):
    return Yh[:, -window:].mean(1) + 1.0          # per-series scale so scores are comparable


def calibrate(actual, q, scale, alpha):
    lo, hi = q[..., 0], q[..., -1]
    s = (np.maximum(lo - actual, actual - hi) / scale[:, None]).ravel()
    n = len(s)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, level, method="higher"))


def apply(q, Q, scale):
    q2 = q.copy()
    adj = Q * scale[:, None]
    q2[..., 0] = np.maximum(q[..., 0] - adj, 0)
    q2[..., -1] = q[..., -1] + adj
    return np.sort(q2, axis=2)
