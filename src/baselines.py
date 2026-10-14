"""Statistical baselines. All take history Yh (n, T0) and return forecasts (n, H)."""
import numpy as np


def naive(Yh, H):
    return np.repeat(Yh[:, -1:], H, axis=1)


def seasonal_naive(Yh, H, m=7):
    return np.tile(Yh[:, -m:], (1, int(np.ceil(H / m))))[:, :H]


def moving_average(Yh, H, window=28):
    return np.repeat(Yh[:, -window:].mean(1, keepdims=True), H, axis=1)


def croston_sba(Yh, H, alpha=0.1, window=365):
    """Croston with Syntetos-Boylan bias correction, vectorised across series (loop over time only)."""
    Yw = Yh[:, -window:]
    n = Yw.shape[0]
    z, p, q = np.zeros(n), np.ones(n), np.ones(n)
    init = np.zeros(n, dtype=bool)
    for t in range(Yw.shape[1]):
        y = Yw[:, t]
        nz = y > 0
        first = nz & ~init
        upd = nz & init
        z[first], p[first] = y[first], q[first]
        z[upd] = alpha * y[upd] + (1 - alpha) * z[upd]
        p[upd] = alpha * q[upd] + (1 - alpha) * p[upd]
        init |= nz
        q = np.where(nz, 1.0, q + 1.0)
    f = np.where(init, (1 - alpha / 2) * z / p, 0.0)
    return np.repeat(f[:, None], H, axis=1)


def empirical_quantiles(Yh, qs, H, window=90):
    """Distribution-free baseline: each series' own recent sales quantiles, constant over horizon."""
    q = np.quantile(Yh[:, -window:], qs, axis=1).T          # (n, K)
    return np.repeat(q[:, None, :], H, axis=1)              # (n, H, K)
