"""Forecast metrics (item-store level; a simplified M5 WRMSSE / scaled pinball loss)."""
import numpy as np


def _scale(Yh, power):
    """Mean |diff|^power of in-sample one-step naive errors, after the first sale."""
    d = np.abs(np.diff(Yh, axis=1)) ** power
    first = (Yh > 0).argmax(axis=1)
    mask = np.arange(1, Yh.shape[1])[None, :] > first[:, None]
    s = (d * mask).sum(1) / np.maximum(mask.sum(1), 1)
    return np.where(s > 0, s, np.nan)


def revenue_weights(Yh, price, window=28):
    r = (Yh[:, -window:] * price[:, None]).sum(1)
    return r / r.sum() if r.sum() > 0 else np.full(len(r), 1 / len(r))


def _wavg(v, w):
    ok = ~np.isnan(v) & (w > 0)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan


def wrmsse(Yh, actual, fc, w):
    r = np.sqrt(((actual - fc) ** 2).mean(1) / _scale(Yh, 2))
    return _wavg(r, w)


def spl(Yh, actual, q, qs, w):
    """Scaled pinball loss, revenue weighted. q: (n, H, K)."""
    diff = actual[:, :, None] - q
    loss = np.maximum(qs * diff, (qs - 1) * diff).mean(axis=(1, 2))
    return _wavg(loss / _scale(Yh, 1), w)


def coverage(actual, q, qs, lo=0.1, hi=0.9):
    il, ih = int(np.argmin(abs(qs - lo))), int(np.argmin(abs(qs - hi)))
    return float(((actual >= q[..., il]) & (actual <= q[..., ih])).mean())
