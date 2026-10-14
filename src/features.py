"""Vectorised, leak-free feature engineering.

All demand-derived features use lags >= 28 (= forecast horizon), so one model predicts the whole
28-day window directly (no recursive forecasting) and features are valid at any forecast origin.
Price and calendar features are legitimately known in advance.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

LAGS = (28, 35, 42, 49, 56, 364)
WINDOWS = (7, 14, 28, 56)
CATS = ["item_code", "dept_code", "cat_code", "store_code", "state_code"]


def shift(A: np.ndarray, k: int) -> np.ndarray:
    out = np.full(A.shape, np.nan, dtype=np.float32)
    if k < A.shape[1]:
        out[:, k:] = A[:, :-k]
    return out


def _win_sum(M: np.ndarray, w: int) -> np.ndarray:
    n, T = M.shape
    c0 = np.concatenate([np.zeros((n, 1)), np.cumsum(M, axis=1, dtype=np.float64)], axis=1)
    lo = np.maximum(0, np.arange(1, T + 1) - w)
    return c0[:, 1:] - c0[:, lo]


def rolling(A: np.ndarray, w: int):
    """Trailing-window mean and std (ignoring NaN) ending at each t, fully vectorised."""
    valid = ~np.isnan(A)
    x = np.where(valid, A, 0.0).astype(np.float64)
    cnt, s1, s2 = _win_sum(valid.astype(np.float64), w), _win_sum(x, w), _win_sum(x * x, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, s1 / cnt, np.nan)
        var = np.where(cnt > 0, s2 / cnt - mean ** 2, np.nan)
    return mean.astype(np.float32), np.sqrt(np.maximum(var, 0)).astype(np.float32)


def build_features(ds):
    Y, P = ds.Y, ds.P
    n, T = Y.shape
    F: dict[str, np.ndarray] = {}

    for k in LAGS:
        F[f"lag_{k}"] = shift(Y, k)
    B = shift(Y, 28)
    for w in WINDOWS:
        F[f"rmean_{w}"], std = rolling(B, w)
        if w == 28:
            F["rstd_28"] = std
    Z = np.where(np.isnan(B), np.nan, (B == 0)).astype(np.float32)
    F["rzero_28"], _ = rolling(Z, 28)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F["lag_dow_mean"] = np.nanmean(np.stack([F[f"lag_{k}"] for k in (28, 35, 42, 49)]), axis=0).astype(np.float32)
    F["trend_ratio"] = F["rmean_7"] / (F["rmean_56"] + 0.01)

    # price features (causal: running max, not full-sample max)
    F["price"] = P
    cm = np.maximum.accumulate(np.nan_to_num(P, nan=0.0), axis=1)
    F["price_norm"] = np.where(cm > 0, P / np.where(cm > 0, cm, 1), np.nan).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        F["price_chg_7"] = (P / shift(P, 7) - 1).astype(np.float32)

    # calendar
    d = pd.DatetimeIndex(ds.cal["date"])
    cal = {"dow": d.dayofweek, "dom": d.day, "month": d.month,
           "week": d.isocalendar().week.to_numpy(dtype=float),
           "event_flag": ds.cal["event_flag"].to_numpy(), "event_type": ds.cal["event_type"].to_numpy()}
    for k, v in cal.items():
        F[k] = np.broadcast_to(np.asarray(v, dtype=np.float32), (n, T))
    F["snap"] = ds.snap.astype(np.float32)

    for col, name in zip(["item_id", "dept_id", "cat_id", "store_id", "state_id"], CATS):
        codes = ds.meta[col].astype("category").cat.codes.to_numpy().astype(np.int32)
        F[name] = np.broadcast_to(codes[:, None], (n, T))

    X = pd.DataFrame({k: np.ascontiguousarray(v).reshape(-1) for k, v in F.items()})
    return X, Y.reshape(-1), np.tile(np.arange(T), n), list(F), CATS
