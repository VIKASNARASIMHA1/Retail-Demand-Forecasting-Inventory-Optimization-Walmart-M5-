"""Turn forecasts into stocking decisions and simulate the cost of those decisions.

Policy: periodic-review, order-up-to (base-stock). Every R days, order up to S, where S covers
demand over the protection period (lead time L + review period R) at a given service level.

 - Point-forecast policies:   S = sum(mu) + z * sigma * sqrt(L+R)        (textbook, normal approx.)
 - Quantile-forecast policies: S = service-level quantile of the *simulated* protection-period demand,
   sampled from the daily predictive quantiles (handles zero-inflated/intermittent demand).
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm


def unit_prices(P, T0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = np.nanmean(P[:, max(0, T0 - 28):T0], axis=1)
        p = np.where(np.isnan(p), np.nanmean(P[:, :T0], axis=1), p)
        return np.where(np.isnan(p), np.nanmedian(P), p)


def pad_forecast(a, extra):
    """Extend a forecast along the horizon axis by repeating its last week."""
    H = a.shape[1]
    idx = [H - 7 + (j % 7) for j in range(extra)]
    return np.concatenate([a, a[:, idx]], axis=1)


def base_stock_point(mu, sigma, review_days, L, R, levels):
    W = L + R
    mu_p = pad_forecast(mu, W)
    out = {}
    for lv in levels:
        z = norm.ppf(lv)
        S = np.stack([mu_p[:, t:t + W].sum(1) + z * sigma * np.sqrt(W) for t in review_days], axis=1)
        out[lv] = np.maximum(np.ceil(S), 0)
    return out


def _sample_demand(Vw, tau, n_draws, rng):
    """Inverse-CDF sampling by linear interpolation between predicted quantiles. Vw: (n, W, K+2)."""
    n, W, _ = Vw.shape
    u = rng.random((n, W, n_draws))
    idx = np.clip(np.searchsorted(tau, u, side="right"), 1, len(tau) - 1)
    lo = idx - 1
    v_lo = np.take_along_axis(Vw, lo, axis=2)
    v_hi = np.take_along_axis(Vw, idx, axis=2)
    w = (u - tau[lo]) / (tau[idx] - tau[lo])
    return np.rint(v_lo + w * (v_hi - v_lo))


def base_stock_quantile(q, qs, review_days, L, R, levels, n_draws, rng, chunk=2000):
    n, H, K = q.shape
    W = L + R
    qp = pad_forecast(q, W)
    slope = (qp[..., -1] - qp[..., -2]) / (qs[-1] - qs[-2])      # extrapolate upper tail to tau=1
    top = qp[..., -1] + slope * (1 - qs[-1])
    V = np.concatenate([np.zeros(qp.shape[:2] + (1,)), qp, top[..., None]], axis=2)
    V = np.maximum.accumulate(V, axis=2)
    tau = np.concatenate([[0.0], qs, [1.0]])
    out = {lv: np.zeros((n, len(review_days))) for lv in levels}
    for j, t in enumerate(review_days):
        for s in range(0, n, chunk):
            total = _sample_demand(V[s:s + chunk, t:t + W], tau, n_draws, rng).sum(1)   # (m, draws)
            S = np.quantile(total, levels, axis=1)                                      # (levels, m)
            for i, lv in enumerate(levels):
                out[lv][s:s + chunk, j] = np.ceil(S[i])
    return out


def simulate(actual, S, review_days, L, hold_cost, margin, penalty):
    """Day-by-day simulation for all series at once. Returns aggregate units and costs."""
    n, H = actual.shape
    arrivals = np.zeros((n, H + L + 1))
    on_hand = S[:, 0].copy()                                  # start fully stocked
    ridx = {d: i for i, d in enumerate(review_days)}
    sold_tot, lost_tot, hold_units = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(H):
        on_hand += arrivals[:, t]
        if t in ridx:
            position = on_hand + arrivals[:, t + 1:].sum(1)
            arrivals[:, t + L] += np.maximum(S[:, ridx[t]] - position, 0)
        d = actual[:, t]
        sold = np.minimum(on_hand, d)
        on_hand = on_hand - sold
        sold_tot += sold
        lost_tot += d - sold
        hold_units += on_hand
    return {"demand": float((sold_tot + lost_tot).sum()), "sold": float(sold_tot.sum()),
            "lost": float(lost_tot.sum()),
            "holding_cost": float((hold_units * hold_cost).sum()),
            "stockout_cost": float((lost_tot * margin * penalty).sum()),
            "avg_inventory": float(hold_units.sum() / H)}


def cost_at_fill(summary, target):
    """Total cost of each policy at a common fill rate (linear interpolation along its cost/service curve)."""
    rows = []
    for pol, g in summary.groupby("policy"):
        g = g.sort_values("fill_rate")
        rows.append(dict(policy=pol,
                         cost_at_target=float(np.interp(target, g["fill_rate"], g["total_cost"])),
                         in_range=bool(g["fill_rate"].min() <= target <= g["fill_rate"].max())))
    c = pd.DataFrame(rows).set_index("policy").sort_values("cost_at_target")
    c["saving_vs_ma28_%"] = 100 * (1 - c["cost_at_target"] / c.loc["ma28+normal_SS", "cost_at_target"])
    return c
