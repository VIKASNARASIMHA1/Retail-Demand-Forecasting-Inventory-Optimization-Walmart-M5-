import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _i(qs, v):
    return int(np.argmin(abs(np.asarray(qs) - v)))


def plot_forecasts(path, ds, Tc, H, q, qs, w, n_items=4, hist=60):
    top = np.argsort(-w)[:n_items]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    xh, xf = np.arange(-hist, 0), np.arange(H)
    for ax, i in zip(axes.ravel(), top):
        ax.plot(xh, ds.Y[i, Tc - hist:Tc], color="gray", label="history")
        ax.plot(xf, ds.Y[i, Tc:Tc + H], "k.-", label="actual")
        ax.fill_between(xf, q[i, :, _i(qs, .1)], q[i, :, _i(qs, .9)], alpha=.2, color="C0", label="P10-P90")
        ax.fill_between(xf, q[i, :, _i(qs, .25)], q[i, :, _i(qs, .75)], alpha=.35, color="C0", label="P25-P75")
        ax.plot(xf, q[i, :, _i(qs, .5)], color="C0", label="median")
        ax.set_title(ds.meta["id"].iloc[i], fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("LightGBM quantile forecasts vs actuals (last backtest fold, top-revenue series)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_frontier(path, summary):
    fig, ax = plt.subplots(figsize=(8, 5))
    for pol, g in summary.groupby("policy"):
        g = g.sort_values("fill_rate")
        ax.plot(g["fill_rate"] * 100, g["total_cost"], "o-", label=pol)
    ax.set_xlabel("Fill rate (% of demand served)")
    ax.set_ylabel("Total cost (holding + stockout)")
    ax.set_title("Cost vs service: lower-left is better")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_importance(path, imp, k=15, title="LightGBM (Tweedie) feature importance - gain"):
    s = imp.head(k)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(s.index, s.values)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_sensitivity(path, pivot, cols=("lgb_quantile+MC", "lgb_tweedie+normal_SS", "croston_sba+normal_SS")):
    cols = [c for c in cols if c in pivot.columns]
    ax = pivot[cols].plot.barh(figsize=(9, 5.5))
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("Cost saving vs 28-day moving average at target fill rate (%)")
    ax.set_title("Sensitivity of the business conclusion to cost assumptions")
    ax.invert_yaxis(); ax.grid(alpha=.3, axis="x")
    fig = ax.get_figure(); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
