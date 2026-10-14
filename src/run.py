"""End-to-end pipeline:  python -m src.run --config config.yaml [--synthetic]"""
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from . import baselines as bl, metrics as mt, inventory as inv, report, conformal as cf
from .data import load_dataset
from .features import build_features
from .lgb_models import fit_predict


def scenario_grid(ic, sens):
    base = {k: ic[k] for k in ("cost_ratio", "holding_rate_annual", "stockout_penalty")}
    sc = {"base": base}
    for key, vals in (sens or {}).items():
        for v in vals:
            sc[f"{key}={v}"] = {**base, key: v}
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--synthetic", action="store_true", help="use generated data (no download needed)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.synthetic:
        cfg["data"]["source"] = "synthetic"
    seed = cfg["seed"]
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    t0 = time.time()
    ds = load_dataset(cfg["data"], seed)
    print(f"Loaded {ds.n} series x {ds.T} days  (zeros: {(ds.Y == 0).mean():.0%})")
    X, y, t, features, cats = build_features(ds)
    print(f"Features: {len(features)}  rows: {len(X):,}  ({time.time() - t0:.0f}s)")

    H = cfg["forecast"]["horizon"]
    K = cfg["forecast"]["n_folds"]
    qs = np.array(cfg["forecast"]["quantiles"])
    alpha = 1 - (qs[-1] - qs[0])                              # e.g. P10-P90 -> alpha = 0.2
    ic = cfg["inventory"]
    L, R = ic["lead_time"], ic["review_period"]
    levels = ic["service_levels"]
    review_days = list(range(0, H, R))
    scenarios = scenario_grid(ic, cfg.get("sensitivity"))
    cutoffs = [ds.T - H * (K - k) for k in range(K)]          # rolling origin, never random splits
    med = int(np.argmin(abs(qs - 0.5)))

    metric_rows, inv_rows, conf_rows, prev = [], [], [], None
    for fi, Tc in enumerate(cutoffs):
        last = fi == K - 1
        print(f"\n=== Fold {fi + 1}/{K}: train on days 1..{Tc}, validate on days {Tc + 1}..{Tc + H} ===")
        Yh, actual = ds.Y[:, :Tc], ds.Y[:, Tc:Tc + H]
        price = inv.unit_prices(ds.P, Tc)
        w = mt.revenue_weights(Yh, price)

        point = {"naive": bl.naive(Yh, H), "seasonal_naive": bl.seasonal_naive(Yh, H),
                 "ma28": bl.moving_average(Yh, H), "croston_sba": bl.croston_sba(Yh, H)}
        res = fit_predict(X, y, t, ds.n, Tc, H, cfg, qs, features, cats,
                          explain_n=cfg["shap"]["n_samples"] if last else 0)
        point["lgb_tweedie"] = res["mean"]
        quant = {"empirical_q": bl.empirical_quantiles(Yh, qs, H), "lgb_quantile": res["q"]}

        # ---------- forecast accuracy ----------
        for name, fc in point.items():
            metric_rows.append(dict(fold=fi, model=name, wrmsse=mt.wrmsse(Yh, actual, fc, w),
                                    spl=np.nan, coverage_10_90=np.nan))
        for name, q in quant.items():
            metric_rows.append(dict(fold=fi, model=name, wrmsse=mt.wrmsse(Yh, actual, q[..., med], w),
                                    spl=mt.spl(Yh, actual, q, qs, w), coverage_10_90=mt.coverage(actual, q, qs)))

        # ---------- conformal calibration (uses previous fold's out-of-sample errors) ----------
        scale = cf.scale_vec(Yh)
        if prev is not None:
            Q = cf.calibrate(prev["actual"], prev["q"], prev["scale"], alpha)
            q_cf = cf.apply(res["q"], Q, scale)
            for variant, q in (("raw", res["q"]), ("conformal", q_cf)):
                conf_rows.append(dict(fold=fi, variant=variant, Q=Q if variant == "conformal" else np.nan,
                                      coverage_10_90=mt.coverage(actual, q, qs), spl=mt.spl(Yh, actual, q, qs, w),
                                      mean_width_10_90=float((q[..., -1] - q[..., 0]).mean())))
        prev = dict(actual=actual, q=res["q"], scale=scale)

        # ---------- inventory decisions + simulation (all cost scenarios) ----------
        sigma = Yh[:, -ic["sigma_window"]:].std(1)
        stocks = {"oracle (perfect foresight)": inv.base_stock_point(actual, np.zeros(ds.n), review_days, L, R, levels)}
        for name in ["ma28", "croston_sba", "lgb_tweedie"]:
            stocks[f"{name}+normal_SS"] = inv.base_stock_point(point[name], sigma, review_days, L, R, levels)
        for name, q in quant.items():
            stocks[f"{name}+MC"] = inv.base_stock_quantile(q, qs, review_days, L, R, levels, ic["n_draws"], rng)
        for sname, sp in scenarios.items():
            hold = sp["holding_rate_annual"] * sp["cost_ratio"] * price / 365
            margin = (1 - sp["cost_ratio"]) * price
            for pol, S_by_level in stocks.items():
                for lv in levels:
                    r = inv.simulate(actual, S_by_level[lv], review_days, L, hold, margin, sp["stockout_penalty"])
                    inv_rows.append(dict(fold=fi, scenario=sname, policy=pol, service_level=lv, **r))

        if last:
            report.plot_forecasts(out / "forecast_examples.png", ds, Tc, H, res["q"], qs, w)
            report.plot_importance(out / "feature_importance.png", res["importance"])
            report.plot_importance(out / "shap_importance.png", res["shap"],
                                   title="Mean |SHAP| (impact on predicted units/day)")
            res["shap"].to_csv(out / "shap_importance.csv", header=["mean_abs_shap"])
            # data for the Streamlit app
            np.savez_compressed(
                out / "app_data.npz", ids=ds.meta["id"].to_numpy().astype(str),
                hist=ds.Y[:, Tc - 90:Tc], actual=actual, q=res["q"].astype(np.float32), qs=qs, price=price,
                hist_dates=ds.cal["date"].dt.strftime("%Y-%m-%d").to_numpy()[Tc - 90:Tc].astype(str),
                fut_dates=ds.cal["date"].dt.strftime("%Y-%m-%d").to_numpy()[Tc:Tc + H].astype(str))

    # ---------- aggregate: forecast accuracy ----------
    m = pd.DataFrame(metric_rows)
    mt_summary = m.groupby("model")[["wrmsse", "spl", "coverage_10_90"]].mean().sort_values("wrmsse")
    mt_summary.to_csv(out / "forecast_metrics.csv"); m.to_csv(out / "forecast_metrics_by_fold.csv", index=False)

    if conf_rows:
        cdf = pd.DataFrame(conf_rows)
        cdf.groupby("variant")[["coverage_10_90", "spl", "mean_width_10_90", "Q"]].mean().to_csv(out / "conformal_check.csv")

    # ---------- aggregate: inventory ----------
    d = pd.DataFrame(inv_rows); d.to_csv(out / "inventory_by_fold.csv", index=False)
    s = d.groupby(["scenario", "policy", "service_level"], as_index=False)[
        ["demand", "sold", "lost", "holding_cost", "stockout_cost", "avg_inventory"]].sum()
    s["fill_rate"] = s["sold"] / s["demand"]
    s["total_cost"] = s["holding_cost"] + s["stockout_cost"]
    s.to_csv(out / "inventory_summary.csv", index=False)
    base = s[s["scenario"] == "base"]
    report.plot_frontier(out / "cost_vs_service.png", base)

    tgt = ic["target_fill_rate"]
    c = inv.cost_at_fill(base, tgt)
    c.to_csv(out / "headline_cost_at_target_fill_rate.csv")

    sens = pd.concat({sn: inv.cost_at_fill(g, tgt)["saving_vs_ma28_%"] for sn, g in s.groupby("scenario")}, axis=1).T
    sens.index.name = "scenario"
    sens.to_csv(out / "sensitivity.csv")
    report.plot_sensitivity(out / "sensitivity.png", sens)

    pd.set_option("display.width", 150)
    print("\n================ FORECAST ACCURACY (mean over folds; lower WRMSSE/SPL is better) ================")
    print(mt_summary.round(4).to_string()); print("(coverage_10_90 should be ~0.80)")
    if conf_rows:
        print("\n================ CONFORMAL CALIBRATION (folds 2..K) ================")
        print(pd.read_csv(out / "conformal_check.csv", index_col=0).round(4).to_string())
    print(f"\n================ TOTAL COST AT {tgt:.0%} FILL RATE (base costs) ================")
    print(c.round(2).to_string())
    print("\n================ SENSITIVITY: % saving vs ma28 at target fill rate ================")
    print(sens.round(1).to_string())
    print(f"\nDone in {time.time() - t0:.0f}s. Outputs in {out}/")


if __name__ == "__main__":
    main()
