"""Global LightGBM models: one Tweedie model for the mean + one model per quantile."""
import numpy as np
import pandas as pd
import lightgbm as lgb


def fit_predict(X, y, t, n, T_cut, H, cfg, qs, features, cats, explain_n=0):
    lc = cfg["lgb"]
    tr = (t >= T_cut - lc["train_window_days"]) & (t < T_cut) & (t >= 63)   # history = days 0..T_cut-1
    va = (t >= T_cut) & (t < T_cut + H)
    Xtr, Xva = X.loc[tr, features], X.loc[va, features]
    dtrain = lgb.Dataset(Xtr, label=y[tr], categorical_feature=cats, free_raw_data=False)
    params = {**lc["params"], "seed": cfg["seed"]}
    rounds = lc["num_boost_round"]

    m_mean = lgb.train({**params, "objective": "tweedie",
                        "tweedie_variance_power": lc["tweedie_variance_power"]}, dtrain, rounds)
    mean = np.clip(m_mean.predict(Xva), 0, None).reshape(n, H)

    q = np.empty((n, H, len(qs)))
    for j, a in enumerate(qs):
        m = lgb.train({**params, "objective": "quantile", "alpha": float(a)}, dtrain, rounds)
        q[:, :, j] = m.predict(Xva).reshape(n, H)
    q = np.sort(np.clip(q, 0, None), axis=2)                 # no negatives, no quantile crossing
    imp = pd.Series(m_mean.feature_importance("gain"), index=features).sort_values(ascending=False)
    shap = None
    if explain_n:   # LightGBM's built-in TreeSHAP (pred_contrib); last column is the bias term
        idx = np.random.default_rng(cfg["seed"]).choice(len(Xva), min(explain_n, len(Xva)), replace=False)
        contrib = m_mean.predict(Xva.iloc[idx], pred_contrib=True)
        shap = pd.Series(np.abs(contrib[:, :-1]).mean(0), index=features).sort_values(ascending=False)
    return {"mean": mean, "q": q, "importance": imp, "shap": shap}
