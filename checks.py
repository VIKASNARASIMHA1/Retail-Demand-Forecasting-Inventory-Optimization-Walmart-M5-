"""Self-checks for the project.   python checks.py            (code checks, no data needed)
                                  python checks.py --results   (also validates results/ from a real run)"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ok_all = True
def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

# ---------- 1. No future leakage in features ----------
from src.data import make_synthetic
from src.features import build_features
cfg = {"synthetic_n_series": 40, "synthetic_days": 500}
ds = make_synthetic(cfg, 1)
X1, *_ = build_features(ds)
Tc, H = 400, 28
ds2 = make_synthetic(cfg, 1)
rng = np.random.default_rng(0)
ds2.Y[:, Tc:] = rng.poisson(50, ds2.Y[:, Tc:].shape)           # scramble all sales from the cutoff onward
X2, y2, t2, *_ = build_features(ds2)
rows = t2 < Tc + H                                              # every row a fold may train on or predict
same = all(np.allclose(X1.loc[rows, c], X2.loc[rows, c], equal_nan=True) for c in X1.columns)
check("Features for training + forecast rows do not change when future sales change (no leakage)", same)

# ---------- 2. Inventory simulator behaves sensibly ----------
from src import inventory as inv
actual = np.full((5, 28), 2.0); price_h = np.full(5, 0.01); m = np.full(5, 1.0)
rd = list(range(0, 28, 7))
big = inv.simulate(actual, np.full((5, 4), 1e6), rd, 2, price_h, m, 1.5)
zero = inv.simulate(actual, np.zeros((5, 4)), rd, 2, price_h, m, 1.5)
check("Huge stock -> fill rate 100%", abs(big["sold"] / big["demand"] - 1) < 1e-9)
check("Zero stock -> fill rate 0% and no holding cost", zero["sold"] == 0 and zero["holding_cost"] == 0)
check("Sold + lost = demand", abs(big["sold"] + big["lost"] - big["demand"]) < 1e-6)
mid = inv.simulate(actual, np.full((5, 4), 20.0), rd, 2, price_h, m, 1.5)
check("More stock -> higher fill rate", big["sold"] >= mid["sold"] >= zero["sold"])

# ---------- 3. Quantile / conformal logic ----------
from src import conformal as cf
q = np.sort(np.random.default_rng(1).random((10, 28, 5)) * 5, axis=2)
a = np.random.default_rng(2).poisson(2, (10, 28)).astype(float)
sc = np.ones(10)
Q = cf.calibrate(a, q, sc, 0.2)
q2 = cf.apply(q, Q, sc)
check("Conformal output has no crossing quantiles and no negatives", (np.diff(q2, axis=2) >= 0).all() and (q2 >= 0).all())
S = inv.base_stock_quantile(q, np.array([.1, .25, .5, .75, .9]), [0], 2, 7, [0.8, 0.95], 500, np.random.default_rng(0))
check("Higher service level -> higher (or equal) order-up-to level", (S[0.95] >= S[0.8]).all())

# ---------- 4. Validate a real run's outputs ----------
if "--results" in sys.argv:
    R = Path("results")
    fm = pd.read_csv(R / "forecast_metrics.csv", index_col=0)
    check("naive is worse than the moving average", fm.loc["naive", "wrmsse"] > fm.loc["ma28", "wrmsse"])
    check("No NaN in WRMSSE", fm["wrmsse"].notna().all())
    cov = fm.loc["lgb_quantile", "coverage_10_90"]
    check("P10-P90 coverage is plausible (0.70-0.95)", 0.70 <= cov <= 0.95, f"(got {cov:.3f})")
    z = np.load(R / "app_data.npz")
    check("Saved quantile forecasts are sorted and non-negative", (np.diff(z["q"], axis=2) >= -1e-6).all() and (z["q"] >= 0).all())
    s = pd.read_csv(R / "inventory_summary.csv"); b = s[s.scenario == "base"]
    check("Fill rates are between 0 and 1", b.fill_rate.between(0, 1).all())
    # raising the service level should (almost always) raise fill rate: allow small Monte-Carlo noise
    viol = 0
    for pol, g in b.groupby("policy"):
        viol += (np.diff(g.sort_values("service_level").fill_rate.values) < -0.005).sum()
    check("Fill rate rises with service level (within noise)", viol == 0, f"({viol} violations)")
    orc = b[b.policy.str.startswith("oracle")].fill_rate.min()
    check("Oracle (perfect foresight) reaches ~100% fill", orc > 0.99, f"(min {orc:.3f})")

print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED - read the FAIL lines above")
sys.exit(0 if ok_all else 1)
