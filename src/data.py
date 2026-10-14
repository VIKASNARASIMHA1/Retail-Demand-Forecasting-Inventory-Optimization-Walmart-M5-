"""Data loading. Everything is held as dense (n_series, n_days) arrays - fast and memory-light."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class Dataset:
    meta: pd.DataFrame      # one row per series: id, item_id, dept_id, cat_id, store_id, state_id
    Y: np.ndarray           # (n, T) units sold
    P: np.ndarray           # (n, T) sell price, NaN = not yet on sale
    snap: np.ndarray        # (n, T) SNAP benefit day flag
    cal: pd.DataFrame       # T rows: date, event_flag, event_type

    @property
    def n(self): return self.Y.shape[0]

    @property
    def T(self): return self.Y.shape[1]


def load_m5(cfg: dict, seed: int) -> Dataset:
    d = Path(cfg["dir"])
    sales = pd.read_csv(d / cfg["sales_file"])
    if cfg.get("n_series"):
        sales = sales.sample(n=min(cfg["n_series"], len(sales)), random_state=seed)
    sales = sales.sort_values("id").reset_index(drop=True)
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    T, n = len(day_cols), len(sales)

    cal = pd.read_csv(d / "calendar.csv").iloc[:T].copy()
    cal["date"] = pd.to_datetime(cal["date"])
    cal["event_flag"] = cal["event_name_1"].notna().astype(np.int8)
    cal["event_type"] = (cal["event_type_1"].astype("category").cat.codes + 1).astype(np.int8)

    meta = sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].copy()
    Y = sales[day_cols].to_numpy(dtype=np.float32)

    snap = np.zeros((n, T), dtype=np.int8)
    for st in meta["state_id"].unique():
        snap[(meta["state_id"] == st).to_numpy()] = cal[f"snap_{st}"].to_numpy(np.int8)[None, :]

    sp = pd.read_csv(d / "sell_prices.csv")
    keys = meta[["item_id", "store_id"]].reset_index().rename(columns={"index": "row"})
    sp = sp.merge(keys, on=["item_id", "store_id"])
    weeks = cal["wm_yr_wk"].to_numpy()
    uw = np.unique(weeks)
    sp = sp[sp["wm_yr_wk"].isin(uw)]
    Pw = np.full((n, len(uw)), np.nan, dtype=np.float32)
    Pw[sp["row"].to_numpy(), np.searchsorted(uw, sp["wm_yr_wk"].to_numpy())] = sp["sell_price"].to_numpy()
    P = Pw[:, np.searchsorted(uw, weeks)]
    return Dataset(meta, Y, P, snap, cal[["date", "event_flag", "event_type"]].reset_index(drop=True))


def make_synthetic(cfg: dict, seed: int) -> Dataset:
    """M5-like intermittent demand with weekly/annual seasonality, promos, events and late launches."""
    rng = np.random.default_rng(seed)
    n, T = cfg["synthetic_n_series"], cfg["synthetic_days"]
    states = ["CA", "TX", "WI"]
    stores = [f"{s}_{k}" for s in states for k in (1, 2)]
    depts = {"FOODS": ["FOODS_1", "FOODS_2"], "HOBBIES": ["HOBBIES_1"], "HOUSEHOLD": ["HOUSEHOLD_1"]}
    rows = []
    for i in range(n):
        store = stores[i % len(stores)]
        cat = str(rng.choice(list(depts)))
        dept = str(rng.choice(depts[cat]))
        item = f"{dept}_{i // len(stores):04d}"
        rows.append(dict(id=f"{item}_{store}", item_id=item, dept_id=dept, cat_id=cat,
                         store_id=store, state_id=store[:2]))
    meta = pd.DataFrame(rows)

    dates = pd.date_range("2014-01-01", periods=T)
    dowvec = np.array([-0.1, -0.2, -0.2, -0.1, 0.1, 0.4, 0.35])
    base = np.clip(rng.lognormal(-0.5, 1.0, n), 0.02, 30)
    dow = 1 + rng.uniform(0.3, 1.2, n)[:, None] * dowvec[dates.dayofweek][None, :]
    annual = 1 + 0.25 * np.sin(2 * np.pi * dates.dayofyear.to_numpy()[None, :] / 365 + rng.uniform(0, 6.28, n)[:, None])

    price0 = rng.uniform(1, 20, n)
    promo = np.zeros((n, T), dtype=bool)
    for i in range(n):
        for s in rng.integers(0, T - 14, rng.integers(3, 7)):
            promo[i, s:s + 14] = True
    P = (price0[:, None] * np.where(promo, 0.8, 1.0)).astype(np.float32)

    event_flag = (rng.random(T) < 0.02).astype(np.int8)
    event_type = (event_flag * rng.integers(1, 4, T)).astype(np.int8)
    lam = base[:, None] * dow * annual * np.where(promo, 1.5, 1.0) * np.where(event_flag[None, :] == 1, 1.3, 1.0)
    Y = rng.poisson(rng.gamma(2.0, lam / 2.0)).astype(np.float32)

    start = np.where(rng.random(n) < 0.2, rng.integers(0, T // 3, n), 0)
    for i in np.where(start > 0)[0]:
        Y[i, :start[i]] = 0
        P[i, :start[i]] = np.nan

    dom = dates.day.to_numpy()
    snap = ((dom <= 10)[None, :] & (meta["state_id"].to_numpy() != "WI")[:, None]).astype(np.int8)
    cal = pd.DataFrame({"date": dates, "event_flag": event_flag, "event_type": event_type})
    return Dataset(meta, Y, P, snap, cal)


def load_dataset(cfg: dict, seed: int) -> Dataset:
    if cfg["source"] == "synthetic":
        return make_synthetic(cfg, seed)
    return load_m5(cfg, seed)
