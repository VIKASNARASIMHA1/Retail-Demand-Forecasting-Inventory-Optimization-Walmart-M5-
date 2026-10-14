# Interview guide: know every decision in this project

Read this, then open the code file named in each answer and trace it. If you can explain it in your own
words in ~30 seconds, you own it. Fill the `[ ]` blanks with YOUR numbers from `docs/RESULTS.md`.

## 30-second pitch
"Retailers lose money two ways: stock-outs and excess inventory. I used Walmart's M5 data to forecast
28-day demand per product-store, predicted full uncertainty ranges rather than single numbers, and fed
those into a simulated reordering policy. I judged the models by dollar cost at the same service level,
not just forecast error. On [N] series the best approach cut simulated cost by [X]% vs a moving average
at a 95% fill rate, [and the result held / did not hold] under different cost assumptions."

## Facts to have memorised
- M5: Walmart, 3,049 products x 10 stores = 30,490 item-store series, 1,941 training days, forecast horizon 28 days,
  ~60%+ of daily values are zero (intermittent), plus prices, SNAP days and events.
- Your run: [n_series] series, [3] rolling folds of 28 days, [29] features, LightGBM.
- Official M5 metric is WRMSSE over 12 aggregation levels. Yours is item-store level only (simplified).

## Core design questions

**Why lags of at least 28? (`features.py`)**
I predict all 28 days in one shot (direct strategy). Any feature using sales newer than 28 days before the
target would not exist at forecast time for later days in the window, so I only use lag 28+ and rolling
stats of the 28-day-shifted series. This avoids both leakage and recursive forecasting, where each day's
prediction feeds the next and errors compound. Trade-off: I discard the freshest 27 days of information,
which costs some accuracy at short horizons.

**Why one global LightGBM instead of a model per series?**
Series are short, noisy and mostly zeros. A global model shares patterns (weekday, promo, event effects)
across products via category codes and rolling features. Trade-off: it can blur series-specific quirks.

**Why Tweedie loss for the mean model?**
Sales are non-negative, right-skewed and zero-heavy. Tweedie with power between 1 and 2 is a compound
Poisson-gamma: a point mass at zero plus a continuous positive part, with a log link so predictions are
never negative. Plain RMSE on such data pulls toward the mean and handles the zeros poorly. I used 1.1;
it is a tunable hyperparameter.

**Why quantile forecasts and pinball loss?**
The stocking decision depends on the upper tail of demand, not the average. The pinball loss is minimised
by the true quantile, so training one model per quantile gives P10...P90 directly. Newsvendor logic: the
optimal stock level is a quantile of demand set by the cost ratio of under- vs over-stocking.

**Why these baselines? (`baselines.py`)**
Naive and seasonal naive are the standard sanity floors. Croston-SBA is the classic intermittent-demand
method: it smooths demand size and the gap between demands separately, with a (1 - alpha/2) bias correction.
The empirical-quantile baseline checks whether ML adds anything beyond each series' own recent distribution.

**Why rolling-origin backtesting? (`run.py`)**
Random splits let the model see the future relative to the test rows (through seasonal patterns and lag
features), inflating scores. Rolling origin trains only on the past and tests on the next 28 days, repeated
3 times to see stability. Weakness: only 3 folds, so results have error bars I have not computed.

**What are WRMSSE and scaled pinball loss? (`metrics.py`)**
RMSSE divides the forecast's RMSE by the in-sample RMSE of a one-step naive forecast, making series on
different scales comparable (<1 means better than naive). Weighting by recent revenue makes big sellers
matter more. Scaled pinball does the same for quantile forecasts. I also report P10-P90 coverage
(should be ~80%) to check calibration, since sharp but wrong intervals are useless.

**Why compare policies at equal fill rate? (`inventory.cost_at_fill`)**
Any policy can look cheap by tolerating more stock-outs. Costs are only comparable at the same service
level, so I simulate each policy across service levels 70-99%, then interpolate each policy's cost at a
common target fill rate (95%). Limitation: interpolation clamps if the target lies outside a policy's range
(the `in_range` column flags this; the oracle is out of range because it hits ~100%).

**Why Monte-Carlo base stock instead of the normal formula? (`inventory.py`)**
Normal safety-stock maths assumes continuous, symmetric demand. Zero-inflated demand is neither. Also the
sum of daily quantiles is not the quantile of the sum. So I sample daily demand by interpolating between
predicted quantiles, sum over lead time + review period, and take the service-level quantile. Assumptions:
days are independent, and the tail above P90 is linearly extrapolated.

**Why include an oracle policy?**
It gets perfect foresight, so it bounds what any forecast could achieve and shows how much headroom is left.

## Stretch items

**Conformal prediction (`conformal.py`)**
Raw quantile models are often mis-calibrated (coverage != 80%). Conformalized quantile regression measures
how far actuals fall outside the P10-P90 band on a calibration set and widens or narrows the band by that
amount, normalised by series scale. I calibrate on the previous backtest fold. Guarantee caveat: it
requires exchangeable data and time series are not, so coverage is approximate. Result: coverage moved
from [raw]% to [conformal]% (see `results/conformal_check.csv`).

**SHAP (`lgb_models.py`)**
TreeSHAP via LightGBM's `pred_contrib=True`: each prediction is decomposed into additive feature
contributions. Mean |SHAP| is global importance in units of predicted demand. It differs from gain
importance, which measures split usefulness during training. Caveat: correlated features (many lag/rolling
features here) share credit, so do not over-read single bars.

**Sensitivity analysis (`run.py: scenario_grid`)**
The cost ratio, holding rate and stock-out penalty are my assumptions, not facts. I vary each one at a
time and re-run the simulation. If the best policy stays best, the conclusion is robust; if not, say so.
Limitation: one-at-a-time, not a full grid.

## Leakage and correctness checks (they WILL ask)
- Demand features: lag >= 28 only -> known at forecast origin.
- Price features are legitimate (planned prices are known ahead); `price_norm` uses a running max, not the full-sample max.
- Training rows end before the cutoff; validation rows are the next 28 days; features are computed once but never use data past the target minus 28 days.
- Category codes are built from the sample, which is not target leakage.
- Weights and scales for metrics use history only.

## Honest limitations (say them before they ask)
Item-store WRMSSE only; no hierarchical reconciliation; no hyperparameter tuning; 3 folds; subsample of
series; assumed costs; independence assumption in the Monte-Carlo step; conformal guarantee approximate.

## Likely follow-ups
- *How would you scale to all 30,490 series?* Train on a longer-but-sampled window, float32 features (already), LightGBM on a bigger machine or Spark/Dask, one model per store or department.
- *How would you deploy it?* Weekly batch job producing quantile forecasts to a table; the order recommendation is a small function on top (that is what the Streamlit app does); monitor coverage and fill rate for drift.
- *What would you try next?* Hierarchical reconciliation (MinT), tuning with Optuna, N-BEATS/TFT, tuning the Tweedie power, more folds, bootstrap confidence intervals on the savings.
- *Why LightGBM?* Fast on tabular data with many categorical and lag features, native Tweedie and quantile objectives, handles NaNs. Won most of the M5 competition.

## Study order (about 90 minutes)
`config.yaml` -> `run.py` (the loop) -> `features.py` -> `lgb_models.py` -> `metrics.py` ->
`inventory.py` -> `conformal.py`. For each: what goes in, what comes out, and one thing that could break.
