"""Interactive demo: pick a product-store, see the forecast band and a recommended order.
Run:  streamlit run app/streamlit_app.py     (needs results/app_data.npz from `python -m src.run`)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
from src import inventory as inv

st.set_page_config(page_title="Demand forecast & order recommendation", layout="wide")


@st.cache_data
def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


path = st.sidebar.text_input("Forecast file", "results/app_data.npz")
if not Path(path).exists():
    st.error(f"`{path}` not found. Run `python -m src.run` first.")
    st.stop()
D = load(path)
ids, qs, q_all = D["ids"], D["qs"], D["q"].astype(float)

st.title("Demand forecast & order recommendation")
st.caption("Backtest view: forecasts were made at the start of the window using only earlier data. "
           "Tick the box to reveal what actually sold.")

# ---- sidebar controls ----
flt = st.sidebar.text_input("Filter products (e.g. FOODS_3 or CA_1)", "")
options = [i for i in ids if flt.lower() in i.lower()] or list(ids)
rev = (D["hist"][:, -28:].sum(1) * D["price"])
options = sorted(options, key=lambda s: -rev[list(ids).index(s)])          # top revenue first
pid = st.sidebar.selectbox("Product-store", options)
i = list(ids).index(pid)
st.sidebar.markdown("### Inventory policy")
lv = st.sidebar.slider("Service level (probability of no stock-out in a cycle)", 0.50, 0.99, 0.95, 0.01)
L = st.sidebar.number_input("Lead time (days)", 1, 14, 2)
Rp = st.sidebar.number_input("Review period (days)", 1, 14, 7)
on_hand = st.sidebar.number_input("Units on hand now", 0, 10000, 0)
on_order = st.sidebar.number_input("Units already on order", 0, 10000, 0)
show_actual = st.sidebar.checkbox("Show actual demand (backtest)", False)

# ---- decision ----
hist, q, W = D["hist"][i], q_all[i], int(L + Rp)
rng = np.random.default_rng(0)
S_q = inv.base_stock_quantile(q[None], qs, [0], int(L), int(Rp), [lv], 3000, rng)[lv][0, 0]
mu, sd = hist[-28:].mean() * W, hist[-90:].std() * np.sqrt(W)
S_c = max(np.ceil(mu + norm.ppf(lv) * sd), 0)
med_i = int(np.argmin(abs(qs - 0.5)))
position = on_hand + on_order

c1, c2, c3, c4 = st.columns(4)
c1.metric("Order-up-to level (quantile model)", f"{S_q:.0f} units")
c2.metric("Order-up-to level (classical: avg + safety stock)", f"{S_c:.0f} units")
c3.metric("Median demand, next %d days" % W, f"{q[:W, med_i].sum():.0f} units")
c4.metric("Recommended order now", f"{max(S_q - position, 0):.0f} units",
          help="order-up-to level minus units on hand and on order")
st.write(f"**{pid}** - average price about **${D['price'][i]:.2f}**. "
         f"The quantile policy stocks to cover {W} days at a {lv:.0%} service level.")

# ---- chart ----
hd, fd = D["hist_dates"], D["fut_dates"]
fig, ax = plt.subplots(figsize=(11, 4.2))
xh, xf = np.arange(-len(hist), 0), np.arange(len(fd))
ax.plot(xh[-45:], hist[-45:], color="gray", label="recent sales")
j = lambda p: int(np.argmin(abs(qs - p)))
ax.fill_between(xf, q[:, j(.1)], q[:, j(.9)], alpha=.2, color="C0", label="P10-P90")
ax.fill_between(xf, q[:, j(.25)], q[:, j(.75)], alpha=.35, color="C0", label="P25-P75")
ax.plot(xf, q[:, med_i], color="C0", label="median forecast")
if show_actual:
    ax.plot(xf, D["actual"][i], "k.-", label="actual")
ax.axvline(-0.5, color="k", ls=":", lw=.8)
ax.set_xlabel(f"days from forecast origin ({fd[0]} onward)"); ax.set_ylabel("units/day")
ax.legend(ncol=5, fontsize=8); ax.grid(alpha=.3)
st.pyplot(fig)

with st.expander("Daily forecast quantiles"):
    import pandas as pd
    df = pd.DataFrame(q, columns=[f"P{int(a * 100)}" for a in qs], index=fd).round(2)
    if show_actual:
        df["actual"] = D["actual"][i]
    st.dataframe(df)
st.caption("Order-up-to level = service-level quantile of simulated demand over lead time + review period, "
           "sampled from the model's daily quantiles (days assumed independent).")
