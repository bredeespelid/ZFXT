"""
Data Preparation Pipeline for TimesFM-Style Foundation Dataset
Author: Brede Espelid
Description:
    This script reconstructs the daily, gap-free FX dataset (1980–1999)
    and enriches it with synthetic macroeconomic and technical features.
    Output: train_df_timeseries.parquet
"""

import numpy as np
import pandas as pd
import requests
from sklearn.decomposition import PCA

# ------------------------------------------------------------
# 1. Fetch and parse Norges Bank SDMX JSON data
# ------------------------------------------------------------
URL = (
    "https://data.norges-bank.no/api/data/EXR/"
    "B.USD+AUD+BDT+BYN+BRL+GBP+BGN+DKK+EUR+PHP+HKD+XDR+I44+INR+IDR+TWI+ISK+JPY+CAD+"
    "CNY+HRK+MYR+MXN+MMK+NZD+ILS+RON+TWD+RUB+PKR+PLN+SGD+CHF+SEK+ZAR+KRW+THB+CZK+"
    "TRY+HUF+VND.NOK.SP"
    "?format=sdmx-json&startPeriod=1970-01-01&endPeriod=1999-12-31&locale=no"
)
data = requests.get(URL, timeout=60).json()

structure = data["data"]["structure"]
series_dict = data["data"]["dataSets"][0]["series"]
series_dims = structure["dimensions"]["series"]
obs_dims = structure["dimensions"]["observation"]

# Parse JSON into long-form DataFrame
records = []
for key, val in series_dict.items():
    idxs = [int(i) for i in key.split(":")]
    meta = {dim["id"]: dim["values"][idx]["id"] for dim, idx in zip(series_dims, idxs)}
    for obs_idx, obs_val in val.get("observations", {}).items():
        t_idx = int(obs_idx)
        t_id = obs_dims[0]["values"][t_idx]["id"]
        rate = obs_val[0]
        if rate is not None:
            records.append({"dato": t_id, "rate": float(rate), **meta})

df = pd.DataFrame(records)
df["dato"] = pd.to_datetime(df["dato"])
df["valuta"] = df["BASE_CUR"]

# Keep relevant columns and pivot to wide format
meta_cols = [c for c in ["FREQ", "BASE_CUR", "QUOTE_CUR", "TENOR"] if c in df.columns]
df = df[["dato", "valuta", "rate"] + meta_cols].sort_values(["valuta", "dato"])
df_pivot = df.pivot(index="dato", columns="valuta", values="rate").sort_index()

print("Parsed OK → shape:", df.shape, "→ pivot:", df_pivot.shape)

# ------------------------------------------------------------
# 2. Filter out incomplete currency series (based on coverage)
# ------------------------------------------------------------
exclude = ["KRW", "SGD", "THB", "TWD", "PHP", "MYR", "I44",
           "IDR", "PLN", "CZK", "HUF", "HKD", "EUR"]
df_clean = df[~df["valuta"].isin(exclude)].copy()
df_pivot_clean = df_pivot.drop(columns=[c for c in exclude if c in df_pivot.columns], errors="ignore")

# ------------------------------------------------------------
# 3. Fill calendar gaps and weekends (daily frequency)
# ------------------------------------------------------------
use_calendar_days = True
noise_alpha = 0.05
noise_window = 30
rng_seed = 42

start, end = df_pivot_clean.index.min(), df_pivot_clean.index.max()
idx = pd.date_range(start, end, freq="D") if use_calendar_days else pd.bdate_range(start, end)
df_reindexed = df_pivot_clean.reindex(idx)

# Time interpolation and boundary filling
df_interp = df_reindexed.interpolate(method="time").ffill().bfill()

# Add small synthetic noise to imputed values
np.random.seed(rng_seed)
if noise_alpha > 0:
    rolling_std = df_reindexed.ffill().bfill().rolling(noise_window, min_periods=5).std()
    global_std = df_reindexed.stack().std()
    rolling_std = rolling_std.fillna(global_std)
    noise = pd.DataFrame(
        np.random.normal(0, 1, size=df_interp.shape),
        index=df_interp.index, columns=df_interp.columns
    ) * (noise_alpha * rolling_std)
    df_filled = df_interp + noise
else:
    df_filled = df_interp.copy()

df_filled = df_filled.clip(lower=0).astype(float)
print("Filled FX matrix:", df_filled.shape)

# ------------------------------------------------------------
# 4. Synthetic macroeconomic & regime features
# ------------------------------------------------------------
idx = df_filled.index
n = len(idx)
t_years = np.arange(n) / 365.25
rng = np.random.default_rng(123)

def zscore(x):
    s = pd.Series(x, index=idx, dtype=float)
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)

# Harmonic cycles
macro_cycle = (0.6 * np.sin(2*np.pi*(1/8)*t_years)
              + 0.3 * np.sin(2*np.pi*(1/2.5)*t_years + 0.7)
              + 0.1 * np.sin(2*np.pi*(1/0.5)*t_years + 1.3))

# Chirp (frequency increases over time)
f0, f1 = 1/10, 1/2
k = (f1 - f0) / max(t_years.max(), 1e-12)
macro_chirp = np.sin(2*np.pi*(f0*t_years + 0.5*k*t_years**2))

# Markov regime process (two states)
p00, p11 = 0.985, 0.975
P = np.array([[p00, 1-p00], [1-p11, p11]])
states = np.zeros(n, dtype=int)
for i in range(1, n):
    states[i] = rng.choice([0,1], p=P[states[i-1]])
regime_mean, regime_sigma = [-0.4, 0.6], [0.10, 0.25]
macro_regime = regime_mean[states] + rng.standard_normal(n)*regime_sigma[states]

# Combine synthetic macro features
macro = pd.DataFrame({
    "macro_cycle": zscore(macro_cycle),
    "macro_chirp": zscore(macro_chirp),
    "macro_regime": zscore(macro_regime),
}, index=idx)

# ------------------------------------------------------------
# 5. Technical features from FX prices
# ------------------------------------------------------------
logp = np.log(df_filled.clip(lower=1e-9))
rets = logp.diff().fillna(0.0).add_suffix("_ret")

def rolling_stats(X, w):
    return pd.concat([
        X.rolling(w).mean().add_suffix(f"_ma{w}"),
        X.rolling(w).std().add_suffix(f"_vol{w}")
    ], axis=1)

feat_20 = rolling_stats(rets, 20)
feat_60 = rolling_stats(rets, 60)
feat_252 = rolling_stats(rets, 252)

# PCA factors
pca = PCA(n_components=min(5, df_filled.shape[1]), random_state=123)
pca_scores = pca.fit_transform(df_filled.ffill().bfill())
pca_df = pd.DataFrame(pca_scores, index=idx,
                      columns=[f"pca_factor_{i+1}" for i in range(pca.n_components_)])

# Fourier and calendar features
def fourier_terms(index, K=[1,2,3,4], period=365.25):
    x = np.arange(len(index))
    ft = {}
    for k in K:
        ft[f"sin_{k}"] = np.sin(2*np.pi*k*x/period)
        ft[f"cos_{k}"] = np.cos(2*np.pi*k*x/period)
    return pd.DataFrame(ft, index=index)

fourier = fourier_terms(idx)
calendar = pd.get_dummies(pd.DataFrame({
    "dow": idx.weekday,
    "month": idx.month,
    "is_month_end": idx.is_month_end.astype(int)
}), columns=["dow", "month"], drop_first=True)

# ------------------------------------------------------------
# 6. Merge all feature blocks into train_df
# ------------------------------------------------------------
train_df = pd.concat([
    df_filled, rets, feat_20, feat_60, feat_252,
    pca_df, fourier, macro, calendar
], axis=1).ffill().bfill().astype(float)

print("train_df ready → shape:", train_df.shape)
print(train_df.head(3))

# ------------------------------------------------------------
# 7. Save final dataset
# ------------------------------------------------------------
train_df.index.name = "dato"
train_df.to_parquet("train_df_timeseries.parquet", compression="snappy")
print("Saved as train_df_timeseries.parquet")
