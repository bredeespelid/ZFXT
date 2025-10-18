from __future__ import annotations

import pandas as pd


def to_freq(df: pd.DataFrame, freq: str, timestamp_col: str = "timestamp", value_col: str = "price", method: str = "ffill") -> pd.DataFrame:
    g = df.set_index(timestamp_col).sort_index()[[value_col]]
    out = g.asfreq(freq)
    if method == "ffill":
        out = out.ffill()
    elif method == "bfill":
        out = out.bfill()
    elif method == "interpolate":
        out = out.interpolate(limit_direction="both")
    out = out.reset_index()
    return out


def to_monthly(df: pd.DataFrame, timestamp_col: str = "timestamp", value_col: str = "price") -> pd.DataFrame:
    g = df.set_index(timestamp_col).sort_index()[[value_col]]
    out = g.resample("ME").last().dropna().reset_index()
    return out


def to_quarterly(df: pd.DataFrame, timestamp_col: str = "timestamp", value_col: str = "price") -> pd.DataFrame:
    g = df.set_index(timestamp_col).sort_index()[[value_col]]
    out = g.resample("QE").last().dropna().reset_index()
    return out
