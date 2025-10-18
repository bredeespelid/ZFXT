from __future__ import annotations

import io
import os
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://data.norges-bank.no/api/data/EXR/"
EUR_NOK_SERIES = "B.EUR.NOK.SP"
URL = (
    f"{BASE}{EUR_NOK_SERIES}?format=csv&startPeriod=2011-01-01&endPeriod=2011-12-31&locale=no&bom=include"
)


def create_session():
    retry_strategy = Retry(total=5, backoff_factor=0.6,
                           status_forcelist=(429, 500, 502, 503, 504),
                           allowed_methods=("GET",))
    s = requests.Session()
    s.headers.update({"User-Agent": "norgesbank-exr-fetch/1.0"})
    s.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    return s


def _detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    cols = {c.strip().upper(): c for c in df.columns}
    time_candidates = ["TIME_PERIOD", "TID", "TIME", "DATE", "OBS_TIME"]
    value_candidates = ["OBS_VALUE", "VALUE", "RATE", "OBSVALUE", "OBSERVATION_VALUE"]
    time_col = None
    val_col = None
    for k in time_candidates:
        if k in cols:
            time_col = cols[k]
            break
    for k in value_candidates:
        if k in cols:
            val_col = cols[k]
            break
    if time_col is None or val_col is None:
        raise KeyError(f"Could not detect time/value columns in Norges Bank CSV. Columns: {list(df.columns)}")
    return time_col, val_col


def fetch_range(start: str, end: str, locale: str = "no") -> pd.DataFrame:
    url = f"{BASE}{EUR_NOK_SERIES}?format=csv&startPeriod={start}&endPeriod={end}&locale={locale}&bom=include"
    return fetch_and_parse(url=url, cache_tag=f"eurnok_{start}_{end}")


def fetch_and_parse(url: str = URL, cache_tag: str = "eurnok_2011") -> pd.DataFrame:
    cache_dir = os.path.join("runs", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"norgesbank_{cache_tag}.csv")
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        # If cache already normalized, just ensure dtypes and return
        if set(["timestamp", "symbol", "price"]).issubset(df.columns):
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            return df[["timestamp", "symbol", "price"]].dropna()
    else:
        with create_session() as s:
            r = s.get(url, timeout=30)
            r.raise_for_status()
            content = r.content
        # Try parsing with Norwegian-conventional separators first
        tried = []
        for sep, dec in [(";", ","), (",", "."), (None, ".")]:
            try:
                if sep is None:
                    df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
                else:
                    df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", sep=sep, decimal=dec)
                time_col, val_col = _detect_columns(df)
                break
            except Exception as e:
                tried.append(f"sep={sep} dec={dec} err={e}")
                df = None  # type: ignore
        if df is None:
            raise RuntimeError("Failed to parse Norges Bank CSV; attempts: " + "; ".join(tried))
    # Normalize columns
    time_col, val_col = _detect_columns(df)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.rename(columns={time_col: "timestamp", val_col: "price"})
    df["symbol"] = "EURNOK"
    out = df[["timestamp", "symbol", "price"]]
    out.to_csv(cache_file, index=False)
    return out
