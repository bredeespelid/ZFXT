from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE = "https://data.norges-bank.no/api/data/EXR/"


def _session() -> requests.Session:
    retry = Retry(total=5, backoff_factor=0.6, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    s = requests.Session()
    s.headers.update({"User-Agent": "fx-timesfm-nb/1.0"})
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _detect_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    cols = {c.strip().upper(): c for c in df.columns}
    time_candidates = ["TIME_PERIOD", "TID", "TIME", "DATE", "OBS_TIME"]
    value_candidates = ["OBS_VALUE", "VALUE", "RATE", "OBSVALUE", "OBSERVATION_VALUE"]
    cur_candidates = ["BASE_CUR", "CURRENCY"]
    t = next((cols[k] for k in time_candidates if k in cols), None)
    v = next((cols[k] for k in value_candidates if k in cols), None)
    c = next((cols[k] for k in cur_candidates if k in cols), None)
    if not (t and v and c):
        raise KeyError(f"Missing expected columns in NB CSV: {list(df.columns)}")
    return t, v, c


def fetch_fx_pair(base: str, quote: str, start: str, end: str, *, locale: str = "no", cache_dir: str = "runs/cache") -> pd.DataFrame:
    series = f"B.{base}.{quote}.SP"
    url = f"{BASE}{series}?format=csv&startPeriod={start}&endPeriod={end}&locale={locale}&bom=include"
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"nb_{base}{quote}_{start}_{end}.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        if set(["timestamp", "symbol", "price"]).issubset(df.columns):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            return df.dropna().sort_values("timestamp")
    with _session() as s:
        r = s.get(url, timeout=60)
        r.raise_for_status()
        content = r.content
    # Try multiple separators
    last_err: Optional[Exception] = None
    for sep, dec in [(";", ","), (",", "."), (None, ".")]:
        try:
            if sep is None:
                df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
            else:
                df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", sep=sep, decimal=dec)
            t, v, c = _detect_columns(df)
            df[t] = pd.to_datetime(df[t], utc=True, errors="coerce")
            df[v] = pd.to_numeric(df[v], errors="coerce")
            out = df[[t, v]].copy()
            out.columns = ["timestamp", "price"]
            out["symbol"] = f"{base}{quote}"
            out = out.dropna().sort_values("timestamp")[ ["timestamp", "symbol", "price"] ]
            out.to_csv(cache, index=False)
            return out
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to parse Norges Bank CSV for {base}/{quote}: {last_err}")


def to_long_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure timestamp,symbol,price long schema
    cols = set(df.columns)
    if {"timestamp", "symbol", "price"}.issubset(cols):
        return df[["timestamp", "symbol", "price"]].copy()
    raise ValueError("Input must contain timestamp,symbol,price columns")
