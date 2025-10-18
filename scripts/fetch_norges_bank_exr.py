from __future__ import annotations

import io
import os
import sys
from typing import Literal

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


EXR_URL = (
    "https://data.norges-bank.no/api/data/EXR/"
    "B.USD+AUD+BDT+BYN+BRL+GBP+BGN+DKK+EUR+PHP+HKD+XDR+I44+INR+IDR+TWI+ISK+JPY+CAD+"
    "CNY+HRK+MYR+MXN+MMK+NZD+ILS+RON+TWD+PKR+PLN+RUB+SGD+CHF+SEK+ZAR+KRW+THB+CZK+TRY+HUF+VND"
    ".NOK.SP?format=csv&startPeriod=1970-01-01&endPeriod=1999-12-31&locale=no&bom=include"
)


def create_session() -> requests.Session:
    retry_strategy = Retry(
        total=5,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "nb-exr-fx-timesfm/1.0"})
    session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    return session


def fetch_csv_bytes(url: str, timeout_s: int = 30) -> bytes:
    with create_session() as s:
        r = s.get(url, timeout=timeout_s)
        r.raise_for_status()
        return r.content


def read_norgesbank_csv(
    csv_bytes: bytes,
    locale: Literal["no", "en"] = "no",
    use_pyarrow: bool = True,
) -> pd.DataFrame:
    dtype_backend = "pyarrow" if use_pyarrow else "numpy"

    if locale == "no":
        df = pd.read_csv(
            io.BytesIO(csv_bytes),
            encoding="utf-8-sig",
            sep=";",
            decimal=",",
            dtype_backend=dtype_backend,
        )
    else:
        df = pd.read_csv(
            io.BytesIO(csv_bytes),
            encoding="utf-8-sig",
            dtype_backend=dtype_backend,
        )

    df.columns = [c.strip().upper() for c in df.columns]

    if "TIME_PERIOD" in df.columns:
        df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")
    if "OBS_VALUE" in df.columns:
        df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    return df


def pivot_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    required = {"TIME_PERIOD", "OBS_VALUE"}
    if not required.issubset(df.columns):
        raise ValueError("Expected TIME_PERIOD and OBS_VALUE in dataframe.")
    key = "BASE_CUR" if "BASE_CUR" in df.columns else ("CURRENCY" if "CURRENCY" in df.columns else None)
    if key is None:
        raise ValueError("Could not find currency column (BASE_CUR/CURRENCY)")
    wide = (
        df.pivot_table(index="TIME_PERIOD", columns=key, values="OBS_VALUE", aggfunc="mean")
        .sort_index()
        .rename_axis(None, axis=1)
        .reset_index()
    )
    return wide


def long_to_fx_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Map to timestamp,symbol,price schema
    key = "BASE_CUR" if "BASE_CUR" in df.columns else ("CURRENCY" if "CURRENCY" in df.columns else None)
    out = df[["TIME_PERIOD", key, "OBS_VALUE"]].copy()
    out.columns = ["timestamp", "symbol", "price"]
    # drop NaNs and sort
    out = out.dropna(subset=["timestamp", "price"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return out


def temporal_split(df: pd.DataFrame, timestamp_col: str = "timestamp", ratios=(0.8, 0.1, 0.1)):
    # Per symbol chronological split
    outs = {"train": [], "val": [], "test": []}
    for sym, g in df.groupby("symbol"):
        g = g.sort_values(timestamp_col)
        n = len(g)
        i1 = int(n * ratios[0])
        i2 = int(n * (ratios[0] + ratios[1]))
        outs["train"].append(g.iloc[:i1])
        outs["val"].append(g.iloc[i1:i2])
        outs["test"].append(g.iloc[i2:])
    return {k: pd.concat(v).reset_index(drop=True) for k, v in outs.items()}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="data")
    parser.add_argument("--start", type=str, default="1970-01-01")
    parser.add_argument("--end", type=str, default="1999-12-31")
    parser.add_argument("--locale", type=str, default="no", choices=["no", "en"])
    args = parser.parse_args()

    # Build URL dynamically to let users change date range
    url = (
        "https://data.norges-bank.no/api/data/EXR/"
        "B.USD+AUD+BDT+BYN+BRL+GBP+BGN+DKK+EUR+PHP+HKD+XDR+I44+INR+IDR+TWI+ISK+JPY+CAD+"
        "CNY+HRK+MYR+MXN+MMK+NZD+ILS+RON+TWD+PKR+PLN+RUB+SGD+CHF+SEK+ZAR+KRW+THB+CZK+TRY+HUF+VND"
        f".NOK.SP?format=csv&startPeriod={args.start}&endPeriod={args.end}&locale={args.locale}&bom=include"
    )

    csv_bytes = fetch_csv_bytes(url)
    df = read_norgesbank_csv(csv_bytes, locale=args.locale, use_pyarrow=True)

    # Save raw long file
    os.makedirs(args.outdir, exist_ok=True)
    long_path = os.path.join(args.outdir, "norges_bank_exr_long.csv")
    df.to_csv(long_path, index=False, encoding="utf-8")
    try:
        df.to_parquet(os.path.join(args.outdir, "norges_bank_exr_long.parquet"), index=False)
    except Exception as e:
        print("Skipped Parquet (long):", e, file=sys.stderr)

    # Pivot to wide for inspection (optional)
    try:
        wide = pivot_to_wide(df)
        wide.to_csv(os.path.join(args.outdir, "norges_bank_exr_wide.csv"), index=False, encoding="utf-8")
        try:
            wide.to_parquet(os.path.join(args.outdir, "norges_bank_exr_wide.parquet"), index=False)
        except Exception as e:
            print("Skipped Parquet (wide):", e, file=sys.stderr)
    except Exception as e:
        print("Skipped pivot to wide:", e, file=sys.stderr)

    # Convert to model schema and split
    fx_long = long_to_fx_schema(df)
    splits = temporal_split(fx_long)
    for k, d in splits.items():
        d.to_csv(os.path.join(args.outdir, f"{k}.csv"), index=False, encoding="utf-8")
        print(f"wrote {k}.csv rows={len(d)}")


if __name__ == "__main__":
    main()
