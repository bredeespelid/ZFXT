from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    rw_df: pd.DataFrame,
    title: str,
    save_path: str,
    actuals_df: Optional[pd.DataFrame] = None,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(history_df["timestamp"], history_df["price"], label="History", color="black")
    plt.plot(forecast_df["timestamp"], forecast_df["forecast"], label="Model Forecast", color="tab:blue")
    plt.plot(rw_df["timestamp"], rw_df["rw"], label="RW Baseline", color="tab:orange", linestyle="--")
    if actuals_df is not None and len(actuals_df) > 0:
        # Only plot actuals beyond the last historical timestamp if possible
        t_last = pd.to_datetime(history_df["timestamp"]).max()
        actuals_future = actuals_df[actuals_df["timestamp"] > t_last]
        if len(actuals_future) > 0:
            plt.plot(
                actuals_future["timestamp"],
                actuals_future["price"],
                label="Actuals",
                color="tab:green",
                linestyle=":",
            )
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    try:
        plt.show()
    except Exception:
        pass
