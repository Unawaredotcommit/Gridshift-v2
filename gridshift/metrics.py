"""Forecast accuracy metrics (computed from real predictions only)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_res, ss_tot = np.sum(err ** 2), np.sum((y_true - y_true.mean()) ** 2)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(np.mean(np.abs(err) / np.abs(y_true)) * 100),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(len(y_true)),
    }


def peak_metrics(pred_df: pd.DataFrame) -> dict:
    """Per-forecast-window peak error. `pred_df` needs origin_time, y, y_pred (24 rows/origin)."""
    g = pred_df.groupby("origin_time")
    peak_true, peak_pred = g["y"].max(), g["y_pred"].max()
    e = peak_pred - peak_true
    return {"peak_MAE": float(e.abs().mean()), "peak_bias": float(e.mean()),
            "peak_MAPE_%": float((e.abs() / peak_true).mean() * 100)}


def by_horizon(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, d in pred_df.groupby("horizon"):
        m = regression_metrics(d["y"], d["y_pred"])
        rows.append({"horizon": h, "MAE": m["MAE"], "RMSE": m["RMSE"], "MAPE_%": m["MAPE_%"]})
    return pd.DataFrame(rows)
