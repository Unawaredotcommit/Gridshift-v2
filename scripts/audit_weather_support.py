
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridshift import config as C
from gridshift import data as D
from gridshift import features as F
from gridshift.simulator import TEMP_RANGE


def describe(series):
    return {
        "min": series.min(),
        "p01": series.quantile(0.01),
        "p05": series.quantile(0.05),
        "median": series.median(),
        "p95": series.quantile(0.95),
        "p99": series.quantile(0.99),
        "max": series.max(),
    }


def main():
    df = D.load_clean()
    feat = F.make_feature_frame(df)
    splits, _ = F.get_splits(df, feat)

    train = df.iloc[np.flatnonzero(splits["train"])]
    test = df.iloc[np.flatnonzero(splits["test"])]

    print("\nTRAINING WEATHER DISTRIBUTIONS")
    for col in ["tmpc", "relh"]:
        print(f"\n{col}:")
        print(pd.Series(describe(train[col])).round(2).to_string())

    print("\nTEST WEATHER DISTRIBUTIONS")
    for col in ["tmpc", "relh"]:
        print(f"\n{col}:")
        print(pd.Series(describe(test[col])).round(2).to_string())

    print("\nSCENARIO SUPPORT AUDIT")
    print("Outside range means outside training-period 1st–99th percentiles.")

    for delta in [-10, -8, -5, -2, 2, 5, 8, 10]:
        shifted = (test["tmpc"] + delta).clip(*TEMP_RANGE)
        lo, hi = train["tmpc"].quantile([0.01, 0.99])

        outside = ((shifted < lo) | (shifted > hi)).mean() * 100
        clipped = (
            ((test["tmpc"] + delta) < TEMP_RANGE[0]) |
            ((test["tmpc"] + delta) > TEMP_RANGE[1])
        ).mean() * 100

        print(
            f"Temperature {delta:+} C | "
            f"outside training P01-P99: {outside:.2f}% | "
            f"clipped: {clipped:.2f}%"
        )

    for delta in [-30, -15, 15, 30]:
        shifted = (test["relh"] + delta).clip(0, 100)
        lo, hi = train["relh"].quantile([0.01, 0.99])

        outside = ((shifted < lo) | (shifted > hi)).mean() * 100
        clipped = (
            ((test["relh"] + delta) < 0) |
            ((test["relh"] + delta) > 100)
        ).mean() * 100

        print(
            f"Humidity {delta:+} pp | "
            f"outside training P01-P99: {outside:.2f}% | "
            f"clipped: {clipped:.2f}%"
        )

    print("\nAudit complete.")


if __name__ == "__main__":
    main()
