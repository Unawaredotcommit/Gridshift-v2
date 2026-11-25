import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridshift import data as D
from gridshift import features as F
from gridshift.simulator import TEMP_RANGE


def main():
    df = D.load_clean()
    feat = F.make_feature_frame(df)
    splits, _ = F.get_splits(df, feat)

    train = df.iloc[np.flatnonzero(splits["train"])][["tmpc", "relh"]].dropna()
    test = df.iloc[np.flatnonzero(splits["test"])][["tmpc", "relh"]].dropna()

    # Standardize both variables using training-period statistics.
    mean = train.mean()
    std = train.std()

    train_scaled = (train - mean) / std
    test_scaled = (test - mean) / std

    # Measure each training observation's distance to its nearest
    # OTHER training observation.
    tree = cKDTree(train_scaled.to_numpy())
    train_distances, _ = tree.query(train_scaled.to_numpy(), k=2)

    # The second returned neighbour is the nearest other observation.
    threshold = np.quantile(train_distances[:, 1], 0.99)

    print("\nJOINT TEMPERATURE-HUMIDITY SUPPORT AUDIT")
    print(f"Training observations: {len(train):,}")
    print(f"Test observations: {len(test):,}")
    print(f"Training 99th-percentile nearest-neighbour distance: {threshold:.4f}")

    # Baseline test support.
    baseline_distances, _ = tree.query(test_scaled.to_numpy(), k=1)

    print(
        f"\nBaseline test outside joint-support threshold: "
        f"{(baseline_distances > threshold).mean() * 100:.2f}%"
    )

    print("\nSCENARIO RESULTS")
    print(
        "Temperature | Humidity | Outside joint support | "
        "Humidity clipped"
    )

    for temp_delta in [-5, -2, 0, 2, 5]:
        for humidity_delta in [-10, -5, 0, 5, 10]:
            shifted_temp = (
                test["tmpc"] + temp_delta
            ).clip(*TEMP_RANGE)

            raw_humidity = test["relh"] + humidity_delta
            shifted_humidity = raw_humidity.clip(0, 100)

            shifted = pd.DataFrame({
                "tmpc": shifted_temp,
                "relh": shifted_humidity,
            })

            shifted_scaled = (shifted - mean) / std
            distances, _ = tree.query(shifted_scaled.to_numpy(), k=1)

            outside = (distances > threshold).mean() * 100

            clipped = (
                (raw_humidity < 0) | (raw_humidity > 100)
            ).mean() * 100

            print(
                f"{temp_delta:+5} C      | "
                f"{humidity_delta:+5} pp   | "
                f"{outside:7.2f}%             | "
                f"{clipped:7.2f}%"
            )

    print("\nAudit complete.")
    print(
        "Note: This checks joint hourly temperature-humidity support, "
        "not 24-hour sequence realism."
    )


if __name__ == "__main__":
    main()