
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridshift import data as D
from gridshift import features as F
from gridshift.simulator import TEMP_RANGE


def sequence_features(weather):
    """
    Summarize each 24-hour weather sequence using eight features:
    mean, minimum and maximum temperature;
    mean, minimum and maximum humidity;
    maximum hourly temperature and humidity changes.
    """
    return np.column_stack([
        weather[:, :, 0].mean(axis=1),
        weather[:, :, 0].min(axis=1),
        weather[:, :, 0].max(axis=1),
        weather[:, :, 1].mean(axis=1),
        weather[:, :, 1].min(axis=1),
        weather[:, :, 1].max(axis=1),
        np.abs(np.diff(weather[:, :, 0], axis=1)).max(axis=1),
        np.abs(np.diff(weather[:, :, 1], axis=1)).max(axis=1),
    ])


def make_sequences(indices, source):
    """Extract the 24 hours immediately following each forecast origin."""
    sequences = []

    for idx in indices:
        idx = int(idx)

        if idx + 24 >= len(source):
            continue

        window = source[idx + 1:idx + 25]

        if len(window) == 24 and np.isfinite(window).all():
            sequences.append(window)

    if not sequences:
        return np.empty((0, 24, 2))

    return np.asarray(sequences)


def main():
    print("Loading GridShift data...")

    df = D.load_clean()
    feat = F.make_feature_frame(df)
    splits, _ = F.get_splits(df, feat)

    # IMPORTANT:
    # These splits contain integer row positions, not timestamps
    # and not Boolean masks.
    train_indices = splits["train"]
    test_indices = splits["test"]

    weather_array = df[["tmpc", "relh"]].to_numpy()

    train_sequences = make_sequences(train_indices, weather_array)
    test_sequences = make_sequences(test_indices, weather_array)

    print(f"Training forecast origins: {len(train_indices):,}")
    print(f"Test forecast origins: {len(test_indices):,}")
    print(f"Valid training sequences: {len(train_sequences):,}")
    print(f"Valid test sequences: {len(test_sequences):,}")

    if len(train_sequences) == 0 or len(test_sequences) == 0:
        raise RuntimeError(
            "No valid 24-hour sequences found. Check data and forecast origins."
        )

    # Summarize the 24-hour sequences.
    train_summary = sequence_features(train_sequences)
    test_summary = sequence_features(test_sequences)

    # Standardize using training-period statistics only.
    mean = train_summary.mean(axis=0)
    std = train_summary.std(axis=0)
    std[std == 0] = 1.0

    train_scaled = (train_summary - mean) / std
    test_scaled = (test_summary - mean) / std

    # Build nearest-neighbour reference from training sequences.
    tree = cKDTree(train_scaled)

    train_distances, _ = tree.query(train_scaled, k=2)
    threshold = np.quantile(train_distances[:, 1], 0.99)

    print("\n24-HOUR WEATHER SEQUENCE AUDIT")
    print(f"Training sequences: {len(train_sequences):,}")
    print(f"Test sequences: {len(test_sequences):,}")
    print(
        "Training 99th-percentile nearest-neighbour threshold: "
        f"{threshold:.4f}"
    )

    # Check unmodified test sequences.
    baseline_distances, _ = tree.query(test_scaled, k=1)

    print(
        "\nBaseline test sequences outside support: "
        f"{(baseline_distances > threshold).mean() * 100:.2f}%"
    )

    print("\nSCENARIO SUPPORT RESULTS")
    print("Temperature | Humidity | Outside sequence support | Humidity clipped")

    # Test moderate temperature and humidity changes.
    for temp_delta in [-5, -2, 0, 2, 5]:
        for humidity_delta in [-5, 0, 5]:
            modified = test_sequences.copy()

            # Apply temperature adjustment.
            modified[:, :, 0] = (
                modified[:, :, 0] + temp_delta
            ).clip(*TEMP_RANGE)

            # Apply humidity adjustment.
            raw_humidity = modified[:, :, 1] + humidity_delta
            modified[:, :, 1] = raw_humidity.clip(0, 100)

            # Calculate percentage of humidity values clipped.
            clipped_pct = (
                ((raw_humidity < 0) | (raw_humidity > 100)).mean() * 100
            )

            # Compare modified sequences against training sequences.
            modified_summary = sequence_features(modified)
            modified_scaled = (modified_summary - mean) / std

            distances, _ = tree.query(modified_scaled, k=1)
            outside_pct = (distances > threshold).mean() * 100

            print(
                f"{temp_delta:+5} C      | "
                f"{humidity_delta:+5} pp   | "
                f"{outside_pct:8.2f}%               | "
                f"{clipped_pct:8.2f}%"
            )

    print("\nAudit complete.")
    print(
        "Note: This compares eight 24-hour summary statistics. "
        "It does not prove full meteorological realism."
    )


if __name__ == "__main__":
    main()
