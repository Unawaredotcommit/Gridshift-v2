
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from gridshift import config as C
from gridshift import data as D
from gridshift.simulator import TEMP_RANGE


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "artifacts" / "hourly_clean.csv"

TEMP_DELTAS = [-5, -2, 0, 2, 5]
HUMIDITY_DELTAS = [-5, 0, 5]
QUANTILE = 0.99


def sequence_features(sequence):
    """Summarise one 24-hour temperature/RH sequence."""

    temp = sequence[:, 0]
    humidity = sequence[:, 1]

    return [
        np.mean(temp),
        np.min(temp),
        np.max(temp),
        np.mean(humidity),
        np.min(humidity),
        np.max(humidity),
        np.max(np.abs(np.diff(temp))),
        np.max(np.abs(np.diff(humidity))),
    ]


def build_sequences(weather, origins):
    """Build T+1 to T+24 weather sequences."""

    features = []
    valid_origins = []

    for idx in origins:
        sequence = weather[idx + 1:idx + 25]

        if sequence.shape != (24, 2):
            continue

        if not np.isfinite(sequence).all():
            continue

        features.append(sequence_features(sequence))
        valid_origins.append(idx)

    return (
        np.asarray(features, dtype=float),
        np.asarray(valid_origins, dtype=int),
    )


def outside_rate(tree, values, threshold):
    """Percentage of sequences outside calibrated support."""

    distances, _ = tree.query(values, k=1)

    return 100 * np.mean(distances > threshold)


def main():

    print("Loading GridShift data...")

    df = D.load_clean()

    valid_origins = D.valid_origin_positions(df)

    splits = D.split_origins(
        df,
        valid_origins,
        horizon=C.HORIZON,
        stride=C.ORIGIN_STRIDE,
    )

    train_origins = splits["train"]
    val_origins = splits["val"]
    test_origins = splits["test"]

    weather = df[["tmpc", "relh"]].to_numpy(dtype=float)

    print(f"Training origins:   {len(train_origins):,}")
    print(f"Validation origins: {len(val_origins):,}")
    print(f"Test origins:       {len(test_origins):,}")

    # Build the original weather sequences.
    train, _ = build_sequences(weather, train_origins)
    validation, _ = build_sequences(weather, val_origins)
    test, test_indices = build_sequences(weather, test_origins)

    print("\nValid sequences:")
    print(f"Training:   {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Test:       {len(test):,}")

    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError(
            "One or more splits produced no valid sequences."
        )

    # Standardise using training statistics only.
    mean = train.mean(axis=0)
    std = train.std(axis=0)

    std[std == 0] = 1.0

    train_scaled = (train - mean) / std
    val_scaled = (validation - mean) / std
    test_scaled = (test - mean) / std

    # Training sequences form the reference distribution.
    tree = cKDTree(train_scaled)

    # Calibrate the distance threshold using validation data.
    val_distances, _ = tree.query(val_scaled, k=1)

    threshold = np.quantile(
        val_distances,
        QUANTILE,
    )

    validation_rate = 100 * np.mean(
        val_distances > threshold
    )

    baseline_rate = outside_rate(
        tree,
        test_scaled,
        threshold,
    )

    print("\n" + "=" * 65)
    print("VALIDATION-CALIBRATED SEQUENCE SUPPORT AUDIT")
    print("=" * 65)

    print(f"\nCalibration quantile: {QUANTILE:.0%}")
    print(f"Distance threshold:   {threshold:.4f}")
    print(f"Validation outside:   {validation_rate:.2f}%")
    print(f"Baseline test outside:{baseline_rate:7.2f}%")

    print("\nScenario results")
    print("-" * 65)

    print(
        f"{'Temp °C':>9} "
        f"{'RH pp':>9} "
        f"{'Outside %':>12} "
        f"{'Change pp':>12} "
        f"{'RH clipped %':>14}"
    )

    print("-" * 65)

    # Each forecast window is modified independently.
    # This prevents overlapping windows from receiving
    # repeated modifications.

    for temp_delta in TEMP_DELTAS:

        for humidity_delta in HUMIDITY_DELTAS:

            scenario_features = []

            clipped_count = 0
            total_count = 0

            for idx in test_indices:

                # Always start from original weather.
                sequence = weather[idx + 1:idx + 25].copy()

                original_temp = sequence[:, 0]
                original_humidity = sequence[:, 1]

                # Apply temperature adjustment.
                modified_temp = np.clip(
                    original_temp + temp_delta,
                    TEMP_RANGE[0],
                    TEMP_RANGE[1],
                )

                # Apply humidity adjustment.
                raw_humidity = (
                    original_humidity + humidity_delta
                )

                modified_humidity = np.clip(
                    raw_humidity,
                    0,
                    100,
                )

                clipped_count += np.count_nonzero(
                    raw_humidity != modified_humidity
                )

                total_count += len(raw_humidity)

                sequence[:, 0] = modified_temp
                sequence[:, 1] = modified_humidity

                scenario_features.append(
                    sequence_features(sequence)
                )

            scenario_features = np.asarray(
                scenario_features,
                dtype=float,
            )

            scenario_scaled = (
                scenario_features - mean
            ) / std

            rate = outside_rate(
                tree,
                scenario_scaled,
                threshold,
            )

            change = rate - baseline_rate

            clipped_pct = (
                100 * clipped_count / total_count
                if total_count
                else 0
            )

            print(
                f"{temp_delta:>9.0f} "
                f"{humidity_delta:>9.0f} "
                f"{rate:>11.2f}% "
                f"{change:>+11.2f} "
                f"{clipped_pct:>13.2f}%"
            )

    print("-" * 65)
    print("\nAudit complete.")


if __name__ == "__main__":
    main()
