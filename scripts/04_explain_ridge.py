
"""Step 4 - Ridge explainability and Ridge vs LightGBM comparison."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from gridshift import config as C
from gridshift import data as D
from gridshift import explain as E
from gridshift import features as F
from gridshift import models as MD
from gridshift.metrics import regression_metrics


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = root / "artifacts" / "explain_ridge"
    out.mkdir(parents=True, exist_ok=True)

    print("Loading GridShift data...")

    df = D.load_clean()
    feat = F.make_feature_frame(df)
    splits, _ = F.get_splits(df, feat)

    test = F.build_tabular(feat, splits["test"])

    print(f"Test rows: {len(test):,}")

    print("\nLoading trained models...")

    ridge = MD.load_model("ridge")
    lightgbm = MD.load_model("lightgbm")

    # Use the same test sample for both models.
    n_samples = 5000 if args.quick else 20000
    n_repeats = 2 if args.quick else 5

    sample = test.sample(
        min(n_samples, len(test)),
        random_state=C.SEED,
    )

    X = sample[F.TABULAR_FEATURES]
    y = sample["y"].to_numpy()

    # ---------------------------------------------------------
    # 1. Baseline model metrics
    # ---------------------------------------------------------

    print("\nCalculating baseline metrics...")

    metrics_rows = []

    for name, model in [
        ("Ridge", ridge),
        ("LightGBM", lightgbm),
    ]:
        predictions = MD.predict(model, sample)

        metrics = regression_metrics(y, predictions)
        metrics["model"] = name

        metrics_rows.append(metrics)

        print(
            f"{name}: RMSE={metrics['RMSE']:.2f} MW | "
            f"MAE={metrics['MAE']:.2f} MW"
        )

    pd.DataFrame(metrics_rows).to_csv(
        out / "baseline_metrics.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 2. Ridge standardized coefficients
    # ---------------------------------------------------------

    print("\nExtracting Ridge coefficients...")

    ridge_estimator = ridge.named_steps["ridge"]
    scaler = ridge.named_steps["scale"]

    coefficients = pd.DataFrame({
        "feature": F.TABULAR_FEATURES,
        "coefficient_mw_per_1sd": ridge_estimator.coef_,
        "abs_coefficient": np.abs(ridge_estimator.coef_),
        "original_feature_mean": scaler.mean_,
        "original_feature_scale": scaler.scale_,
    })

    coefficients = coefficients.sort_values(
        "abs_coefficient",
        ascending=False,
    )

    coefficients.to_csv(
        out / "ridge_coefficients.csv",
        index=False,
    )

    print("\nTop 10 Ridge coefficients:")
    print(
        coefficients[
            ["feature", "coefficient_mw_per_1sd"]
        ].head(10).to_string(index=False)
    )

    # ---------------------------------------------------------
    # 3. Grouped coefficient summaries
    # ---------------------------------------------------------

    print("\nCalculating grouped coefficient summaries...")

    group_rows = []

    coefficient_map = coefficients.set_index("feature")[
        "coefficient_mw_per_1sd"
    ]

    for group_name, columns in E.SHAP_GROUPS.items():

        available = [
            col for col in columns
            if col in coefficient_map.index
        ]

        values = coefficient_map.loc[available].to_numpy()

        if len(values) == 0:
            continue

        group_rows.append({
            "group": group_name,
            "n_features": len(values),
            "sum_abs_coefficients": np.abs(values).sum(),
            "mean_abs_coefficient": np.abs(values).mean(),
            "l2_coefficient_norm": np.linalg.norm(values),
            "signed_coefficient_sum": values.sum(),
        })

    group_coefficients = pd.DataFrame(group_rows)

    group_coefficients = group_coefficients.sort_values(
        "sum_abs_coefficients",
        ascending=False,
    )

    group_coefficients.to_csv(
        out / "ridge_grouped_coefficients.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 4. Individual feature permutation importance
    # ---------------------------------------------------------

    print("\nCalculating individual feature permutation importance...")

    individual_groups = {
        col: [col]
        for col in F.TABULAR_FEATURES
    }

    individual_results = []

    for name, model in [
        ("Ridge", ridge),
        ("LightGBM", lightgbm),
    ]:

        print(f"  Processing {name}...")

        result = E.permutation_importance(
            model,
            sample,
            y,
            individual_groups,
            n_repeats=n_repeats,
            seed=C.SEED,
        )

        result["model"] = name
        individual_results.append(result)

    individual_comparison = pd.concat(
        individual_results,
        ignore_index=True,
    )

    individual_comparison.to_csv(
        out / "individual_permutation_comparison.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 5. Grouped permutation importance
    # ---------------------------------------------------------

    print("\nCalculating grouped permutation importance...")

    grouped_results = []

    for name, model in [
        ("Ridge", ridge),
        ("LightGBM", lightgbm),
    ]:

        print(f"  Processing {name}...")

        result = E.permutation_importance(
            model,
            sample,
            y,
            E.PERM_GROUPS,
            n_repeats=n_repeats,
            seed=C.SEED,
        )

        result["model"] = name
        grouped_results.append(result)

    grouped_comparison = pd.concat(
        grouped_results,
        ignore_index=True,
    )

    grouped_comparison.to_csv(
        out / "grouped_permutation_comparison.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 6. Final comparison table
    # ---------------------------------------------------------

    print("\nPreparing model comparison...")

    comparison = grouped_comparison.pivot(
        index="name",
        columns="model",
        values="rmse_increase_mw",
    ).reset_index()

    comparison.columns.name = None

    if "Ridge" in comparison.columns and "LightGBM" in comparison.columns:
        comparison["LightGBM_minus_Ridge"] = (
            comparison["LightGBM"] - comparison["Ridge"]
        )

    comparison.to_csv(
        out / "model_group_importance_comparison.csv",
        index=False,
    )

    print("\nGrouped permutation comparison:")
    print(comparison.to_string(index=False))

    print("\nAll Ridge explainability analyses completed.")
    print("Results saved to:", out)


if __name__ == "__main__":
    main()
