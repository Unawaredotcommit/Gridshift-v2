
"""Evaluate saved Ridge and LightGBM models on the validation split.

Does not retrain models or modify the existing test metrics.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridshift import config as C
from gridshift import data as D
from gridshift import features as F
from gridshift import metrics as M
from gridshift import models as MD


def main():
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Recreate the original prepared data and chronological splits.
    df = D.load_clean()
    feat = F.make_feature_frame(df)
    splits, dropped = F.get_splits(df, feat)

    print("Validation origins:", len(splits["val"]))
    print("Dropped origins:", dropped)

    # Build validation rows using the same feature pipeline.
    val = F.build_tabular(feat, splits["val"])

    print("Validation rows:", len(val))

    results = {}

    for name in ["ridge", "lightgbm"]:
        model = MD.load_model(name)
        y_pred = MD.predict(model, val)

        # Preserve the same structure used by the original test evaluator.
        pred = val[
            ["origin_time", "target_time", "horizon", "y"]
        ].copy()

        pred["y_pred"] = y_pred

        results[name] = {
            **M.regression_metrics(pred["y"], pred["y_pred"]),
            **M.peak_metrics(pred),
        }

        pred.to_csv(
            C.REPORTS_DIR / f"validation_predictions_{name}.csv",
            index=False,
        )

        M.by_horizon(pred).to_csv(
            C.REPORTS_DIR / f"validation_metrics_by_horizon_{name}.csv",
            index=False,
        )

        print(f"\n[{name}] validation:")
        print(json.dumps(results[name], indent=2))

    output = C.REPORTS_DIR / "metrics_validation.json"
    output.write_text(json.dumps(results, indent=2))

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
