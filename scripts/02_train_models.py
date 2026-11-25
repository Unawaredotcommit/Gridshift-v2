"""Step 2 - train Ridge + LightGBM (and optionally CNN-LSTM-Attention); evaluate on the test split.

Usage:
    python scripts/02_train_models.py                # Ridge + LightGBM
    python scripts/02_train_models.py --deep         # also CNN-LSTM-Attention (needs tensorflow)
    python scripts/02_train_models.py --deep-only
Evaluation protocol: train on `train` origins, early-stop / pick Ridge alpha on `val` (2023), and touch `test`
(2024) only for the final predictions/metrics below. Metrics are computed over ALL (origin, horizon) rows of
OVERLAPPING 24h windows (an origin every ORIGIN_STRIDE hours) -- see GRIDSHIFT_README.md; horizon-wise
metrics are written separately to metrics_by_horizon_<model>.csv.
Outputs (artifacts/): models/*.joblib|*.keras, reports/metrics.json, reports/test_predictions_<model>.csv,
                      reports/metrics_by_horizon_<model>.csv, reports/ridge_alpha_search.csv
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from gridshift import config as C
from gridshift import data as D
from gridshift import features as F
from gridshift import metrics as M
from gridshift import models as MD


def eval_and_dump(name, test_rows, y_pred, metrics):
    pred = test_rows[["origin_time", "target_time", "horizon", "y"]].copy()
    pred["y_pred"] = y_pred
    pred.to_csv(C.REPORTS_DIR / f"test_predictions_{name}.csv", index=False)
    M.by_horizon(pred).to_csv(C.REPORTS_DIR / f"metrics_by_horizon_{name}.csv", index=False)
    metrics[name] = {**M.regression_metrics(pred["y"], pred["y_pred"]), **M.peak_metrics(pred)}
    print(f"[{name}] test:", json.dumps(metrics[name], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--deep-only", action="store_true")
    args = ap.parse_args()

    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = D.load_clean()
    feat = F.make_feature_frame(df)
    # One shared set of forecast origins for ALL models (tabular + CNN-LSTM-Attention) -> comparable targets.
    splits, dropped = F.get_splits(df, feat)
    print({k: len(v) for k, v in splits.items()}, "origins; dropped by completeness filter:", dropped)

    mpath = C.REPORTS_DIR / "metrics.json"
    metrics = json.loads(mpath.read_text()) if mpath.exists() else {}

    if not args.deep_only:
        tab = {k: F.build_tabular(feat, v) for k, v in splits.items()}
        assert all(not v[F.TABULAR_FEATURES + ["y"]].isna().any().any() for v in tab.values())
        print({k: len(v) for k, v in tab.items()}, "tabular rows")

        t0 = time.time()
        ridge, alpha_log = MD.fit_ridge(tab["train"], tab["val"])
        alpha_log.to_csv(C.REPORTS_DIR / "ridge_alpha_search.csv", index=False)
        MD.save_model(ridge, "ridge")
        eval_and_dump("ridge", tab["test"], MD.predict(ridge, tab["test"]), metrics)
        print(f"ridge done in {time.time()-t0:.0f}s")

        t0 = time.time()
        lgbm = MD.fit_lgbm(tab["train"], tab["val"])
        print("LightGBM best_iteration:", lgbm.best_iteration_)
        MD.save_model(lgbm, "lightgbm")
        metrics["lightgbm_best_iteration"] = int(lgbm.best_iteration_)
        eval_and_dump("lightgbm", tab["test"], MD.predict(lgbm, tab["test"]), metrics)
        print(f"lightgbm done in {time.time()-t0:.0f}s")

    if args.deep or args.deep_only:
        from gridshift import deep
        Xtr, ytr = F.build_sequences(feat, splits["train"])
        Xva, yva = F.build_sequences(feat, splits["val"])
        Xte, yte = F.build_sequences(feat, splits["test"])
        model, sc, hist = deep.train(Xtr, ytr, Xva, yva)
        yp = deep.predict(model, sc, Xte)
        n, H = yp.shape
        rows = pd.DataFrame({
            "origin_time": np.repeat(feat.index[splits["test"]].to_numpy(), H),
            "horizon": np.tile(np.arange(1, H + 1), n),
            "y": yte.reshape(-1)})
        rows["target_time"] = rows["origin_time"] + pd.to_timedelta(rows["horizon"], unit="h")
        eval_and_dump("cnn_lstm_attn", rows, yp.reshape(-1), metrics)
        pd.DataFrame(hist.history).to_csv(C.REPORTS_DIR / "cnn_lstm_attn_history.csv", index=False)

    mpath.write_text(json.dumps(metrics, indent=2))
    print("Saved", mpath)


if __name__ == "__main__":
    main()
