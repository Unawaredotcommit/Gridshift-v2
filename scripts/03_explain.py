"""Step 3 - explainability + sensitivity analysis for the trained LightGBM model (test period).

Usage:  python scripts/03_explain.py [--quick]
  --quick : fewer origins/samples (fast smoke run)
All outputs are MODEL-ATTRIBUTED / model-based what-if estimates of the trained LightGBM, not causal effects.
The 2024 test split is used here only to explain the already-final model (no model selection happens in this step).
Needs: artifacts/models/lightgbm.joblib (from step 2); optional `shap` for plots/interactions.
Outputs (artifacts/explain/): shap_global.csv, shap_groups.csv, shap_values_sample.csv,
  shap_interactions_weather.csv, shap_beeswarm.png, shap_bar.png, perm_importance.csv,
  perm_importance_groups.csv, weather_feature_correlations.csv, pdp.csv, ale.csv, sensitivity.csv, interaction_grid.csv
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from gridshift import config as C
from gridshift import data as D
from gridshift import explain as E
from gridshift import features as F
from gridshift import models as MD
from gridshift import simulator as S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = C.EXPLAIN_DIR; out.mkdir(parents=True, exist_ok=True)

    df = D.load_clean()
    feat = F.make_feature_frame(df)
    splits, _ = F.get_splits(df, feat)
    test = F.build_tabular(feat, splits["test"])
    lgbm = MD.load_model("lightgbm")

    n_shap, n_perm = (500, 5000) if args.quick else (4000, 20000)
    sample = test.sample(min(n_shap, len(test)), random_state=C.SEED)

    print("SHAP ..."); sv = E.shap_values(lgbm, sample)
    glob, grp = E.shap_summary(sv)
    glob.to_csv(out / "shap_global.csv", index=False); grp.to_csv(out / "shap_groups.csv", index=False)
    pd.concat([sample[F.TABULAR_FEATURES].add_prefix("x_"), sv], axis=1) \
        .to_csv(out / "shap_values_sample.csv", index=False)
    inter = E.shap_interactions_weather(lgbm, sample, n=100 if args.quick else 400)
    if inter is not None:
        inter.to_csv(out / "shap_interactions_weather.csv")
    else:
        print("  (shap package not installed: skipping interaction values/plots)")
    E.save_shap_plots(lgbm, sample, out, n=500 if args.quick else 2000)

    print("Permutation importance ...")
    ps = test.sample(min(n_perm, len(test)), random_state=C.SEED)
    E.permutation_importance(lgbm, ps, ps["y"].to_numpy(), {c: [c] for c in F.TABULAR_FEATURES},
                             n_repeats=2 if args.quick else 5).to_csv(out / "perm_importance.csv", index=False)
    E.permutation_importance(lgbm, ps, ps["y"].to_numpy(), E.PERM_GROUPS,
                             n_repeats=2 if args.quick else 5).to_csv(out / "perm_importance_groups.csv", index=False)

    ps[F.WEATHER_FEATURES].corr().to_csv(out / "weather_feature_correlations.csv")  # for interpreting importances

    print("ALE ...")
    pd.concat([E.ale_1d(lgbm, ps, f) for f in ["tmpc", "relh", "sped", "feel"]]).to_csv(out / "ale.csv", index=False)

    print("Scenario-engine analyses (PDP / sensitivity / interaction) ...")
    fc = S.Forecaster(df, S.tabular_predictor(lgbm))
    t_origins = df.index[splits["test"]]
    t_origins = t_origins[t_origins.hour == 0]
    origins = list(t_origins[:: 12 if args.quick else 4])
    tr = df.loc[:C.TRAIN_END]
    grids = {"tmpc": np.round(np.linspace(tr.tmpc.quantile(.02), tr.tmpc.quantile(.98), 12), 1),
             "relh": np.linspace(20, 100, 9), "sped": np.linspace(0, 20, 9), "p01m": np.array([0, .25, .5, 1, 2, 4])}
    E.pdp_curves(fc, origins[:: (1 if args.quick else 2)], grids).to_csv(out / "pdp.csv", index=False)
    E.sensitivity_table(fc, origins).to_csv(out / "sensitivity.csv", index=False)
    summer = [o for o in origins if o.month in (6, 7, 8, 9)] or origins
    E.interaction_grid(fc, summer).to_csv(out / "interaction_grid.csv", index=False)
    print("Done ->", out)


if __name__ == "__main__":
    main()
