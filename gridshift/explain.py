"""Explainability + weather sensitivity analysis for the tabular models (LightGBM primary).

IMPORTANT INTERPRETATION NOTES
* SHAP values are MODEL-ATTRIBUTED contributions to the model's prediction, not causal effects of weather on
  demand. Permutation importance, PDP, ALE and the scenario sensitivity/interaction grids are likewise
  properties of the fitted model (model-based what-if estimates).
* Temperature, feels-like, CDD/HDD and the trailing temperature extremes are strongly correlated, so credit is
  split arbitrarily between them in single-feature SHAP/permutation scores. Use the GROUP results (which treat
  correlated features jointly) to judge "how much does temperature matter"; see weather_feature_correlations.csv.
  `feel` is also a function of humidity and wind, so the temperature group is not a pure-temperature effect.

Methods
* SHAP        : LightGBM native TreeSHAP (`pred_contrib=True`), plus optional shap-package
                interaction values on a small sample. Group SHAP = mean over rows of |sum of member SHAP|
                (joint attribution; sign cancellation inside the group is respected).
* Permutation : out-of-sample (test) RMSE increase when a feature / feature-group is shuffled jointly.
                Rows are shuffled across seasons/hours, so unrealistic combinations are evaluated (a known
                limitation, worst for correlated features).
* PDP         : consistent partial dependence -- the weather variable is set in the 24h window and ALL
                derived weather features are recomputed by the normal feature pipeline.
* ALE         : 1-D accumulated local effects of ONE raw column on the feature matrix. DIRECT effect only:
                derived features (feel, CDD/HDD, ...) are held at their observed values, so it does not
                include the effect that flows through them.
* Sensitivity : scenario-engine response of peak/mean demand to +-weather shifts, by season.
* Interaction : 2-D grid (temperature shift x humidity shift) of peak response.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import simulator as S
from .metrics import regression_metrics

TEMP_GROUP = ["tmpc", "feel", "cdd", "hdd", "tmpc_max24", "tmpc_min24", "tmpc_chg3"]
PERM_GROUPS = {
    "Temperature-derived (incl. feels-like)": TEMP_GROUP,
    "Humidity": ["relh"],
    "Wind": ["sped"],
    "Precipitation": ["p01m", "precip_6h"],
    "All weather": F.WEATHER_FEATURES,
    "Calendar": F.CALENDAR_FEATURES,
    "Demand history": F.DEMAND_FEATURES,
}


# SHAP groups mirror the permutation groups (+ horizon). Groups overlap on purpose ("All weather" contains
# the temperature/humidity/wind/precipitation groups), so do not add them up.
SHAP_GROUPS = {**PERM_GROUPS, "Horizon": ["horizon"]}


# ------------------------------------------------------------------------------------- SHAP
def shap_values(lgbm, X: pd.DataFrame) -> pd.DataFrame:
    """TreeSHAP contributions (MW) per feature; column 'expected_value' is the model base value."""
    contrib = lgbm.predict(X[F.TABULAR_FEATURES], pred_contrib=True)
    return pd.DataFrame(contrib, columns=F.TABULAR_FEATURES + ["expected_value"], index=X.index)


def shap_summary(sv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-feature mean|SHAP|, per-group joint mean|SHAP|). Model-attributed, not causal."""
    g = sv[F.TABULAR_FEATURES].abs().mean().sort_values(ascending=False)
    glob = g.rename("mean_abs_shap_mw").reset_index().rename(columns={"index": "feature"})
    grp = pd.DataFrame([{"group": k, "mean_abs_shap_mw": float(sv[v].sum(axis=1).abs().mean()),
                         "n_features": len(v)} for k, v in SHAP_GROUPS.items()]
                       ).sort_values("mean_abs_shap_mw", ascending=False)
    return glob, grp


def shap_interactions_weather(lgbm, X: pd.DataFrame, n: int = 400) -> pd.DataFrame | None:
    """Mean |SHAP interaction| matrix over core weather+hour features (needs `shap`)."""
    try:
        import shap
    except Exception:
        return None
    cols = ["tmpc", "relh", "sped", "p01m", "feel", "hour", "is_weekend", "load_lag24"]
    Xs = X[F.TABULAR_FEATURES].sample(min(n, len(X)), random_state=C.SEED)
    iv = shap.TreeExplainer(lgbm).shap_interaction_values(Xs)
    idx = [F.TABULAR_FEATURES.index(c) for c in cols]
    m = np.abs(iv).mean(axis=0)[np.ix_(idx, idx)]
    return pd.DataFrame(m, index=cols, columns=cols)


def save_shap_plots(lgbm, X: pd.DataFrame, out_dir, n: int = 2000) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
    except Exception:
        return
    Xs = X[F.TABULAR_FEATURES].sample(min(n, len(X)), random_state=C.SEED)
    sv = shap.TreeExplainer(lgbm).shap_values(Xs)
    plt.figure(); shap.summary_plot(sv, Xs, show=False, max_display=20); plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm.png", dpi=150); plt.close()
    plt.figure(); shap.summary_plot(sv, Xs, plot_type="bar", show=False, max_display=20); plt.tight_layout()
    plt.savefig(out_dir / "shap_bar.png", dpi=150); plt.close()


# ------------------------------------------------------------------------ permutation importance
def permutation_importance(model, X: pd.DataFrame, y: np.ndarray, groups: dict[str, list[str]],
                           n_repeats: int = 5, seed: int = C.SEED) -> pd.DataFrame:
    """RMSE increase (MW) after shuffling each column set jointly (rows permuted together)."""
    rng = np.random.default_rng(seed)
    X = X[F.TABULAR_FEATURES].reset_index(drop=True)
    base = regression_metrics(y, model.predict(X))["RMSE"]
    rows = []
    for name, cols in groups.items():
        inc = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            inc.append(regression_metrics(y, model.predict(Xp))["RMSE"] - base)
        rows.append({"name": name, "rmse_increase_mw": float(np.mean(inc)), "std": float(np.std(inc)),
                     "baseline_rmse": base, "n_cols": len(cols)})
    return pd.DataFrame(rows).sort_values("rmse_increase_mw", ascending=False)


# ------------------------------------------------------------------------------------------ ALE
def ale_1d(model, X: pd.DataFrame, feature: str, n_bins: int = 20) -> pd.DataFrame:
    X = X[F.TABULAR_FEATURES].reset_index(drop=True)
    v = X[feature].to_numpy()
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return pd.DataFrame(columns=["feature", "x", "ale_mw"])
    b = np.clip(np.digitize(v, edges[1:-1], right=True), 0, len(edges) - 2)
    effects, counts = [], []
    for k in range(len(edges) - 1):
        m = b == k
        if not m.any():
            effects.append(0.0); counts.append(0); continue
        lo, hi = X[m].copy(), X[m].copy()
        lo[feature], hi[feature] = edges[k], edges[k + 1]
        effects.append(float(np.mean(model.predict(hi) - model.predict(lo)))); counts.append(int(m.sum()))
    ale = np.concatenate([[0.0], np.cumsum(effects)])
    w = np.array(counts); mid = (ale[:-1] + ale[1:]) / 2
    ale = ale - (mid * w).sum() / max(w.sum(), 1)  # center to mean 0
    return pd.DataFrame({"feature": feature, "x": edges, "ale_mw": ale})


# ------------------------------------------------------------ scenario-engine based analyses
def _stats(fc: S.Forecaster, origin, weather: pd.DataFrame) -> tuple[float, float]:
    y = fc.forecast(origin, weather)["forecast_mw"].to_numpy()
    return float(y.max()), float(y.mean())


def pdp_curves(fc: S.Forecaster, origins, grids: dict[str, np.ndarray]) -> pd.DataFrame:
    """Classic PDP: variable set to a constant for the whole 24h window (derived features recomputed);
    averaged over origins. Reports mean-hourly and peak demand."""
    rows = []
    for var, grid in grids.items():
        for val in grid:
            peaks, means = [], []
            for o in origins:
                w = fc.baseline_weather(o).copy()
                w[var] = val
                p, m = _stats(fc, o, w)
                peaks.append(p); means.append(m)
            rows.append({"variable": var, "value": float(val), "mean_demand_mw": np.mean(means),
                         "peak_demand_mw": np.mean(peaks)})
    return pd.DataFrame(rows)


SENS_GRID = {
    "temp_delta_c": [-10, -5, -2, 2, 5, 10],
    "humidity_delta_pct": [-30, -15, 15, 30],
    "wind_scale": [0.0, 0.5, 1.5, 2.0],
    "precip_add_mm": [0.5, 2.0],
}


def _season(ts) -> str:
    return {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer"}.get(ts.month, "Fall")


def sensitivity_table(fc: S.Forecaster, origins, grid: dict[str, list] = SENS_GRID) -> pd.DataFrame:
    """Mean change in peak / mean demand vs baseline for single-variable perturbations, by season."""
    rows = []
    for var, vals in grid.items():
        for v in vals:
            sc = S.Scenario(**{var: v})
            per = []
            for o in origins:
                s = fc.compare(o, sc)["summary"]
                per.append({"season": _season(o), "peak": s["peak_delta_mw"], "mean": s["mean_delta_mw"]})
            d = pd.DataFrame(per)
            for season, g in list(d.groupby("season")) + [("All", d)]:
                rows.append({"variable": var, "perturbation": v, "season": season, "n_origins": len(g),
                             "peak_delta_mw_mean": g["peak"].mean(), "peak_delta_mw_std": g["peak"].std(),
                             "mean_delta_mw_mean": g["mean"].mean()})
    return pd.DataFrame(rows)


def interaction_grid(fc: S.Forecaster, origins, temp_deltas=(-8, -4, 0, 4, 8),
                     hum_deltas=(-30, -15, 0, 15, 30)) -> pd.DataFrame:
    """2-D response of mean peak change to joint temperature x humidity shifts (summer origins are the
    informative ones; all given origins are averaged -- pass a season-filtered list to specialise)."""
    rows = []
    for dt in temp_deltas:
        for dh in hum_deltas:
            sc = S.Scenario(temp_delta_c=dt, humidity_delta_pct=dh)
            vals = [fc.compare(o, sc)["summary"]["peak_delta_mw"] for o in origins]
            rows.append({"temp_delta_c": dt, "humidity_delta_pct": dh, "peak_delta_mw": float(np.mean(vals))})
    return pd.DataFrame(rows)
