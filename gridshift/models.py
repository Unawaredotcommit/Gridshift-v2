"""Ridge (baseline) and LightGBM (primary) forecasters on the horizon-stacked tabular dataset."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import TABULAR_FEATURES
from .metrics import regression_metrics


def fit_ridge(train: pd.DataFrame, val: pd.DataFrame):
    """StandardScaler (fit on train only) + Ridge; alpha chosen on validation RMSE."""
    X_tr, y_tr = train[TABULAR_FEATURES], train["y"]
    X_va, y_va = val[TABULAR_FEATURES], val["y"]
    best, best_rmse, log = None, np.inf, []
    for a in C.RIDGE_ALPHAS:
        pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=a))]).fit(X_tr, y_tr)
        rmse = regression_metrics(y_va, pipe.predict(X_va))["RMSE"]
        log.append({"alpha": a, "val_RMSE": rmse})
        if rmse < best_rmse:
            best, best_rmse = pipe, rmse
    return best, pd.DataFrame(log)


def fit_lgbm(train: pd.DataFrame, val: pd.DataFrame):
    """LightGBM with early stopping on the chronological validation set."""
    import lightgbm as lgb
    model = lgb.LGBMRegressor(**C.LGBM_PARAMS)
    model.fit(
        train[TABULAR_FEATURES], train["y"],
        eval_set=[(val[TABULAR_FEATURES], val["y"])], eval_metric="l2",
        callbacks=[lgb.early_stopping(C.LGBM_EARLY_STOPPING, verbose=False), lgb.log_evaluation(200)],
    )
    return model


def predict(model, df: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(df[TABULAR_FEATURES]), dtype=float)


def save_model(model, name: str, models_dir: Path = C.MODELS_DIR) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    p = models_dir / f"{name}.joblib"
    joblib.dump(model, p)
    return p


def load_model(name: str, models_dir: Path = C.MODELS_DIR):
    p = models_dir / f"{name}.joblib"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run scripts/02_train_models.py first.")
    return joblib.load(p)
