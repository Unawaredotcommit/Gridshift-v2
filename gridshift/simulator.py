"""Weather scenario simulator.

Given a forecast origin, the baseline future weather is the observed weather of the next 24h
(historical replay; or pass `future_weather` explicitly).  A `Scenario` edits the RAW weather
variables (temperature, humidity, wind, precipitation); all derived features (feels-like, CDD/HDD,
trailing 24h temperature extremes, 3h temperature change, 6h precipitation) are then recomputed
by the same code used in training, so they stay consistent with the edited raw values.
Demand-history features are never changed by a scenario.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import features as F

TEMP_RANGE = (-30.0, 55.0)  # degC safety clip


@dataclass
class Scenario:
    temp_delta_c: float = 0.0        # additive shift, degC
    humidity_delta_pct: float = 0.0  # additive shift, percentage points (result clipped to 0-100)
    wind_scale: float = 1.0          # multiplier on wind speed (>=0)
    precip_scale: float = 1.0        # multiplier on hourly precipitation
    precip_add_mm: float = 0.0       # extra mm/h added in the window
    hour_start: int = 1              # first forecast hour (1..24) the change applies to
    hour_end: int = 24               # last forecast hour (1..24) the change applies to

    def is_identity(self) -> bool:
        return (self.temp_delta_c == 0 and self.humidity_delta_pct == 0 and self.wind_scale == 1
                and self.precip_scale == 1 and self.precip_add_mm == 0)

    def to_dict(self) -> dict:
        return asdict(self)


def apply_scenario(future_weather: pd.DataFrame, sc: Scenario) -> pd.DataFrame:
    """future_weather: 24 rows with raw columns tmpc/relh/sped/p01m. Returns an edited copy."""
    w = future_weather[C.WEATHER_COLS].astype(float).copy()
    n = len(w)
    h = np.arange(1, n + 1)
    m = (h >= sc.hour_start) & (h <= sc.hour_end)
    w.loc[m, "tmpc"] = (w.loc[m, "tmpc"] + sc.temp_delta_c).clip(*TEMP_RANGE)
    w.loc[m, "relh"] = (w.loc[m, "relh"] + sc.humidity_delta_pct).clip(0, 100)
    w.loc[m, "sped"] = (w.loc[m, "sped"] * max(sc.wind_scale, 0.0)).clip(lower=0)
    w.loc[m, "p01m"] = (w.loc[m, "p01m"] * max(sc.precip_scale, 0.0) + sc.precip_add_mm).clip(lower=0)
    return w


# ---------------------------------------------------------------------------------------------
# Predictors: (feature_frame, origin_position) -> np.ndarray of 24 MW values
# ---------------------------------------------------------------------------------------------
Predictor = Callable[[pd.DataFrame, int], np.ndarray]


def tabular_predictor(model) -> Predictor:
    def _p(feat: pd.DataFrame, opos: int) -> np.ndarray:
        rows = F.build_tabular(feat, [opos], with_target=False)
        return np.asarray(model.predict(rows[F.TABULAR_FEATURES]), dtype=float)
    return _p


def deep_predictor(model, scaler) -> Predictor:
    from . import deep
    def _p(feat: pd.DataFrame, opos: int) -> np.ndarray:
        X, _ = F.build_sequences(feat, [opos], with_target=False)
        return deep.predict(model, scaler, X)[0]
    return _p


class Forecaster:
    def __init__(self, df_clean: pd.DataFrame, predictor: Predictor, horizon: int = C.HORIZON):
        self.df = df_clean
        self.predict_window = predictor
        self.horizon = horizon

    # -- helpers -------------------------------------------------------------------------------
    def origin_choices(self) -> pd.DatetimeIndex:
        """Origins with full history and 24h of observed future (usable for historical replay)."""
        return self.df.index[D.valid_origin_positions(self.df, self.horizon)]

    def baseline_weather(self, origin_time) -> pd.DataFrame:
        p = self.df.index.get_loc(pd.Timestamp(origin_time))
        fw = self.df.iloc[p + 1: p + 1 + self.horizon][C.WEATHER_COLS]
        if len(fw) < self.horizon or fw.isna().any().any():
            raise ValueError("Observed future weather is missing/incomplete after this origin; "
                             "choose another origin or pass future_weather.")
        return fw

    def _frame(self, origin_time, future_weather: pd.DataFrame):
        p = self.df.index.get_loc(pd.Timestamp(origin_time))
        if p < C.HISTORY_HOURS:
            raise ValueError("Not enough history before this origin.")
        if not D.origin_is_valid(self.df, p, self.horizon, include_future=False):
            raise ValueError("Load/weather needed by this origin's features contains missing values "
                             "(missing data is never filled); choose another origin.")
        if future_weather[C.WEATHER_COLS].isna().any().any():
            raise ValueError("future_weather contains missing values.")
        hist = self.df.iloc[p - C.HISTORY_HOURS: p + 1][[C.LOAD_COL] + C.WEATHER_COLS]
        o = hist.index[-1]
        fidx = pd.date_range(o + pd.Timedelta(hours=1), periods=self.horizon, freq="h")
        fut = pd.DataFrame(index=fidx, columns=[C.LOAD_COL] + C.WEATHER_COLS, dtype=float)
        fut[C.WEATHER_COLS] = future_weather[C.WEATHER_COLS].to_numpy(float)
        full = pd.concat([hist, fut])
        feat = F.make_feature_frame(full)
        return feat, len(hist) - 1

    # -- public API ----------------------------------------------------------------------------
    def forecast(self, origin_time, future_weather: pd.DataFrame | None = None) -> pd.DataFrame:
        fw = self.baseline_weather(origin_time) if future_weather is None else future_weather
        feat, opos = self._frame(origin_time, fw)
        yhat = self.predict_window(feat, opos)
        out = feat.iloc[opos + 1: opos + 1 + self.horizon][C.WEATHER_COLS + ["feel"]].copy()
        out.insert(0, "forecast_mw", yhat)
        out.insert(0, "horizon", np.arange(1, self.horizon + 1))
        return out

    def compare(self, origin_time, scenario: Scenario, base_weather: pd.DataFrame | None = None) -> dict:
        """Baseline vs scenario 24h forecast + summary stats."""
        base_w = self.baseline_weather(origin_time) if base_weather is None else base_weather[C.WEATHER_COLS]
        scen_w = apply_scenario(base_w.reset_index(drop=True), scenario)
        scen_w.index = base_w.index
        base = self.forecast(origin_time, base_w)
        scen = self.forecast(origin_time, scen_w)
        hourly = pd.DataFrame({
            "horizon": base["horizon"],
            "baseline_mw": base["forecast_mw"], "scenario_mw": scen["forecast_mw"]})
        hourly["delta_mw"] = hourly["scenario_mw"] - hourly["baseline_mw"]
        hourly["delta_pct"] = hourly["delta_mw"] / hourly["baseline_mw"] * 100
        actual = self.df[C.LOAD_COL].reindex(hourly.index)
        if actual.notna().all():
            hourly["actual_mw"] = actual  # present in historical replay only
        bi, si = hourly["baseline_mw"].idxmax(), hourly["scenario_mw"].idxmax()
        summary = {
            "baseline_peak_mw": float(hourly.loc[bi, "baseline_mw"]), "baseline_peak_time": bi,
            "scenario_peak_mw": float(hourly.loc[si, "scenario_mw"]), "scenario_peak_time": si,
            "peak_delta_mw": float(hourly["scenario_mw"].max() - hourly["baseline_mw"].max()),
            "peak_delta_pct": float((hourly["scenario_mw"].max() / hourly["baseline_mw"].max() - 1) * 100),
            "energy_delta_mwh": float(hourly["delta_mw"].sum()),  # hourly MW * 1h
            "mean_delta_mw": float(hourly["delta_mw"].mean()),
            "max_hourly_increase_mw": float(hourly["delta_mw"].max()),
            "max_hourly_decrease_mw": float(hourly["delta_mw"].min()),
        }
        return {"hourly": hourly, "summary": summary,
                "weather_baseline": base[C.WEATHER_COLS + ["feel"]],
                "weather_scenario": scen[C.WEATHER_COLS + ["feel"]],
                "scenario": scenario.to_dict()}
