"""Feature engineering shared by training, explainability and the scenario simulator.

Forecast framing
----------------
A forecast is issued at an *origin* hour T (load up to and including T is known) for
target hours t = T+1 .. T+24.  Features for a (origin, horizon h) row:

* Demand features -- strictly from hours <= T  (origin load, 24h stats at T) or lags that are
  always <= T for h <= 24 (same hour yesterday t-24, same hour last week t-168).
* Weather features at the target hour t, plus trailing 24h temperature extremes ending at t.
  These are treated as *known exogenous inputs* (a weather forecast).  Historical replays use
  observed weather as a proxy for the forecast, so reported accuracy is optimistic relative to
  operating with real, imperfect weather forecasts.
* Calendar features of the target hour t.

Units: tmpc degC, relh %, sped per `config.WIND_UNIT` (mph), p01m mm/h, feel degF.

Every derived weather feature (feel, CDD/HDD, trailing extremes, 3h change) is computed in ONE
place (`add_weather_derived`) so edited scenario weather stays internally consistent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from . import config as C
from . import data as D

BASE_TEMP_C = C.BASE_TEMP_C  # CDD/HDD balance point (config; 65F)

# ---------------------------------------------------------------------------------------------
# Weather-derived features
# ---------------------------------------------------------------------------------------------

def feels_like_f(tmpc: pd.Series, relh: pd.Series, sped: pd.Series) -> pd.Series:
    """NWS apparent temperature in degF.

    Inputs (dataset units): tmpc in degC, relh in %, sped in `config.WIND_UNIT` (mph for this dataset).
    Internally: temperature -> degF, wind -> mph (NWS wind chill needs mph; heat index needs degF and %RH).
      * heat index (NWS Rothfusz + low/high-humidity adjustments; simple formula below 80F)
        used when T >= 80F
      * wind chill when T <= 50F and wind > 3 mph
      * otherwise the air temperature
    Checked against the raw per-station `feel` column: MAE ~0.04F where T>=80F, ~0.2F elsewhere.
    """
    t = tmpc.astype(float) * 9.0 / 5.0 + 32.0
    r = relh.astype(float).clip(0, 100)
    w = sped.astype(float).clip(lower=0) * C.WIND_TO_MPH[C.WIND_UNIT]
    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + r * 0.094)
    roth = (-42.379 + 2.04901523 * t + 10.14333127 * r - 0.22475541 * t * r - 6.83783e-3 * t * t
            - 5.481717e-2 * r * r + 1.22874e-3 * t * t * r + 8.5282e-4 * t * r * r - 1.99e-6 * t * t * r * r)
    adj_low = np.where((r < 13) & (t >= 80) & (t <= 112),
                       -((13 - r) / 4) * np.sqrt(np.clip((17 - (t - 95).abs()) / 17, 0, None)), 0.0)
    adj_high = np.where((r > 85) & (t >= 80) & (t <= 87), ((r - 85) / 10) * ((87 - t) / 5), 0.0)
    hi = np.where((simple + t) / 2 >= 80, roth + adj_low + adj_high, simple)
    w16 = w.pow(0.16)
    wc = 35.74 + 0.6215 * t - 35.75 * w16 + 0.4275 * t * w16
    out = np.where(t >= 80, hi, np.where((t <= 50) & (w > 3), wc, t))
    out = np.where(t.isna() | r.isna() | w.isna(), np.nan, out)   # never invent values for missing inputs
    return pd.Series(out, index=tmpc.index, name="feel")


def add_weather_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived weather columns from raw tmpc/relh/sped/p01m. Needs a contiguous hourly index."""
    out = df.copy()
    out["feel"] = feels_like_f(out["tmpc"], out["relh"], out["sped"])
    out["cdd"] = (out["tmpc"] - BASE_TEMP_C).clip(lower=0)
    out["hdd"] = (BASE_TEMP_C - out["tmpc"]).clip(lower=0)
    # Trailing windows (no look-ahead beyond t). Windows need ALL their hours: a missing temperature or a
    # series start gives NaN, never a value computed from a partial window or an artificial 0 change.
    out["tmpc_max24"] = out["tmpc"].rolling(24, min_periods=24).max()
    out["tmpc_min24"] = out["tmpc"].rolling(24, min_periods=24).min()
    out["tmpc_chg3"] = out["tmpc"].diff(3)
    out["precip_6h"] = out["p01m"].rolling(6, min_periods=6).sum()   # NaN if any hour missing (no silent zeros)
    return out


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    hol = USFederalHolidayCalendar().holidays(idx.min().normalize() - pd.Timedelta(days=1),
                                              idx.max().normalize() + pd.Timedelta(days=1))
    out["hour"] = idx.hour
    out["dow"] = idx.dayofweek
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_holiday"] = idx.normalize().isin(hol).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 366)
    out["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 366)
    return out


def add_demand_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing demand statistics (value at position i uses hours <= i only)."""
    out = df.copy()
    load = out[C.LOAD_COL]
    out["load_roll24_mean"] = load.rolling(24, min_periods=24).mean()
    out["load_roll24_max"] = load.rolling(24, min_periods=24).max()
    out["load_roll24_min"] = load.rolling(24, min_periods=24).min()
    out["load_lag1"] = load.shift(1)
    out["load_lag2"] = load.shift(2)
    out["load_lag23"] = load.shift(23)
    return out


def make_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Per-hour feature frame (weather derived + calendar + demand stats). Index must be hourly."""
    return add_demand_stats(add_calendar(add_weather_derived(df)))


# ---------------------------------------------------------------------------------------------
# Tabular (horizon-stacked) dataset
# ---------------------------------------------------------------------------------------------
WEATHER_FEATURES = ["tmpc", "relh", "sped", "p01m", "feel", "cdd", "hdd",
                    "tmpc_max24", "tmpc_min24", "tmpc_chg3", "precip_6h"]
CALENDAR_FEATURES = ["hour", "dow", "month", "is_weekend", "is_holiday",
                     "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
DEMAND_FEATURES = ["load_origin", "load_origin_lag1", "load_origin_lag2", "load_origin_lag23",
                   "load_roll24_mean", "load_roll24_max", "load_roll24_min",
                   "load_lag24", "load_lag168"]
TABULAR_FEATURES = DEMAND_FEATURES + WEATHER_FEATURES + CALENDAR_FEATURES + ["horizon"]

FEATURE_GROUPS = {
    "Demand history": DEMAND_FEATURES,
    "Weather": WEATHER_FEATURES,
    "Calendar": CALENDAR_FEATURES,
    "Horizon": ["horizon"],
}


def build_tabular(feat: pd.DataFrame, origin_pos: np.ndarray, horizon: int = C.HORIZON,
                  with_target: bool = True) -> pd.DataFrame:
    """Stack (origin, h) rows. `origin_pos` are integer positions into `feat` (hourly, contiguous).

    Row columns: TABULAR_FEATURES + origin_time, target_time, y (if with_target).
    Leakage guard: only load at positions <= origin is read (origin stats; lag24/lag168 of target).
    """
    origin_pos = np.asarray(origin_pos, dtype=int)
    h = np.arange(1, horizon + 1)
    opos = np.repeat(origin_pos, horizon)
    hh = np.tile(h, len(origin_pos))
    tpos = opos + hh
    assert (tpos - 24 <= opos).all() and (tpos - 168 <= opos).all()  # lags never look past origin

    load = feat[C.LOAD_COL].to_numpy()
    rows = {
        "load_origin": load[opos],
        "load_origin_lag1": feat["load_lag1"].to_numpy()[opos],
        "load_origin_lag2": feat["load_lag2"].to_numpy()[opos],
        "load_origin_lag23": feat["load_lag23"].to_numpy()[opos],
        "load_roll24_mean": feat["load_roll24_mean"].to_numpy()[opos],
        "load_roll24_max": feat["load_roll24_max"].to_numpy()[opos],
        "load_roll24_min": feat["load_roll24_min"].to_numpy()[opos],
        "load_lag24": load[tpos - 24],
        "load_lag168": load[tpos - 168],
        "horizon": hh,
    }
    for c in WEATHER_FEATURES + CALENDAR_FEATURES:
        rows[c] = feat[c].to_numpy()[tpos]
    out = pd.DataFrame(rows)
    out["origin_time"] = feat.index.to_numpy()[opos]
    out["target_time"] = feat.index.to_numpy()[tpos]
    if with_target:
        out["y"] = load[tpos]
    return out


def complete_origin_mask(feat: pd.DataFrame, origin_pos: np.ndarray, horizon: int = C.HORIZON) -> np.ndarray:
    """True for origins whose tabular rows (all horizons, incl. target) AND sequence inputs are fully
    observed. Used once in the scripts so Ridge/LightGBM and CNN-LSTM-Attention are trained/evaluated on
    exactly the same forecast origins and target hours."""
    origin_pos = np.asarray(origin_pos, dtype=int)
    if len(origin_pos) == 0:
        return np.zeros(0, dtype=bool)
    tab = build_tabular(feat, origin_pos, horizon)
    bad_tab = tab[TABULAR_FEATURES + ["y"]].isna().any(axis=1).to_numpy().reshape(len(origin_pos), horizon).any(axis=1)
    X, y = build_sequences(feat, origin_pos, horizon)
    bad_seq = np.isnan(X).reshape(len(origin_pos), -1).any(axis=1) | np.isnan(y).any(axis=1)
    return ~(bad_tab | bad_seq)


def get_splits(df: pd.DataFrame, feat: pd.DataFrame) -> tuple[dict, dict]:
    """Chronological train/val/test origin positions used by EVERY model (tabular and CNN-LSTM-Attention):
    valid origins (no missing required input) -> chronological split -> drop any origin whose rows/sequence
    still contain a NaN. Returns (splits, n_dropped_by_completeness_filter)."""
    raw = D.split_origins(df, D.valid_origin_positions(df), stride=C.ORIGIN_STRIDE)
    out, dropped = {}, {}
    for k, v in raw.items():
        keep = complete_origin_mask(feat, v)
        out[k], dropped[k] = v[keep], int((~keep).sum())
    return out, dropped


# ---------------------------------------------------------------------------------------------
# Sequence dataset for the CNN-LSTM-Attention model
# ---------------------------------------------------------------------------------------------
# 48-step window: 24 history steps (ERCOT known) + 24 future steps (ERCOT masked to 0, mask=1).
SEQ_WEATHER = ["tmpc", "relh", "sped", "p01m", "feel"]
SEQ_CAL = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "is_weekend", "is_holiday"]
SEQ_FEATURES = ["load", "future_mask"] + SEQ_WEATHER + SEQ_CAL
SEQ_SCALED = ["load"] + SEQ_WEATHER  # columns standardised (fit on train only)


def build_sequences(feat: pd.DataFrame, origin_pos: np.ndarray, horizon: int = C.HORIZON,
                    with_target: bool = True):
    """Returns X (n, 2*horizon, len(SEQ_FEATURES)) UNSCALED, y (n, horizon) UNSCALED or None."""
    origin_pos = np.asarray(origin_pos, dtype=int)
    n = len(origin_pos)
    X = np.zeros((n, 2 * horizon, len(SEQ_FEATURES)), dtype=np.float32)
    load = feat[C.LOAD_COL].to_numpy()
    cols = {c: feat[c].to_numpy() for c in SEQ_WEATHER + SEQ_CAL}
    for k, c in enumerate(SEQ_FEATURES):
        if c in ("load", "future_mask"):
            continue
        for j in range(2 * horizon):
            X[:, j, k] = cols[c][origin_pos - (horizon - 1) + j]
    for j in range(horizon):  # history load: positions origin-23 .. origin
        X[:, j, 0] = load[origin_pos - (horizon - 1) + j]
    X[:, horizon:, 1] = 1.0   # future mask; future load stays 0 (unknown)
    y = None
    if with_target:
        y = np.stack([load[origin_pos + 1 + j] for j in range(horizon)], axis=1).astype(np.float32)
    return X, y
