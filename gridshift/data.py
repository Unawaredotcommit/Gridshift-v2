"""Data loading, cleaning, (optional) raw rebuild, and chronological splits."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------------------------
def _parse_ts(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, format="%m/%d/%Y %H:%M", errors="coerce")
    if t.isna().mean() > 0.5:  # different format (e.g. ISO)
        t = pd.to_datetime(s, errors="coerce")
    return t


def _interp_short_gaps(x: pd.Series, max_gap: int) -> pd.Series:
    """Time-interpolate only INTERNAL gaps whose whole length is <= max_gap. Longer gaps stay entirely NaN
    (pandas `limit=` would partially fill them); leading/trailing NaNs are never filled (no extrapolation)."""
    isna = x.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")           # length of the NaN run each NaN belongs to
    filled = x.interpolate(method="time", limit_area="inside")
    return x.where(~(isna & (run_len <= max_gap)), filled)


# Plausible physical ranges in the units declared in config.UNITS (guards against a silently changed dataset)
_UNIT_RANGES = {"tmpc": (-50, 60), "relh": (0, 100.5), "sped": (0, 150), "p01m": (0, 250), "ERCOT": (5_000, 200_000)}


def validate_units(df: pd.DataFrame) -> None:
    """Raise if any column falls outside the physically plausible range for its declared unit
    (e.g. tmpc in degF, relh as a 0-1 fraction, wind in km/h would trip these checks)."""
    bad = []
    for c, (lo, hi) in _UNIT_RANGES.items():
        if c in df.columns:
            x = df[c].dropna()
            if len(x) and (x.min() < lo or x.max() > hi):
                bad.append(f"{c}: observed [{x.min():.2f}, {x.max():.2f}] outside plausible [{lo}, {hi}] "
                           f"for declared unit '{C.UNITS[c]}'")
    if "relh" in df.columns and df["relh"].dropna().max() <= 1.0:
        bad.append("relh looks like a 0-1 fraction, expected percent")
    if bad:
        raise ValueError("Unit check failed -- update gridshift/config.py UNITS/WIND_UNIT or convert the data:\n  "
                         + "\n  ".join(bad))


def clean_with_report(df: pd.DataFrame, max_gap: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Clean to a contiguous hourly frame [ERCOT, tmpc, relh, sped, p01m] and return (frame, report).

    Missing-data policy (config.py):
    * bad timestamps dropped; duplicate timestamps averaged; sorted; reindexed to a full hourly grid
      (hours absent from the file become NaN rows)
    * ERCOT load: NEVER filled/interpolated (would use future load and invent target data). Non-positive load
      is treated as missing.
    * p01m: NEVER zero-filled or interpolated. Missing stays NaN; measured 0 stays 0.
    * tmpc/relh/sped: interpolated only for internal gaps of <= max_gap hours (default config.WEATHER_MAX_GAP).
    * physical clipping (NaN preserved): relh 0-100, sped >= 0, p01m >= 0
    Windows that need any remaining NaN are excluded by `valid_origin_positions`.
    """
    max_gap = C.WEATHER_MAX_GAP if max_gap is None else max_gap
    cols = [C.LOAD_COL] + C.WEATHER_COLS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    d = df[["timestamp"] + cols].copy() if "timestamp" in df.columns else df[cols].copy()
    n_in, n_bad_ts = len(d), 0
    if "timestamp" in d.columns:
        d["timestamp"] = _parse_ts(d["timestamp"])
        n_bad_ts = int(d["timestamp"].isna().sum())
        d = d.dropna(subset=["timestamp"]).set_index("timestamp")
    d = d.apply(pd.to_numeric, errors="coerce")
    n_dup = int(d.index.duplicated().sum())
    d = d.groupby(level=0).mean().sort_index()
    n_file_hours = len(d)
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="h"))
    d.index.name = "timestamp"
    validate_units(d)
    before = d.isna().sum()
    interpolated = {}
    for c in ["tmpc", "relh", "sped"]:
        na0 = d[c].isna()
        if max_gap > 0:
            d[c] = _interp_short_gaps(d[c], max_gap)
        interpolated[c] = int((na0 & d[c].notna()).sum())
    d["relh"] = d["relh"].clip(0, 100)
    d["sped"] = d["sped"].clip(lower=0)
    d["p01m"] = d["p01m"].clip(lower=0)
    d[C.LOAD_COL] = d[C.LOAD_COL].where(d[C.LOAD_COL] > 0)
    after = d.isna().sum()
    report = {
        "input_rows": n_in, "unparseable_timestamps_dropped": n_bad_ts, "duplicate_timestamps_averaged": n_dup,
        "hours_absent_from_file_added_as_NaN": int(len(d) - n_file_hours),
        "weather_max_gap_hours": max_gap,
        "missing_before_cleaning": {c: int(before[c]) for c in cols},
        "interpolated_weather_values": interpolated,
        "missing_after_cleaning": {c: int(after[c]) for c in cols},
        "precipitation_missing_retained": int(after["p01m"]),
        "precipitation_zeros_kept": int((d["p01m"] == 0).sum()),
    }
    d.attrs["cleaning_report"] = report
    return d, report


def clean_hourly(df: pd.DataFrame, max_gap: int | None = None) -> pd.DataFrame:
    """Same as `clean_with_report` but returns only the frame (report kept in `.attrs['cleaning_report']`)."""
    return clean_with_report(df, max_gap)[0]


def load_clean(path: Path | None = None) -> pd.DataFrame:
    """Load the cleaned hourly file produced by scripts/01_prepare_data.py (or build from the
    processed CSV on the fly if it does not exist yet)."""
    if C.CLEAN_PARQUET.exists() and path is None:
        d = pd.read_csv(C.CLEAN_PARQUET, index_col=0, parse_dates=True)
        d.index.name = "timestamp"
        return d.asfreq("h")
    return clean_hourly(pd.read_csv(path or C.PROCESSED_CSV))


# ---------------------------------------------------------------------------------------------
# Optional: rebuild the combined dataset from the raw files
# ---------------------------------------------------------------------------------------------
def read_raw_load(load_dir: Path = C.RAW_LOAD_DIR) -> pd.Series:
    """Concatenate Native_Load_YYYY.xlsx (columns 'Hour Ending', ..., 'ERCOT'). 'Hour Ending'
    timestamps are kept as hour-ending (matches the processed CSV). '24:00' is rolled to next day."""
    frames = []
    for f in sorted(Path(load_dir).glob("Native_Load_*.xlsx")):
        x = pd.read_excel(f)  # header is 'Hour_Ending' in 2018-20 and 'Hour Ending' later -> use col 0
        x = x.rename(columns={x.columns[0]: "Hour Ending"})
        s = x["Hour Ending"].astype(str).str.strip()
        is24 = s.str.contains(r"\s24:00")
        s = s.str.replace(r"\s24:00", " 00:00", regex=True)
        t = pd.to_datetime(s, format="%m/%d/%Y %H:%M", errors="coerce")
        t = t + pd.to_timedelta(is24.astype(int), unit="D")
        frames.append(pd.DataFrame({"timestamp": t, C.LOAD_COL: pd.to_numeric(x[C.LOAD_COL], errors="coerce")}))
    d = pd.concat(frames).dropna(subset=["timestamp"])
    return d.groupby("timestamp")[C.LOAD_COL].mean().sort_index()


def read_raw_weather(path: Path = C.RAW_WEATHER_CSV) -> pd.DataFrame:
    """ASOS multi-station obs (suffix _1.._3) -> hourly station-average, hour-ending aligned."""
    w = pd.read_csv(path)
    w["timestamp"] = pd.to_datetime(w["timestamp"], errors="coerce")
    w = w.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    out = {}
    for var in C.WEATHER_COLS:
        cols = [c for c in w.columns if re.fullmatch(rf"{var}_\d+", c)]
        if not cols:
            raise ValueError(f"No station columns found for {var}")
        # tmpc/relh/sped: mean of obs within the hour ending at t; p01m: mean as in processed set
        out[var] = w[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    h = pd.DataFrame(out)
    # hour-ending bin: obs in (t-1h, t] -> label t
    h = h.resample("1h", closed="right", label="right").mean()
    return h


def mask_upstream_imputed_load(df: pd.DataFrame, load_dir: Path = C.RAW_LOAD_DIR) -> tuple[pd.DataFrame, list]:
    """The processed CSV was linearly interpolated upstream (Data Preprocessing notebook), including ERCOT.
    Hours that are absent from the raw Native_Load_*.xlsx files are therefore imputed target values
    (found: 8 hours -- 7 spring-forward DST hours + 2022-12-01 01:00, all exact neighbour midpoints).
    Set them back to NaN so windows touching them are excluded. No-op if raw files are unavailable."""
    if not Path(load_dir).exists() or not list(Path(load_dir).glob("Native_Load_*.xlsx")):
        return df, []
    raw_ok = read_raw_load(load_dir).dropna().index
    bad = df.index[df[C.LOAD_COL].notna() & ~df.index.isin(raw_ok)]
    out = df.copy()
    out.loc[bad, C.LOAD_COL] = np.nan
    return out, list(bad)


def rebuild_from_raw_frame() -> pd.DataFrame:
    """Raw ERCOT load + hourly station-averaged weather, joined on the hour, NOT yet cleaned."""
    d = read_raw_weather().join(read_raw_load(), how="inner")
    return d.reset_index().rename(columns={d.index.name or "index": "timestamp"})


def rebuild_from_raw() -> pd.DataFrame:
    """Combine raw ERCOT load + weather into the hourly modelling frame (missing-data policy applied)."""
    return clean_hourly(rebuild_from_raw_frame())


# ---------------------------------------------------------------------------------------------
# Origins and chronological splits
# ---------------------------------------------------------------------------------------------
def _required_ranges(horizon: int, include_future: bool = True):
    """Offsets (relative to origin position o) of every hour a forecast actually reads.

    load   : [o-23, o]            origin load, lag1/2/23, 24h rolling stats, sequence history
             [o-167, o-144]-ish   same-hour-last-week lags  (t-168 for t = o+1..o+H)
             [o+1, o+H]           training/evaluation target (only if include_future)
    weather: [o-24, o+H]          history for the sequence input + trailing 24h/6h/3h features at the
                                  target hours (future part only if include_future)
    """
    hi = horizon if include_future else 0
    load = [(-23, hi if include_future else 0), (1 - 168, horizon - 168)]
    weather = [(-24, hi)]
    return load, weather


def _bad_cumsum(series: pd.Series) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(series.isna().to_numpy().astype(int))])


def origin_ok_by_variable(df: pd.DataFrame, horizon: int = C.HORIZON, history: int = C.HISTORY_HOURS,
                          include_future: bool = True):
    """Candidate origin positions and, per variable, whether every hour that variable is read at is present."""
    load_r, wx_r = _required_ranges(horizon, include_future)
    pos = np.arange(history, len(df) - (horizon if include_future else 0))
    ok = {}
    for c, ranges in [(C.LOAD_COL, load_r)] + [(w, wx_r) for w in C.WEATHER_COLS]:
        cs = _bad_cumsum(df[c])
        m = np.ones(len(pos), dtype=bool)
        for lo, hi in ranges:
            m &= (cs[pos + hi + 1] - cs[pos + lo]) == 0
        ok[c] = m
    return pos, ok


def valid_origin_positions(df: pd.DataFrame, horizon: int = C.HORIZON,
                           history: int = C.HISTORY_HOURS, include_future: bool = True) -> np.ndarray:
    """Positions i whose forecast window needs no missing value (complete-case on REQUIRED hours only).

    Only hours that features/targets actually read are checked (`_required_ranges`), so an isolated missing
    hour removes just the origins that depend on it; missing load/precip are never filled.
    """
    pos, ok = origin_ok_by_variable(df, horizon, history, include_future)
    allok = np.all(np.stack(list(ok.values())), axis=0)
    return pos[allok]


def origin_exclusion_report(df: pd.DataFrame, horizon: int = C.HORIZON) -> dict:
    """How many candidate origins are discarded and why (a discarded origin can fail for several variables)."""
    pos, ok = origin_ok_by_variable(df, horizon)
    allok = np.all(np.stack(list(ok.values())), axis=0)
    fails = {c: int((~m).sum()) for c, m in ok.items()}
    only = {c: int((~m & np.all(np.stack([v for k, v in ok.items() if k != c]), axis=0)).sum()) for c, m in ok.items()}
    return {"candidate_origins": int(len(pos)), "valid_origins": int(allok.sum()),
            "discarded_origins": int((~allok).sum()),
            "discarded_where_variable_missing": fails,     # non-exclusive
            "discarded_only_because_of": only}             # exclusive


def origin_is_valid(df: pd.DataFrame, pos: int, horizon: int = C.HORIZON, include_future: bool = True) -> bool:
    """Same rule for a single origin position (used by the simulator)."""
    lo = max(pos - C.HISTORY_HOURS, 0)
    sub = df.iloc[lo: min(pos + horizon + 1, len(df))]
    v = valid_origin_positions(sub, horizon, history=pos - lo, include_future=include_future)
    return bool(len(v) and v[0] == pos - lo)


def split_origins(df: pd.DataFrame, positions: np.ndarray, horizon: int = C.HORIZON,
                  stride: int = 1) -> dict[str, np.ndarray]:
    """Chronological train/val/test split on forecast-origin time.

    train : origin + horizon  <  TRAIN_END   (target window fully inside train period)
    val   : origin >= TRAIN_END and origin + horizon < VAL_END
    test  : origin >= VAL_END
    `stride` keeps only origins whose hour-of-day is a multiple of `stride`.
    """
    t = df.index[positions]
    keep = (t.hour % stride == 0)
    positions, t = positions[keep], t[keep]
    end = t + pd.Timedelta(hours=horizon)
    tr_end, va_end = pd.Timestamp(C.TRAIN_END), pd.Timestamp(C.VAL_END)
    return {
        "train": positions[end < tr_end],
        "val": positions[(t >= tr_end) & (end < va_end)],
        "test": positions[t >= va_end],
    }
