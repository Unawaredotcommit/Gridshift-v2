"""Lightweight tests (no model training). Run: python -m pytest tests -q   (or python tests/test_core.py)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from gridshift import config as C
from gridshift import data as D
from gridshift import features as F
from gridshift import simulator as S


def synth(n_days=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01 01:00", periods=24 * n_days, freq="h")
    t = np.arange(len(idx))
    df = pd.DataFrame({
        C.LOAD_COL: 40000 + 5000 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 200, len(idx)),
        "tmpc": 15 + 10 * np.sin(2 * np.pi * (t - 6) / 24) + rng.normal(0, 1, len(idx)),
        "relh": rng.uniform(30, 90, len(idx)), "sped": rng.uniform(0, 12, len(idx)),
        "p01m": np.where(rng.random(len(idx)) < .1, rng.uniform(0, 3, len(idx)), 0.0)}, index=idx)
    df.index.name = "timestamp"
    return df


def test_tabular_no_leakage():
    """Changing load strictly AFTER the origin must not change any feature of that origin."""
    df = synth()
    pos = 24 * 20
    feat_a = F.make_feature_frame(df)
    df2 = df.copy()
    df2.iloc[pos + 1:, df2.columns.get_loc(C.LOAD_COL)] += 99999  # future load corrupted
    feat_b = F.make_feature_frame(df2)
    a = F.build_tabular(feat_a, [pos], with_target=False)[F.TABULAR_FEATURES]
    b = F.build_tabular(feat_b, [pos], with_target=False)[F.TABULAR_FEATURES]
    pd.testing.assert_frame_equal(a, b)


def test_sequences_no_leakage():
    df = synth()
    pos = 24 * 20
    X1, _ = F.build_sequences(F.make_feature_frame(df), [pos], with_target=False)
    df2 = df.copy(); df2.iloc[pos + 1:, 0] += 99999
    X2, _ = F.build_sequences(F.make_feature_frame(df2), [pos], with_target=False)
    assert np.allclose(X1, X2)
    assert (X1[0, 24:, 0] == 0).all() and (X1[0, 24:, 1] == 1).all()


def test_target_alignment():
    df = synth(); feat = F.make_feature_frame(df); pos = 24 * 20
    rows = F.build_tabular(feat, [pos])
    assert (rows["target_time"] - rows["origin_time"]).dt.total_seconds().div(3600).tolist() == list(range(1, 25))
    assert np.allclose(rows["y"].to_numpy(), df[C.LOAD_COL].to_numpy()[pos + 1: pos + 25])


def test_splits_chronological_and_disjoint():
    idx = pd.date_range("2018-01-01 01:00", "2025-01-01", freq="h")
    df = pd.DataFrame({c: 1.0 for c in [C.LOAD_COL] + C.WEATHER_COLS}, index=idx)
    sp = D.split_origins(df, D.valid_origin_positions(df), stride=3)
    tr, va, te = (df.index[sp[k]] for k in ("train", "val", "test"))
    assert tr.max() + pd.Timedelta(hours=24) < pd.Timestamp(C.TRAIN_END)
    assert va.min() >= pd.Timestamp(C.TRAIN_END) and va.max() + pd.Timedelta(hours=24) < pd.Timestamp(C.VAL_END)
    assert te.min() >= pd.Timestamp(C.VAL_END)


def test_derived_features_follow_edits():
    df = synth(); base = df.iloc[-48:][C.WEATHER_COLS].reset_index(drop=True)
    hot = S.apply_scenario(base.iloc[24:].reset_index(drop=True), S.Scenario(temp_delta_c=8, humidity_delta_pct=10, wind_scale=0.5))
    edited = pd.concat([base.iloc[:24], hot], ignore_index=True)           # only the last 24h edited
    f0, f1 = F.add_weather_derived(base).iloc[24:], F.add_weather_derived(edited).iloc[24:]
    assert (f1["cdd"] >= f0["cdd"] - 1e-9).all() and (f1["tmpc_max24"] >= f0["tmpc_max24"] - 1e-9).all() \
        and f1["tmpc_max24"].iloc[-1] > f0["tmpc_max24"].iloc[-1]       # trailing window fully inside the edit
    assert (hot["relh"] <= 100).all() and (hot["sped"] >= 0).all()
    assert np.allclose(hot["tmpc"] - base["tmpc"].iloc[24:].to_numpy(), 8)
    assert not np.allclose(f1["feel"], f0["feel"])                          # feels-like recomputed from edits


def test_scenario_hour_window():
    df = synth(); base = df.iloc[-24:][C.WEATHER_COLS].reset_index(drop=True)
    out = S.apply_scenario(base, S.Scenario(temp_delta_c=5, hour_start=10, hour_end=14))
    d = out["tmpc"] - base["tmpc"]
    assert np.allclose(d[9:14], 5) and np.allclose(d[:9], 0) and np.allclose(d[14:], 0)


class _Stub:
    """Fake model: load rises with temperature (no training involved)."""
    def predict(self, X):
        return 30000 + 800 * X["cdd"].to_numpy() + 0.2 * X["load_lag24"].to_numpy()


def test_simulator_end_to_end_with_stub():
    df = synth(60)
    fc = S.Forecaster(df, S.tabular_predictor(_Stub()))
    origin = df.index[24 * 40]
    res = fc.compare(origin, S.Scenario(temp_delta_c=5))
    h = res["hourly"]
    assert len(h) == 24 and (h["delta_mw"] >= -1e-9).all() and res["summary"]["peak_delta_mw"] > 0
    ident = fc.compare(origin, S.Scenario())
    assert np.allclose(ident["hourly"]["delta_mw"], 0)


def test_seq_scaler_train_only_and_mask():
    from gridshift.deep import SeqScaler
    df = synth(60); feat = F.make_feature_frame(df)
    Xa, ya = F.build_sequences(feat, np.arange(200, 700, 3))
    Xb, _ = F.build_sequences(feat, np.arange(900, 1100, 3))
    sc = SeqScaler().fit(Xa)
    Za = sc.transform_X(Xa)
    assert Za.shape == (len(Xa), 48, len(F.SEQ_FEATURES)) and (Za[:, 24:, 0] == 0).all()
    assert abs(Za[:, :24, 0].mean()) < 1e-3                      # standardised on train history
    assert np.allclose(sc.inverse_y(sc.transform_y(ya)), ya, rtol=1e-4)
    assert not np.allclose(sc.transform_X(Xb)[:, :24, 0].mean(), 0, atol=1e-6)  # test not re-fit


def _raw_with_timestamp(df):
    r = df.reset_index(); r["timestamp"] = r["timestamp"].dt.strftime("%m/%d/%Y %H:%M"); return r


def test_missing_load_and_precip_preserved_not_filled():
    df = synth(30)
    df.iloc[100:103, df.columns.get_loc(C.LOAD_COL)] = np.nan      # 3h load gap
    df.iloc[200:202, df.columns.get_loc("p01m")] = np.nan          # precip gap
    df.iloc[300:302, df.columns.get_loc("tmpc")] = np.nan          # short temp gap (interpolable)
    df.iloc[400:420, df.columns.get_loc("tmpc")] = np.nan          # long temp gap (not filled)
    out = D.clean_hourly(_raw_with_timestamp(df))
    assert out[C.LOAD_COL].iloc[100:103].isna().all()               # load never interpolated
    assert out["p01m"].iloc[200:202].isna().all()                   # precip never zero-filled
    assert out["tmpc"].iloc[300:302].notna().all()                  # short weather gap interpolated
    assert out["tmpc"].iloc[400:420].isna().all()                   # long gap preserved


def test_clean_load_not_future_aware():
    """Cleaned load up to k must be identical whether or not later data exists."""
    df = synth(30); df.iloc[100:103, 0] = np.nan; df.iloc[500, 0] = np.nan
    full = D.clean_hourly(_raw_with_timestamp(df))
    part = D.clean_hourly(_raw_with_timestamp(df.iloc[:400]))
    pd.testing.assert_series_equal(full[C.LOAD_COL].iloc[:400], part[C.LOAD_COL], check_freq=False)


def test_valid_origins_match_bruteforce():
    df = synth(30)
    df.iloc[150, df.columns.get_loc(C.LOAD_COL)] = np.nan
    df.iloc[400, df.columns.get_loc("p01m")] = np.nan
    df.iloc[430, df.columns.get_loc("relh")] = np.nan
    got = set(D.valid_origin_positions(df, history=C.HISTORY_HOURS).tolist())
    H = C.HORIZON
    load_pos = lambda o: list(range(o - 23, o + H + 1)) + list(range(o + 1 - 168, o + H + 1 - 168))
    wx_pos = lambda o: list(range(o - 24, o + H + 1))
    want = set()
    for o in range(C.HISTORY_HOURS, len(df) - H):
        if (not df[C.LOAD_COL].iloc[load_pos(o)].isna().any()
                and not df[C.WEATHER_COLS].iloc[wx_pos(o)].isna().any().any()):
            want.add(o)
    assert got == want
    assert 150 + 5 not in got and 400 + 5 not in got      # excluded
    assert len(got) < len(range(C.HISTORY_HOURS, len(df) - H))


def test_simulator_rejects_missing_inputs():
    df = synth(60); df.iloc[24 * 40 + 5, df.columns.get_loc("p01m")] = np.nan   # inside future window
    fc = S.Forecaster(df, S.tabular_predictor(_Stub()))
    assert df.index[24 * 40] not in fc.origin_choices()
    try:
        fc.forecast(df.index[24 * 40]); assert False
    except ValueError:
        pass


def test_feels_like_reference_values_and_units():
    s = lambda v: pd.Series([v], dtype=float)
    hi = F.feels_like_f(s(35.0), s(50.0), s(5.0)).iloc[0]                  # 95F, 50% RH -> NWS table ~105F
    assert 103.5 < hi < 106.5, hi
    wc = F.feels_like_f(s((20 - 32) * 5 / 9), s(50.0), s(15.0)).iloc[0]    # 20F, 15 mph -> NWS ~6F
    assert 4.5 < wc < 7.5, wc
    assert abs(F.feels_like_f(s(15.0), s(50.0), s(10.0)).iloc[0] - 59.0) < 1e-9   # mild: = air temp in F
    assert np.isnan(F.feels_like_f(s(np.nan), s(50.0), s(5.0)).iloc[0])
    old = C.WIND_UNIT
    try:                                  # wind unit conversion is honoured: 13.03 kt == 15 mph
        C.WIND_UNIT = "knots"
        kt = F.feels_like_f(s((20 - 32) * 5 / 9), s(50.0), s(15.0 / 1.15078)).iloc[0]
    finally:
        C.WIND_UNIT = old
    assert abs(kt - wc) < 1e-6


def test_precip_6h_nan_when_any_hour_missing():
    df = synth(5); df.iloc[50, df.columns.get_loc("p01m")] = np.nan
    f = F.add_weather_derived(df)
    assert f["precip_6h"].iloc[50:56].isna().all() and f["precip_6h"].iloc[56:].notna().all()


# ----------------------------------------------------------------------------------------------
# Missing-data policy
# ----------------------------------------------------------------------------------------------
def test_load_never_filled_with_zero_mean_or_interpolation():
    df = synth(10); df.iloc[50:52, 0] = np.nan; df.iloc[0, 0] = np.nan; df.iloc[-1, 0] = np.nan
    out = D.clean_hourly(_raw_with_timestamp(df))
    assert out[C.LOAD_COL].isna().sum() == 4 and out[C.LOAD_COL].dropna().min() > 1000


def test_zero_precip_stays_zero_and_missing_stays_nan():
    df = synth(10); df.iloc[60:63, df.columns.get_loc("p01m")] = np.nan
    df.iloc[70:75, df.columns.get_loc("p01m")] = 0.0
    out = D.clean_hourly(_raw_with_timestamp(df))
    assert out["p01m"].iloc[60:63].isna().all()          # missing != 0
    assert (out["p01m"].iloc[70:75] == 0).all()          # measured zero kept
    assert out["p01m"].isna().sum() == 3


def test_short_gap_policy_configurable_and_counted():
    df = synth(10)
    df.iloc[100:102, df.columns.get_loc("tmpc")] = np.nan    # 2h gap
    df.iloc[150:153, df.columns.get_loc("tmpc")] = np.nan    # 3h gap
    out, rep = D.clean_with_report(_raw_with_timestamp(df))          # default max gap = 2
    assert C.WEATHER_MAX_GAP == 2 and rep["weather_max_gap_hours"] == 2
    assert out["tmpc"].iloc[100:102].notna().all() and out["tmpc"].iloc[150:153].isna().all()
    assert rep["interpolated_weather_values"]["tmpc"] == 2 and rep["missing_after_cleaning"]["tmpc"] == 3
    out3, rep3 = D.clean_with_report(_raw_with_timestamp(df), max_gap=3)
    assert out3["tmpc"].iloc[150:153].notna().all() and rep3["interpolated_weather_values"]["tmpc"] == 5
    out0, _ = D.clean_with_report(_raw_with_timestamp(df), max_gap=0)
    assert out0["tmpc"].iloc[100:102].isna().all()


def test_weather_not_extrapolated_at_ends():
    df = synth(10); df.iloc[:2, df.columns.get_loc("relh")] = np.nan; df.iloc[-2:, df.columns.get_loc("relh")] = np.nan
    out = D.clean_hourly(_raw_with_timestamp(df))
    assert out["relh"].iloc[:2].isna().all() and out["relh"].iloc[-2:].isna().all()


def test_rolling_features_never_use_incomplete_windows():
    df = synth(10)
    df.iloc[100, 0] = np.nan; df.iloc[120, df.columns.get_loc("tmpc")] = np.nan
    f = F.make_feature_frame(df)
    assert f["load_roll24_mean"].iloc[100:124].isna().all() and f["load_roll24_mean"].iloc[124:].notna().all()
    assert f["tmpc_max24"].iloc[120:144].isna().all() and f["tmpc_max24"].iloc[144] == df["tmpc"].iloc[121:145].max()
    assert f["tmpc_chg3"].iloc[:3].isna().all()                    # no artificial 0 change at the start
    assert f["tmpc_chg3"].iloc[[120, 123]].isna().all() and f["tmpc_chg3"].iloc[[121, 122]].notna().all()  # NaN only where defined


def test_units_declared_and_validated():
    assert C.UNITS == {"tmpc": "degC", "relh": "percent", "sped": "mph", "p01m": "mm per hour", "feel": "degF", "ERCOT": "MW"}
    assert C.WIND_UNIT == "mph" and C.UNITS["sped"] == C.WIND_UNIT
    ok = synth(5); D.validate_units(ok)
    for col, f in [("tmpc", lambda x: x * 9 / 5 + 32 + 40), ("relh", lambda x: x / 100)]:   # degF-like / fraction
        bad = ok.copy(); bad[col] = f(bad[col])
        try:
            D.validate_units(bad); assert False, col
        except ValueError:
            pass


def test_derived_features_identical_in_training_and_simulation_paths():
    """Same functions everywhere: simulator's per-hour features at the forecast hours == training frame's."""
    df = synth(60)
    fc = S.Forecaster(df, S.tabular_predictor(_Stub()))
    o = df.index[24 * 40]
    sim = fc.forecast(o)                                           # baseline = observed weather
    train = F.make_feature_frame(df).loc[sim.index]
    for c in ["feel"]:
        assert np.allclose(sim[c], train[c])
    full = F.make_feature_frame(df)
    feat, opos = fc._frame(o, fc.baseline_weather(o))
    cols = F.WEATHER_FEATURES + F.CALENDAR_FEATURES
    pd.testing.assert_frame_equal(feat.iloc[opos + 1: opos + 25][cols], full.loc[feat.index[opos + 1: opos + 25], cols],
                                  check_freq=False)


# ----------------------------------------------------------------------------------------------
# Scenario simulator guarantees
# ----------------------------------------------------------------------------------------------
class _Recorder:
    """Stub model that records every feature matrix it is asked to predict."""
    def __init__(self): self.X = []
    def predict(self, X):
        self.X.append(X.copy()); return 30000 + 800 * X["cdd"].to_numpy() + 0.2 * X["load_lag24"].to_numpy()


def test_identity_scenario_equals_baseline_exactly():
    df = synth(60); fc = S.Forecaster(df, S.tabular_predictor(_Stub()))
    o = df.index[24 * 40]
    r = fc.compare(o, S.Scenario())
    assert np.array_equal(r["hourly"]["baseline_mw"].to_numpy(), r["hourly"]["scenario_mw"].to_numpy())
    assert np.array_equal(r["hourly"]["baseline_mw"].to_numpy(), fc.forecast(o)["forecast_mw"].to_numpy())


def test_scenario_changes_only_selected_weather_not_history():
    df = synth(60); rec = _Recorder(); fc = S.Forecaster(df, S.tabular_predictor(rec))
    o = df.index[24 * 40]
    fc.compare(o, S.Scenario(temp_delta_c=6, hour_start=10, hour_end=14))
    base, scen = rec.X[0].reset_index(drop=True), rec.X[1].reset_index(drop=True)
    assert base[F.DEMAND_FEATURES].equals(scen[F.DEMAND_FEATURES])            # identical demand history / start
    assert base[F.CALENDAR_FEATURES + ["horizon"]].equals(scen[F.CALENDAR_FEATURES + ["horizon"]])
    for c in ["relh", "sped", "p01m"]:                                         # untouched variables
        assert base[c].equals(scen[c])
    d = (scen["tmpc"] - base["tmpc"]).to_numpy()
    assert np.allclose(d[9:14], 6) and np.allclose(d[:9], 0) and np.allclose(d[14:], 0)   # raw edit only in window
    assert (scen["cdd"] != base["cdd"]).sum() > 0 and (scen["feel"] != base["feel"]).sum() > 0  # derived recomputed
    assert (scen["tmpc_max24"] >= base["tmpc_max24"] - 1e-9).all()


def test_missing_weather_rejected_consistently():
    df = synth(60); fc = S.Forecaster(df, S.tabular_predictor(_Stub()))
    o = df.index[24 * 40]; w = fc.baseline_weather(o).copy(); w.iloc[3, 0] = np.nan
    for fn in (lambda: fc.forecast(o, w), lambda: fc.compare(o, S.Scenario(temp_delta_c=2), base_weather=w)):
        try:
            fn(); assert False
        except ValueError:
            pass


def test_models_share_identical_origins():
    """get_splits applies one origin filter for tabular AND sequence models; every kept origin is complete."""
    df = synth(60)
    df.iloc[24 * 30 + 3, df.columns.get_loc("tmpc")] = np.nan; df.iloc[24 * 45, 0] = np.nan
    feat = F.make_feature_frame(df)
    old = (C.TRAIN_END, C.VAL_END)
    try:
        C.TRAIN_END, C.VAL_END = "2022-01-25", "2022-02-05"
        sp, _ = F.get_splits(df, feat)
    finally:
        C.TRAIN_END, C.VAL_END = old
    allo = np.concatenate(list(sp.values()))
    assert len(allo) > 0
    tab = F.build_tabular(feat, allo); X, y = F.build_sequences(feat, allo)
    assert not tab[F.TABULAR_FEATURES + ["y"]].isna().any().any() and not np.isnan(X).any() and not np.isnan(y).any()
    assert len(tab) == len(allo) * C.HORIZON and X.shape[0] == len(allo)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)

def test_scenario_weather_values_are_clipped_to_valid_bounds():
    base = pd.DataFrame({
        "tmpc": np.full(24, 20.0),
        "relh": np.full(24, 80.0),
        "sped": np.full(24, 10.0),
        "p01m": np.full(24, 2.0),
    })

    upper = S.apply_scenario(
        base,
        S.Scenario(
            temp_delta_c=100,
            humidity_delta_pct=50,
            wind_scale=-1,
            precip_scale=-1,
            precip_add_mm=-2,
        ),
    )

    assert (upper["tmpc"] <= 55).all()
    assert (upper["tmpc"] >= -30).all()
    assert (upper["relh"] == 100).all()
    assert (upper["sped"] >= 0).all()
    assert (upper["sped"] == 0).all()
    assert (upper["p01m"] >= 0).all()
    assert (upper["p01m"] == 0).all()

    lower = S.apply_scenario(
        base,
        S.Scenario(temp_delta_c=-100, humidity_delta_pct=-200),
    )

    assert (lower["tmpc"] == -30).all()
    assert (lower["relh"] == 0).all()


def test_energy_delta_equals_sum_of_hourly_demand_deltas():
    df = synth(60)
    fc = S.Forecaster(df, S.tabular_predictor(_Stub()))
    origin = df.index[24 * 40]

    result = fc.compare(origin, S.Scenario(temp_delta_c=5))

    hourly = result["hourly"]
    energy_delta = result["summary"]["energy_delta_mwh"]

    assert len(hourly) == 24
    assert np.isclose(energy_delta, hourly["delta_mw"].sum())
