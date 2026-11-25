"""Step 1 - clean/align the hourly ERCOT + weather data, apply the missing-data policy, write a report.

Usage:
    python scripts/01_prepare_data.py                    # from Data/Final_dataset_ERCOT_v2.csv
    python scripts/01_prepare_data.py --from-raw         # rebuild from raw xlsx + ASOS csv (preserves gaps)
    python scripts/01_prepare_data.py --csv path/to.csv --confirm-units   # a different dataset: units must be confirmed
Outputs: artifacts/hourly_clean.csv, artifacts/reports/data_summary.json (+ printed summary)

Missing-data policy (details: gridshift/config.py, gridshift/data.py):
  ERCOT load  - never filled/interpolated; windows needing a missing hour are excluded
  p01m        - never zero-filled/interpolated (NaN kept, measured 0 kept); windows needing it are excluded
  tmpc/relh/sped - interpolated only for internal gaps <= WEATHER_MAX_GAP hours; no extrapolation
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from gridshift import config as C
from gridshift import data as D
from gridshift import features as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=C.PROCESSED_CSV)
    ap.add_argument("--from-raw", action="store_true")
    ap.add_argument("--max-gap", type=int, default=None, help="override config.WEATHER_MAX_GAP (hours)")
    ap.add_argument("--confirm-units", action="store_true",
                    help="I have verified units of this file match gridshift/config.py UNITS")
    ap.add_argument("--keep-upstream-imputed", action="store_true",
                    help="do NOT mask ERCOT hours that were interpolated upstream in the processed CSV")
    args = ap.parse_args()

    masked, xcheck = [], {}
    if args.from_raw:
        combined = D.rebuild_from_raw_frame()
        df, rep = D.clean_with_report(combined, args.max_gap)
        source = "raw ERCOT xlsx + ASOS weather csv"
        if C.PROCESSED_CSV.exists():  # sanity cross-check against the processed dataset
            ref = D.clean_hourly(pd.read_csv(C.PROCESSED_CSV))
            j = df.join(ref, how="inner", rsuffix="_ref")
            xcheck = {c: round(float(j[c].corr(j[c + "_ref"])), 4) for c in [C.LOAD_COL] + C.WEATHER_COLS}
    else:
        if args.csv.name not in C.UNITS_VERIFIED_FILES and not args.confirm_units:
            sys.exit(f"Units were only verified for {sorted(C.UNITS_VERIFIED_FILES)}. Check that '{args.csv.name}' uses "
                     f"{C.UNITS} (see gridshift/config.py), then re-run with --confirm-units.")
        df, rep = D.clean_with_report(pd.read_csv(args.csv), args.max_gap)
        source = str(args.csv)
        if not args.keep_upstream_imputed:
            df, masked = D.mask_upstream_imputed_load(df)
        rep["missing_after_cleaning"][C.LOAD_COL] = int(df[C.LOAD_COL].isna().sum())

    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.CLEAN_PARQUET)

    excl = D.origin_exclusion_report(df)
    feat = F.make_feature_frame(df)
    splits, dropped = F.get_splits(df, feat)
    summary = {
        "source": source,
        "rows": int(len(df)), "start": str(df.index.min()), "end": str(df.index.max()),
        "units": C.UNITS, "wind_unit_evidence": C.WIND_UNIT_VERIFIED,
        "cleaning": rep,
        "ERCOT_hours_imputed_upstream_and_masked_to_NaN": [str(t) for t in masked],
        "note_upstream": (None if args.from_raw else
                          "The processed CSV was linearly interpolated upstream for ALL columns (Data Preprocessing "
                          "notebook). ERCOT hours that were imputed are detected from the raw load files and masked; "
                          "imputed weather values (incl. precipitation) cannot be identified or undone here - use "
                          "--from-raw to preserve true gaps (its weather aggregation differs from the processed file)."),
        "raw_vs_processed_correlation": xcheck or None,
        "forecast_origins": {**excl, "origins_dropped_by_completeness_filter_after_split": dropped},
        "split_origins": {k: int(len(v)) for k, v in splits.items()},
        "split_tabular_rows(origins x 24h)": {k: int(len(v) * C.HORIZON) for k, v in splits.items()},
        "split_ranges": {k: [str(df.index[v[0]]), str(df.index[v[-1]])] for k, v in splits.items() if len(v)},
        "origin_stride_hours": C.ORIGIN_STRIDE,
    }
    (C.REPORTS_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    t = pd.DataFrame({"missing_before": rep["missing_before_cleaning"],
                      "interpolated": {**{c: 0 for c in rep["missing_before_cleaning"]}, **rep["interpolated_weather_values"]},
                      "missing_after": rep["missing_after_cleaning"]})
    print(f"Source: {source}\nRows: {len(df):,}  ({df.index.min()} -> {df.index.max()})  max weather gap filled: {rep['weather_max_gap_hours']} h")
    print(t.to_string())
    print(f"Missing precipitation retained as NaN: {rep['precipitation_missing_retained']}  | measured zeros kept: {rep['precipitation_zeros_kept']:,}")
    if masked:
        print(f"ERCOT hours imputed upstream and masked to NaN: {len(masked)}")
    print(f"Forecast origins: candidates {excl['candidate_origins']:,} | valid {excl['valid_origins']:,} | "
          f"discarded {excl['discarded_origins']:,}  (missing-input counts: {excl['discarded_where_variable_missing']})")
    print("Samples (origins, stride %d h) -> " % C.ORIGIN_STRIDE + ", ".join(
        f"{k}: {len(v):,} origins / {len(v) * C.HORIZON:,} rows" for k, v in splits.items()))
    print("Wrote", C.CLEAN_PARQUET, "and", C.REPORTS_DIR / "data_summary.json")


if __name__ == "__main__":
    main()
