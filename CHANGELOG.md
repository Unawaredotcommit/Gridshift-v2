# GridShift changelog (data-handling & audit revision)

| file | change |
|---|---|
| `gridshift/config.py` | explicit `UNITS` (tmpc °C, relh %, sped mph, p01m mm/h, feel °F, ERCOT MW) with source/evidence comments; `WIND_TO_MPH`; `UNITS_VERIFIED_FILES`; `BASE_TEMP_C`; documented missing-data policy; `WEATHER_MAX_GAP` 3 → **2**. |
| `gridshift/data.py` | no precipitation zero-fill; ERCOT never interpolated; per-column weather interpolation of internal gaps ≤ max-gap only (whole-gap rule, no extrapolation, no partial fill of long gaps); `clean_with_report()` (counts before/after/interpolated); `validate_units()`; `origin_ok_by_variable()` / `valid_origin_positions()` check exactly the hours read; `origin_exclusion_report()`; `mask_upstream_imputed_load()` (8 hours interpolated upstream in the CSV); `rebuild_from_raw_frame()`; raw-load reader handles `Hour_Ending`/`Hour Ending`. `clean_hourly()` signature kept. |
| `gridshift/features.py` | feels-like rewritten to full NWS procedure with explicit °F/mph conversion (hot-regime MAE vs raw feel 0.080→0.035 °F), NaN for missing inputs; CDD/HDD from config; `tmpc_max24/min24` need 24 valid hours; `precip_6h` needs 6; `tmpc_chg3` no zero fill; `complete_origin_mask()` + `get_splits()` (one origin set for all models). |
| `gridshift/simulator.py` | origins offered = valid origins only; missing history/future weather raises; validity uses same rule as training. Architecture unchanged. |
| `gridshift/explain.py` | joint (not summed) group SHAP, `SHAP_GROUPS`; non-causal / correlated-feature / direct-effect ALE documentation. |
| `scripts/01_prepare_data.py` | full transparency report (before/after missing, interpolated, retained precip NaN, valid/discarded origins by variable, split sizes), `--max-gap`, `--confirm-units`, upstream-imputed-load masking, `--from-raw`. |
| `scripts/02_train_models.py` | uses `get_splits()` (shared origins; no row-level dropna); protocol/overlap note. |
| `scripts/03_explain.py` | shared splits; interpretation notes; writes `weather_feature_correlations.csv`. |
| `streamlit_app.py` | captions: overlapping-window metrics, model-attributed (non-causal) explanations, correlated features, direct-effect ALE, what-if estimates, historical-replay limitation; correlation expander; joint group SHAP title. |
| `tests/test_core.py` | 14 → 25 tests (see README / PROGRESS). |
| `GRIDSHIFT_README.md`, `PROGRESS.md`, `CHANGELOG.md` | rewritten / new. |
