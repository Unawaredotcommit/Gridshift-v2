# GridShift – run guide

Weather-aware ERCOT 24-hour load forecasting + what-if weather simulation
(Ridge / LightGBM / CNN-LSTM-Attention, SHAP & friends, Streamlit).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-gridshift.txt
```
Paths live in `gridshift/config.py` (override: env vars `GRIDSHIFT_DATA`, `GRIDSHIFT_ARTIFACTS`, `GRIDSHIFT_ROOT`).
Default input: `Data/Final_dataset_ERCOT_v2.csv` (raw ERCOT xlsx are used to detect upstream-imputed load hours).

## Run order
```bash
python -m pytest tests -q                    # 25 quick tests, seconds
python scripts/01_prepare_data.py            # cleaning + missing-data report -> artifacts/
python scripts/02_train_models.py            # Ridge + LightGBM (add --deep for CNN-LSTM-Attention; needs tensorflow)
python scripts/03_explain.py --quick         # fast smoke run, then without --quick
streamlit run streamlit_app.py
```
Step-01 output prints: missing values per variable before/after cleaning, interpolated weather values, retained
missing precipitation, valid/discarded forecast origins (with the variable responsible), and train/val/test sample
counts. The same content is saved in `artifacts/reports/data_summary.json`.

## Units (explicit; checked, never silently assumed)
| column | unit | | column | unit |
|---|---|---|---|---|
| `tmpc` | °C | | `p01m` | mm per hour |
| `relh` | % (0–100) | | `feel` (dataset & recomputed) | °F |
| `sped` | **mph** | | `ERCOT` | MW |

Wind = mph: the repo ships no units metadata, so this was established from (1) the IEM/ASOS download form
("Wind Speed [mph]", "Air Temperature [C]", "1 hour Precipitation [mm]", `feel` in °F) and (2) the data: raw `sped`
values are integer-knots × 1.15078 (3.45, 4.60, 5.75 …, i.e. already converted to mph) and the raw per-station `feel`
is reproduced by the NWS wind-chill formula with mph to ~0.04 °F MAE (as knots 0.6 °F, km/h 2.1 °F, m/s 3.8 °F).
Evidence is recorded in `gridshift/config.py`. Safeguards for other data: `01_prepare_data.py` refuses a CSV not in
`config.UNITS_VERIFIED_FILES` unless you pass `--confirm-units`, and `data.validate_units()` range-checks every run
(e.g. °F temperatures, fractional humidity, implausible wind all raise). Feels-like uses NWS formulas with
°C→°F, % RH and wind converted to mph via `config.WIND_UNIT`; tested against reference values (95 °F/50 % → ~105 °F;
20 °F/15 mph → ~6 °F). CDD/HDD use `config.BASE_TEMP_C` (18.3 °C).

## Missing-data policy
| variable | policy |
|---|---|
| `ERCOT` | never interpolated / zero- or mean-filled; stays NaN; any window that needs it is excluded |
| `p01m` | never zero-filled or interpolated (a missing reading is not evidence of no rain); NaN kept, measured 0 kept |
| `tmpc`, `relh`, `sped` | linear time interpolation only for **internal** gaps ≤ `WEATHER_MAX_GAP` (default **2 h**); no extrapolation at ends; longer gaps kept; count reported |

Derived features need complete windows: `tmpc_max24/min24` 24 valid hours, `precip_6h` 6 valid hours, `tmpc_chg3`
is NaN (not 0) when undefined; rolling demand statistics need all 24 hours. `valid_origin_positions()` checks only
the hours a forecast actually reads (load [o−23, o+24] and the last-week lag hours; weather [o−24, o+24]), so one
missing hour removes only the origins that truly depend on it.

**Important data caveat.** The processed CSV was linearly interpolated for *all* columns upstream (Data Preprocessing
notebook). We detect the 8 ERCOT hours that were imputed (7 spring-forward DST hours + 2022-12-01 01:00; exact
neighbour midpoints) from the raw load files and mask them. Imputed *weather* (including precipitation) in the
processed CSV cannot be identified or undone. `01_prepare_data.py --from-raw` rebuilds from raw files and preserves
true gaps, but its weather aggregation differs from the processed file (correlation with it: tmpc 0.99, sped 0.92,
relh 0.80, p01m 0.34), so use it as a sensitivity check, not as a drop-in replacement.

## Forecasting design and leakage audit
* **Task:** at origin hour T predict ERCOT for T+1…T+24 (direct multi-horizon; unchanged).
* **Demand inputs:** only hours ≤ T (origin load, lags 1/2/23, 24 h stats) plus same-hour-yesterday (t−24) and
  same-hour-last-week (t−168), which are ≤ T for every horizon ≤ 24 (asserted in code, unit-tested by corrupting future load).
* **Weather = known exogenous input.** Historical replays use *observed* weather in place of a forecast, so accuracy is
  optimistic vs. operating with real, imperfect weather forecasts.
* **Splits (chronological, by origin):** train = windows ending before 2023, validation = 2023, test = 2024; no window
  straddles a boundary. Ridge alpha and LightGBM/CNN early stopping use **validation only**; the test split is used
  only for the final metrics in step 02 (and, afterwards, to explain the finished model in step 03).
* **Fit-on-train only:** `StandardScaler` (Ridge pipeline) and `SeqScaler` (CNN-LSTM) are fit on train; no imputers exist.
  (The old notebook scaled on all data and was 1-step-ahead using the previous 24 h of real load.)
* **Comparable models:** `features.get_splits()` creates one origin set (valid origins, then any origin with a NaN in rows
  or sequence removed) used by Ridge, LightGBM *and* CNN-LSTM-Attention, so all are scored on identical origins/targets.

## How metrics are computed
Metrics in `artifacts/reports/metrics.json` are computed across **all (origin, horizon) rows of the test set**.
Forecast windows are **overlapping**: an origin is used every `ORIGIN_STRIDE` (3) hours and each window covers 24 h, so
every target hour is predicted by several windows at different horizons, and rows are strongly autocorrelated – the
pooled numbers are not independent-sample statistics. **Horizon-wise** metrics (h = 1…24) are written separately to
`metrics_by_horizon_<model>.csv` (also plotted on the Overview page); `peak_*` metrics are per 24 h window. Weather
for target hours is observed weather (see above).

## Explainability – how to read it
* SHAP, permutation importance, PDP, ALE and the sensitivity/interaction grids describe the **fitted model**:
  *model-attributed*, not causal effects of weather on demand.
* Temperature, feels-like, CDD/HDD and trailing extremes are highly correlated (`weather_feature_correlations.csv`), so
  single-feature scores split credit arbitrarily – prefer the **group** results. `feel` also depends on humidity and
  wind, so the "Temperature-derived" group is not a pure-temperature effect. Groups overlap (“All weather” contains the
  weather groups); do not add them.
* Group SHAP = mean over rows of |sum of member SHAP| (joint attribution). Group permutation shuffles members together
  (rows are shuffled across seasons/hours, a known limitation).
* **PDP** is scenario-based: the variable is set across the 24 h window and all derived features are recomputed with the
  normal pipeline. **ALE** shows the *direct* effect of one raw column with derived features held fixed.

## Scenario simulator – guarantees
Baseline and scenario share identical demand history and origin; only the chosen raw weather variables **and hours**
change; feels-like, CDD/HDD, trailing extremes, 3 h change and 6 h precipitation are recomputed by the same functions
as training (trailing features of hours after the edited window legitimately change). The model is never retrained;
an identity scenario reproduces the baseline exactly; missing weather raises an error instead of being filled. Results
are **model-based what-if estimates**, not proven causal effects. Historical-replay limitation: the baseline weather is
the observed weather of the next 24 h – no live forecast feed.

## Layout
```
gridshift/{config,data,features,metrics,models,deep,explain,simulator}.py
scripts/01_prepare_data.py  02_train_models.py  03_explain.py
streamlit_app.py   tests/test_core.py   artifacts/ (generated)   app.py (legacy, untouched)
```
