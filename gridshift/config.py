"""Central configuration. Edit paths/values here (or override with environment variables)."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("GRIDSHIFT_ROOT", Path(__file__).resolve().parents[1]))

# ---- Inputs -----------------------------------------------------------------
PROCESSED_CSV = Path(os.environ.get("GRIDSHIFT_DATA", ROOT / "Data" / "Final_dataset_ERCOT_v2.csv"))
RAW_LOAD_DIR = ROOT / "Data" / "Load data"            # Native_Load_YYYY.xlsx
RAW_WEATHER_CSV = ROOT / "Data" / "Weather data" / "asos_weather_data.csv"
LEGACY_SCALER = ROOT / "Save Model" / "scaler.pkl"    # old notebook scaler (unused; leaks)

# ---- Outputs ----------------------------------------------------------------
ARTIFACTS = Path(os.environ.get("GRIDSHIFT_ARTIFACTS", ROOT / "artifacts"))
CLEAN_PARQUET = ARTIFACTS / "hourly_clean.csv"        # CSV (no pyarrow dependency)
MODELS_DIR = ARTIFACTS / "models"
REPORTS_DIR = ARTIFACTS / "reports"
EXPLAIN_DIR = ARTIFACTS / "explain"

# ---- Columns ----------------------------------------------------------------
LOAD_COL = "ERCOT"
WEATHER_COLS = ["tmpc", "relh", "sped", "p01m"]       # user-editable raw weather

# ---- Units of the raw weather columns (explicit; used everywhere weather is computed) -----------------
UNITS = {
    "tmpc": "degC",            # air temperature
    "relh": "percent",         # relative humidity, 0-100
    "sped": "mph",             # wind speed  (== WIND_UNIT)
    "p01m": "mm per hour",     # 1-hour precipitation
    "feel": "degF",            # apparent temperature (dataset column AND our recomputed feature)
    "ERCOT": "MW",
}
WIND_UNIT = "mph"
WIND_TO_MPH = {"mph": 1.0, "knots": 1.15078, "kmh": 0.621371, "ms": 2.23694}   # factor -> mph
assert WIND_UNIT in WIND_TO_MPH and UNITS["sped"] == WIND_UNIT
# Evidence for these units (the repo itself ships NO units metadata):
#  * IEM/ASOS download form lists "Wind Speed [mph]", "Air Temperature [C]", "1 hour Precipitation [mm]"
#    and describes `feel` as apparent temperature in Fahrenheit
#    (https://mesonet.agron.iastate.edu/request/download.phtml).
#  * Empirical, Data/Weather data/asos_weather_data.csv: raw sped values are k*1.15078 for integer k
#    (0, 3.45, 4.60, 5.75 ...) i.e. knots already converted to mph; raw per-station `feel` is reproduced by
#    the NWS wind-chill formula with sped as mph to ~0.03-0.05 degF MAE (as knots: 0.6, km/h: 2.1, m/s: 3.8).
WIND_UNIT_VERIFIED = "IEM documentation + empirical check (see comment); not stated inside the repo"
# Units were verified ONLY for these files. Any other input file must be re-checked and then confirmed
# explicitly (scripts/01_prepare_data.py --confirm-units); data.validate_units() also range-checks every run.
UNITS_VERIFIED_FILES = {"Final_dataset_ERCOT_v2.csv", "asos_weather_data.csv"}

BASE_TEMP_C = 18.3        # CDD/HDD balance point (65 degF)

# ---- Missing-data policy ---------------------------------------------------------------------------
# * ERCOT load: never interpolated/filled. Missing stays NaN; windows needing it are excluded.
# * Precipitation (p01m): never zero-filled or interpolated. Missing stays NaN (a missing observation is not
#   evidence of no rain); windows needing it are excluded. Measured zeros stay zero.
# * tmpc / relh / sped (exogenous weather): linear time interpolation ONLY for internal gaps whose whole
#   length is <= WEATHER_MAX_GAP consecutive hours; never extrapolated at the ends; longer gaps preserved.
#   Set 0 to disable. Number of interpolated values is reported by scripts/01_prepare_data.py.
WEATHER_MAX_GAP = 2

# ---- Forecast setup ---------------------------------------------------------
HORIZON = 24                  # hours ahead
ORIGIN_STRIDE = 3             # use every 3rd hour as a forecast origin when training (data size/speed)
HISTORY_HOURS = 24 * 8 + 24   # history required before an origin (168h lag + slack)

# Chronological split on forecast-origin time (inclusive start, exclusive end).
TRAIN_END = "2023-01-01"      # train: origins < 2023-01-01
VAL_END = "2024-01-01"        # val:   2023
                              # test:  2024 (origins whose 24h window ends inside the data)

SEED = 42

# ---- LightGBM ---------------------------------------------------------------
LGBM_PARAMS = dict(
    n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=SEED, n_jobs=-1, verbose=-1,
)
LGBM_EARLY_STOPPING = 100

# ---- Ridge ------------------------------------------------------------------
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# ---- CNN-LSTM-Attention (optional; needs tensorflow) -------------------------
DEEP_EPOCHS = 60
DEEP_BATCH = 128
DEEP_LR = 3e-3
DEEP_PATIENCE = 8
