# GridShift progress

**Status: all requested code/doc changes of this revision are done (≈100 % of the requested scope). Nothing that trains
a model or runs the explain/simulation sweeps has been executed here – that is yours to run locally.**

## Verified here (lightweight)
- `python -m pytest tests -q` → **25 passed**; all modules/scripts compile.
- `01_prepare_data.py` on the processed CSV and with `--from-raw` (both run; reports below).
- Dashboard rendered headless (Streamlit AppTest) on all four pages with a throwaway dummy model: no exceptions (plumbing only).
- Stub-model smoke tests of the explain/simulator plumbing; joint group-SHAP unit check.

Observed data facts (processed CSV): 61,368 hourly rows; no raw gaps; 8 upstream-imputed ERCOT hours masked → 576 of
61,128 candidate origins discarded (all because of ERCOT); samples train 14,383 / val 2,888 / test 2,897 origins
(×24 rows). `--from-raw`: 35–80 weather and 44 precipitation hours missing after cleaning, 1,551 origins discarded
(p01m 1,222, relh 349, tmpc/sped 227, ERCOT 576; overlapping reasons).

## NOT executed (written only)
Ridge/LightGBM training, CNN-LSTM-Attention (TensorFlow absent here), SHAP/permutation/PDP/ALE/sensitivity on a real
model, the real dashboard with trained models.

## Done in this revision
Units documented + guarded; missing-data policy (load, precip, weather) implemented and reported; strict derived-feature
windows; shared origins for all models; leakage audit (README); joint group SHAP + interpretation labels; simulator
guarantees tested; 11 new tests; docs/changelog.

## Known limitations
1. Processed CSV weather (incl. precipitation) was interpolated upstream and can't be un-imputed; `--from-raw` preserves
   gaps but its weather aggregation differs from the processed file (p01m corr 0.34 – the original aggregation is unknown).
2. Observed weather stands in for a forecast → optimistic accuracy; no live forecast feed.
3. Metrics pool overlapping windows (stride 3 h); see horizon-wise files.
4. Permutation importance shuffles across seasons/hours; correlated temperature features share credit; ALE is direct-effect only.
5. Wind unit verified from IEM docs + data, not from repo metadata; units checks guard other files.
6. CNN-LSTM-Attention unverified (needs TensorFlow); its 48-step design differs from the notebook's 1-step model.
7. `feel` formula matches the dataset's to ~0.04 °F (hot) / ~0.2 °F (elsewhere), not exactly.
8. Strict complete-case windows: a missing precipitation hour removes ~50 surrounding origins.

## Next steps
Run the commands in `GRIDSHIFT_README.md` in order; send any traceback from the first real run of steps 02/03.
