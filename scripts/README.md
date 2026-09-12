# How to run

All scripts use relative paths (`data/`, `plots/`, `models/`), so run them from
the repository root. The three CEPPI Excel files are already present in `data/`;
script 00 uses `UPLOADS = "data"` by default.

```bash
mkdir -p data plots models
python3 scripts/00_extract_data.py            # build the cleaned CSVs from the raw Excel files
python3 scripts/01_train_fo_flux_longterm.py  # Model 1a -> Fig. 6A
python3 scripts/02_train_fo_flux_shortterm.py # Model 1b -> Fig. 3A
python3 scripts/03_train_ro_rejection.py      # Model 2  -> Fig. 2B
python3 scripts/04_train_toc_rejection.py     # Model 3  -> Table 4/5
python3 scripts/05_improved_fo_flux_longterm.py
python3 scripts/06_improved_fo_flux_shortterm.py
python3 scripts/07_improved_ro_rejection.py
python3 scripts/08_improved_toc_rejection.py
```

Requires: `pandas`, `numpy`, `openpyxl`, `scikit-learn`, `matplotlib`, `joblib`, `scipy`,
`xgboost` (only for script 05).

## Script order & purpose

| Script | Reads | Writes | Replicates |
|---|---|---|---|
| `scripts/00_extract_data.py` | 3 raw `.xlsx` files | `data/*.csv` | — (data prep) |
| `scripts/01_train_fo_flux_longterm.py` | `data/fo_flux_longterm_rep1_full.csv` | `plots/model1a_*.png`, `models/fo_flux_longterm_rf.joblib`, `models/metrics_fo_flux_longterm.txt` | Fig. 6A |
| `scripts/02_train_fo_flux_shortterm.py` | `data/fo_flux_shortterm.csv`, `data/sample_analysis.csv` | `plots/model1b_*.png`, `data/fo_flux_shortterm_enriched.csv`, `models/metrics_fo_flux_shortterm.txt` | Fig. 3A |
| `scripts/03_train_ro_rejection.py` | `data/ro_nacl_calibration.csv`, dataset B raw sheets | `plots/model2_*.png`, `models/metrics_ro_rejection.txt` | Fig. 2B |
| `scripts/04_train_toc_rejection.py` | `data/sample_analysis.csv` | `plots/model3_*.png`, `data/toc_rejection_derived.csv`, `models/metrics_toc_rejection.txt` | Table 4/5 |

Note: script 03 additionally reads `CEPPI_2025_dataset_B.xlsx` directly (not through
script 00) to pull TMP-stability context from the raw replicate-1 log — this is
inlined in the script itself rather than routed through `00_extract_data.py`.

## Improved models (05-08)

See `../IMPROVEMENTS_REPORT.md` for the full writeup of what was tried and why. Short
version: model choice was matched to sample size/structure per task rather than using
one "best" algorithm everywhere — feature engineering + proper time-series CV for the
709-row dataset, Gaussian Process Regression for the two small (n<15) per-point
datasets, and a physically-derived membrane transport model (not an ML model at all)
for the 3-point RO calibration, where no ML model can be honestly validated.

Run these after 00-04 (they read the CSVs 00 produces):

| Script | Upgrades | Result |
|---|---|---|
| `05_improved_fo_flux_longterm.py` | lag/rolling features, `TimeSeriesSplit` CV, tuned HistGradientBoosting & XGBoost | Holdout R² -0.05 → 0.57 (RF+features); best CV model: HistGradientBoosting |
| `06_improved_fo_flux_shortterm.py` | Gaussian Process Regression (1D, after fixing a feature-collinearity bug) | LOOCV R² 0.43 → 0.59 |
| `07_improved_ro_rejection.py` | Solution-diffusion physical model vs blind polynomial | Same fit in-range; polynomial is unphysical when extrapolated outside 30-50 bar, physical model isn't |
| `08_improved_toc_rejection.py` | Per-stream Gaussian Process Regression vs one-size-fits-all linear-CF | Mixed/stream-dependent — see report for which model to use per stream |

## Notebook versions

The root-level `*.ipynb` notebooks contain the same code as these `.py` scripts,
split into cells at blank-line boundaries with the module docstring as a markdown
title cell. They were generated from these `.py` files by
`scripts/_py_to_ipynb.py` — regenerate with:

```bash
python3 scripts/_py_to_ipynb.py
```

then execute in order (00 → 08) since each notebook reads files that an earlier
one writes.
