# How to run

All scripts use relative paths (`data/`, `plots/`, `models/`), so run them from a
working directory that contains (or will contain) those three subfolders, with the
three CEPPI Excel files available at the path set by `UPLOADS` in script 00
(defaults to `/mnt/user-data/uploads`, edit if running elsewhere).

```bash
mkdir -p data plots models
python3 00_extract_data.py            # build the cleaned CSVs from the raw Excel files
python3 01_train_fo_flux_longterm.py  # Model 1a -> Fig. 6A
python3 02_train_fo_flux_shortterm.py # Model 1b -> Fig. 3A
python3 03_train_ro_rejection.py      # Model 2  -> Fig. 2B
python3 04_train_toc_rejection.py     # Model 3  -> Table 4/5
```

Requires: `pandas`, `numpy`, `openpyxl`, `scikit-learn`, `matplotlib`, `joblib`.

## Script order & purpose

| Script | Reads | Writes | Replicates |
|---|---|---|---|
| `00_extract_data.py` | 3 raw `.xlsx` files | `data/*.csv` | — (data prep) |
| `01_train_fo_flux_longterm.py` | `data/fo_flux_longterm_rep1_full.csv` | `plots/model1a_*.png`, `models/fo_flux_longterm_rf.joblib`, `models/metrics_fo_flux_longterm.txt` | Fig. 6A |
| `02_train_fo_flux_shortterm.py` | `data/fo_flux_shortterm.csv`, `data/sample_analysis.csv` | `plots/model1b_*.png`, `data/fo_flux_shortterm_enriched.csv`, `models/metrics_fo_flux_shortterm.txt` | Fig. 3A |
| `03_train_ro_rejection.py` | `data/ro_nacl_calibration.csv`, dataset B raw sheets | `plots/model2_*.png`, `models/metrics_ro_rejection.txt` | Fig. 2B |
| `04_train_toc_rejection.py` | `data/sample_analysis.csv` | `plots/model3_*.png`, `data/toc_rejection_derived.csv`, `models/metrics_toc_rejection.txt` | Table 4/5 |

Note: script 03 additionally reads `CEPPI_2025_dataset_B.xlsx` directly (not through
script 00) to pull TMP-stability context from the raw replicate-1 log — this is
inlined in the script itself rather than routed through `00_extract_data.py`.

## Notebook versions

`../notebooks/*.ipynb` contain the same code as these `.py` scripts, split into
cells at blank-line boundaries with the module docstring as a markdown title cell,
already executed once end-to-end (so plots/metrics are visible without re-running).
They were generated from these `.py` files by `_py_to_ipynb.py` — regenerate with:

```bash
python3 _py_to_ipynb.py   # writes notebooks/*.ipynb from the current .py files
```

then execute in order (00 → 04) since each notebook reads files that an earlier
one writes.
