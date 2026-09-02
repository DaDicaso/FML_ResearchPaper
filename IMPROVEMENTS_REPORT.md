# Model Upgrade Report: Where "Better" Models Actually Helped (and Where They Didn't)

The instruction was to bring more appropriate/updated models to improve accuracy. The
right way to do that is **not** "use the fanciest model everywhere" — it's matching
model complexity to sample size and problem structure, and being honest when a fancier
model doesn't actually win. Below is what was tried, what worked, and what didn't.

## Model 1a — FO flux, long-term (n=709, rich sensor features)

This is the only dataset large enough to genuinely benefit from more sophisticated ML.

**Changes:**
- Added lag features (Jw at t-1/t-3/t-5), rolling means of conductivity/temperature, and a
  draw/feed conductivity ratio — exploiting the fact this is an autocorrelated time series.
- Replaced random 5-fold CV with `TimeSeriesSplit`. This matters: random K-fold shuffles
  future rows into training folds for an autocorrelated series, which **overstates** how
  well a model generalizes. Switching to `TimeSeriesSplit` alone dropped the baseline
  RandomForest's apparent CV R² from 0.77 (reported earlier) to **-3.6** (!) — the original
  "0.77" was mostly measuring how well the model memorizes nearby timestamps, not real
  generalization.
- Compared RandomForest, tuned HistGradientBoostingRegressor, and tuned XGBoost, all under
  identical features/CV, via `RandomizedSearchCV`.

**Result:**

| Model | Holdout R² (final ~90 min, spans a cleaning event) | TimeSeriesSplit CV R² |
|---|---|---|
| RandomForest, baseline features | -0.05 | -3.58 |
| RandomForest, + engineered features | **0.57** | -0.05 |
| HistGradientBoosting, tuned | 0.38 | -0.03 |
| XGBoost, tuned | 0.34 | -0.35 |

Feature engineering (not a fancier algorithm) drove the real improvement — the same
RandomForest went from failing to extrapolate (-0.05) to actually tracking the end-of-run
dip (0.57) once it had lag/rolling context. HistGradientBoosting had the best (least
negative) CV score, so it's the recommended model, but the honest takeaway is that this
whole run is hard to forecast into its final segment because that dip is caused by a
discrete cleaning event, not a trend the sensors show in advance — no model should be
expected to fully solve that.

## Model 1b — FO flux, short-term (n=11)

**Changes:** Gaussian Process Regression instead of GradientBoosting.

**First attempt failed instructively:** using all 3 original features (time, recovery%,
feed conductivity) in a 3D GPR kernel *hurt* performance (R²=0.01) because those three
features are 93–97% correlated with each other (recovery% and conductivity were both
interpolated from time in the first place) — the kernel was trying to fit length-scales
along a near-degenerate direction with only 11 points. Dropping to a single time input
fixed it.

**Result:** LOOCV R² improved from 0.43 (GradientBoosting) to **0.59** (GPR, 1D). GPR also
gives a calibrated 95% uncertainty band, which is arguably more useful here than a point
estimate — the trade-off is that GPR's smoothness assumption means it blurs over the sharp
cartridge-filter-replacement jump at ~107 min that the tree model captured exactly (visible
in the plot as the widened uncertainty band right where the discontinuity occurs, rather
than a clean step).

## Model 2 — RO rejection vs pressure (n=3)

**Changes:** replaced the degree-2 polynomial with a 2-parameter solution-diffusion
membrane transport model (the standard RO physics: rejection = 1 − B/(A·Δp_net + B)).

With only 3 points, no model can be validated by held-out accuracy — both the polynomial
and the physical model reproduce the 3 calibration points almost perfectly. The real test
is **extrapolation behavior**:

| Pressure | Physical model | Polynomial (old) |
|---|---|---|
| 50 bar (calibrated) | 99.40% | 99.21% (paper: 99.2%) |
| 70 bar (extrapolation) | 99.67% | 95.66% — turns over and declines, unphysical |
| 20 bar (below the draw solution's own osmotic pressure) | correctly → 0% flux | 94.12% — nonsensical |

The polynomial isn't "wrong" inside the calibrated range, but it has no physical
grounding and gives nonsense the moment you step outside 30–50 bar. The physical model is
also more parsimonious in a real sense (2 physically-meaningful parameters vs. a polynomial
that has zero degrees of freedom left with 3 points and 3 coefficients).

## Model 3 — TOC/TDS/conductivity vs recovery (n=5-6 per stream)

**Changes:** Gaussian Process Regression per stream on raw recovery%, instead of forcing
every stream through the same linear-in-concentration-factor (CF) form.

**Result is genuinely mixed — reported as such, not cherry-picked:**

| Stream | Target | Linear-CF R² | GPR R² | Winner |
|---|---|---|---|---|
| Draw out | TOC | -2.07 | -0.53 | GPR (less bad) |
| Draw out | TDS | -2.25 | -1.20 | GPR (less bad) |
| Draw out | Conductivity | -2.26 | -1.22 | GPR (less bad) |
| Feed in | TOC | **0.98** | 0.56 | Linear-CF |
| Feed in | TDS | **0.98** | 0.35 | Linear-CF |
| Feed in | Conductivity | **0.95** | 0.42 | Linear-CF |
| Feed out | TOC | **0.89** | 0.48 | Linear-CF |
| Feed out | TDS | 0.81 | **0.86** | GPR |
| Feed out | Conductivity | 0.76 | **0.95** | GPR |

**Recommendation: use different models for different streams**, not GPR everywhere:
- **Feed-in**: linear-in-CF wins clearly. The feed side genuinely concentrates like a
  batch mass balance (paper's own framing), so the physically-motivated feature beats a
  flexible nonparametric model at this sample size.
- **Draw-out**: GPR is the better (or rather, less-bad) choice, because the true
  physical behavior — the paper's central claim — is that the draw solution does **not**
  follow a concentration-factor trend at all (RO keeps it roughly flat). Forcing a
  CF-linear shape onto flat data is the wrong inductive bias; GPR at least doesn't assume
  a shape it has to fit.
- **Feed-out**: mixed/ambiguous — reasonable to keep the simpler linear-CF model for
  interpretability given the small edge is not decisive at n=6.

## Bottom line

| Model | Genuine improvement? | What actually helped |
|---|---|---|
| 1a (long-term flux) | Yes | Feature engineering + honest CV (TimeSeriesSplit), not a fancier algorithm per se |
| 1b (short-term flux) | Yes | Switching to GPR, but only after fixing feature collinearity |
| 2 (RO rejection) | Yes (for extrapolation) | Physical constraints, not more data-fitting flexibility |
| 3 (TOC/TDS/cond) | Partial / stream-dependent | Right model chosen per-stream based on physical reasoning, not uniformly "the newer model" |

The two biggest lessons transfer beyond this project: (1) cross-validation strategy must
match data structure (time-series data needs `TimeSeriesSplit`, not random K-fold) — get
that wrong and you'll believe a model is much better than it is; and (2) at small sample
sizes, a physically-motivated inductive bias usually beats a more flexible model, unless
that physical assumption is actually wrong for the stream/regime in question.

## Files added
- `scripts/05_improved_fo_flux_longterm.py` (+ notebook)
- `scripts/06_improved_fo_flux_shortterm.py` (+ notebook)
- `scripts/07_improved_ro_rejection.py` (+ notebook)
- `scripts/08_improved_toc_rejection.py` (+ notebook)
- `plots/model1a_improved_comparison.png`, `model1b_improved_gpr.png`,
  `model2_improved_physical.png`, `model3_improved_gpr.png`
- `models/*_improved.txt`, `models/*_model_comparison.csv`
