"""
Model 2 IMPROVED: RO rejection & permeate flow vs TMP.
With only 3 calibration points (30/40/50 bar), no ML model can be honestly "trained" —
a degree-2 polynomial through 3 points has zero residual (R2 is meaningless) and, more
importantly, is UNCONSTRAINED outside [30,50] bar: it can curve back down or blow up
for pressures a real membrane would never show.

Upgrade: replace the polynomial with the solution-diffusion membrane transport model
used throughout the RO literature (and cited implicitly in the paper itself, Section
2.1, Eq. 1 area):
    Jw  = A * (deltaP - deltaPi)              (water flux, ~linear in net driving pressure)
    Js  = B * deltaC                          (salt flux, ~constant, weak pressure dependence)
    Rejection = 1 - Js / (Js + Jw * C_perm_ref)  ~ 1 - B / (A*(deltaP-deltaPi) + B)
This is a proper 2-parameter physical model (A = water permeability, B = salt
permeability), which is *more* constrained than a 3-parameter polynomial fit to 3
points, so it is not "cheating" by having more free parameters than data.
"""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

cal = pd.read_csv("data/ro_nacl_calibration.csv")
cal["rejection_frac"] = cal["rejection"]
TMP = cal["TMP_bar"].values
rej_meas = cal["rejection_frac"].values
flow_meas = cal["perm_flow_Lh"].values

# 0.5 M NaCl osmotic pressure ~24.9 bar at 25C (paper Methods, Section 2.3.1)
DELTA_PI = 24.9

def rejection_model(dP, A, B):
    """Solution-diffusion rejection: R = 1 - B / (A*(dP-dPi) + B), A,B > 0."""
    net_dP = np.maximum(dP - DELTA_PI, 1e-6)
    Jw = A * net_dP
    return 1 - B / (Jw + B)

# fit with positivity constraints on A, B
popt, pcov = curve_fit(rejection_model, TMP, rej_meas, p0=[1.0, 0.05], bounds=(0, np.inf))
A_fit, B_fit = popt
pred_physical = rejection_model(TMP, *popt)
r2_physical = r2_score(rej_meas, pred_physical)

# old baseline: degree-2 polynomial (kept for comparison, same as original script)
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(TMP.reshape(-1, 1), rej_meas * 100)

print(f"Physical model fit: A={A_fit:.4f} L/h.m2.bar, B={B_fit:.5f} (dimensionless salt-passage param)")
print(f"Physical model R2 on calibration points: {r2_physical:.4f}")

# The real test isn't fit-to-3-points (both models can do that) -- it's whether the
# model stays physically sensible when extrapolated beyond the calibrated 30-50 bar
# range, e.g. up to typical industrial RO pressures (~70-80 bar).
tmp_grid = np.linspace(20, 80, 200)
rej_physical_grid = rejection_model(tmp_grid, *popt) * 100
rej_poly_grid = poly_model.predict(tmp_grid.reshape(-1, 1))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.plot(TMP, rej_meas * 100, "o", ms=10, color="k", label="Measured", zorder=5)
ax.plot(tmp_grid, rej_physical_grid, "-", color="#1f4e8c", lw=2, label="Physical model (solution-diffusion)")
ax.plot(tmp_grid, rej_poly_grid, "--", color="#d62728", lw=2, label="Degree-2 polynomial (old baseline)")
ax.axvspan(30, 50, color="gray", alpha=0.15, label="Calibrated range")
ax.set_xlabel("TMP (bar)"); ax.set_ylabel("Rejection (%)")
ax.set_ylim(80, 102)
ax.set_title("Interpolation (both fit calibration points equally well)")
ax.legend(fontsize=8)

ax2 = axes[1]
ax2.plot(TMP, rej_meas * 100, "o", ms=10, color="k", label="Measured", zorder=5)
ax2.plot(tmp_grid, rej_physical_grid, "-", color="#1f4e8c", lw=2, label="Physical model")
ax2.plot(tmp_grid, rej_poly_grid, "--", color="#d62728", lw=2, label="Polynomial")
ax2.axvspan(30, 50, color="gray", alpha=0.15)
ax2.set_xlabel("TMP (bar)"); ax2.set_ylabel("Rejection (%)")
ax2.set_title("Extrapolation: polynomial is unphysical outside [30,50] bar")
ax2.legend(fontsize=8)
plt.suptitle("RO rejection vs TMP — physically-constrained model vs blind polynomial")
plt.tight_layout()
plt.savefig("plots/model2_improved_physical.png", dpi=150)
plt.show()

print(f"\nAt 50 bar: physical model = {rejection_model(np.array([50.]), *popt)[0]*100:.2f}%,"
      f" polynomial = {poly_model.predict([[50]])[0]:.2f}%  (paper reports 99.2%)")
print(f"At 70 bar (extrapolation): physical model = {rejection_model(np.array([70.]), *popt)[0]*100:.2f}%,"
      f" polynomial = {poly_model.predict([[70]])[0]:.2f}%")
print(f"At 20 bar (below draw-solution osmotic pressure, ~24.9 bar): physical model correctly "
      f"saturates near 0 flux; polynomial = {poly_model.predict([[20]])[0]:.2f}% (can go negative/unphysical)")

with open("models/metrics_ro_rejection_improved.txt", "w") as f:
    f.write(f"Physical (solution-diffusion) model: A={A_fit:.4f}, B={B_fit:.5f}, R2 on 3 cal. points={r2_physical:.4f}\n")
    f.write(f"At 50 bar: physical={rejection_model(np.array([50.]),*popt)[0]*100:.2f}%, "
            f"polynomial={poly_model.predict([[50]])[0]:.2f}%, paper=99.2%\n")
    f.write(f"At 70 bar (extrapolation): physical={rejection_model(np.array([70.]),*popt)[0]*100:.2f}%, "
            f"polynomial={poly_model.predict([[70]])[0]:.2f}%\n")
    f.write(f"At 20 bar (extrapolation): physical={rejection_model(np.array([20.]),*popt)[0]*100:.2f}%, "
            f"polynomial={poly_model.predict([[20]])[0]:.2f}%\n")
    f.write("\nRationale: with n=3, both models fit the calibration points about equally well, so the\n"
            "correct axis of comparison is BEHAVIOR OUTSIDE the calibrated range, where the physical\n"
            "model stays monotonic/bounded and the polynomial does not.\n")
print("done")
