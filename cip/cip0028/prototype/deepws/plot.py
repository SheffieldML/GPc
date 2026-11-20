import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

deep = json.load(open("deep_results.json"))["results"]
tied = json.load(open("tied_scan.json"))
ws = json.load(open("ws_results.json"))["b"]
dims = json.load(open("dim_sweep.json"))
Ls = [1, 2, 4, 8, 16, 32, 64]

def series(res):
    return [res[str(L)]["spread"] for L in Ls]

curves = [
    ("GP prior, erf, Gaussian W", series(deep["GP prior, erf, Gaussian W"]), "#c0392b", "-"),
    ("GP prior, erf, orthogonal W", series(deep["GP prior, erf, orthogonal W"]), "#e67e22", "--"),
    ("GP prior, ReLU", series(deep["GP prior, ReLU, Gaussian W"]), "#8e44ad", ":"),
    ("GP prior, |det J| = 1 per layer", series(deep["GP prior, erf, det-normalised (|det J_l| = 1)"]), "#7f8c8d", "-."),
    ("Weight-space prior (M=256)", series(ws["WS prior, M=256, kappa=1.0, Gaussian centres"]["res"]), "#d35400", "-"),
    ("Mean path, tied, ReLU", series(deep["Mean path, tied orthogonal, ReLU"]), "#2980b9", "--"),
    ("Mean path, tied, erf, q*≈0.03", series([r for r in tied if abs(r["g2"] - 1.002) < 1e-9][0]["res"]), "#1abc9c", "-"),
    ("Residual GP, χ_f = 1", series(deep["Residual GP, chi_f = 1"]), "#27ae60", ":"),
    ("Residual GP, χ_f = 1/64", series(deep["Residual GP, chi_f = 1/L_max"]), "#16a085", "-"),
    ("Residual weight-space, χ_f = 1/64", series(ws["WS residual, M=256, kappa=0.5, chi_f=1/64"]["res"]), "#2c3e50", "--"),
]

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
for name, y, c, ls in curves:
    ax[0].plot(Ls, y, ls, color=c, marker="o", ms=3, label=name)
ax[0].set_xscale("log", base=2)
ax[0].set_yscale("log")
ax[0].set_xlabel("depth L")
ax[0].set_ylabel("sd of log singular values of J_L…J_1")
ax[0].set_title("Spread of the end-to-end Jacobian spectrum (D = 16, all tuned near χ = 1)")
ax[0].legend(fontsize=7.5, loc="upper left")
ax[0].grid(alpha=0.3, which="both")

for M, c in [(64, "#c0392b"), (1024, "#2980b9")]:
    d = [r for r in dims if r["M"] == M]
    ax[1].plot([r["D"] for r in d], [r["captured"] for r in d], "o-", color=c, label=f"M = {M} windows")
ax[1].set_xscale("log", base=2)
ax[1].set_xlabel("input dimension D (weight space is D+1)")
ax[1].set_ylabel("prior variance captured, Q(h,h)/k(h,h)")
ax[1].set_ylim(0, 1)
ax[1].set_title("Isotropic weight-space windows vs dimension (best width)")
ax[1].legend()
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("deep_ws_jacobians.png", dpi=150)
print("saved")
