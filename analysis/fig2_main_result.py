"""
Figure 2. (a) worst-group accuracy against cue reliance for the 180
main-analysis models. (b) predicting models at a held-out rho level from cue
reliance, a linear probe, and rho itself. Plotted numbers are checked against
results/.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

import fixed_effects
import load_runs

INK, MUTED = "#333333", "#767676"
CIVIL, BIOS, GOOD, BAD = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"

# ---- panel (a): pooled 10-cell grid, cell-adjusted ----
d = fixed_effects.load()
d["z"] = d.groupby("cell").cd.transform(lambda s: (s - s.mean()) / s.std(ddof=1))
m = smf.ols("worst ~ z + C(cell)", d).fit()
beta = m.params["z"]
d["worst_adj"] = d.worst - d.groupby("cell").worst.transform("mean") + d.worst.mean()

# ---- panel (b): leave-one-rho-out, identical to validity_controls.c6 ----
h = load_runs.load().reset_index(drop=True)
r2 = {}
for key in ("cd", "rho", "probe"):
    yhat = np.full(len(h), np.nan)
    for hold in sorted(h.rho.unique()):
        tr, te = h[h.rho != hold], h[h.rho == hold]
        mm = smf.ols(f"worst ~ {key} + C(cell)", tr).fit()
        yhat[te.index] = mm.predict(te)
    y = h.worst.values
    r2[key] = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)

# pin to captured values (results/fixed_effects.txt, results/validity_controls.txt C6)
assert abs(beta - (-0.1129)) < 2e-3, beta
assert abs(r2["cd"] - 0.571) < 2e-3 and abs(r2["rho"] - (-0.182)) < 2e-3 \
    and abs(r2["probe"] - 0.037) < 2e-3, r2

plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.75),
                             gridspec_kw={"width_ratios": [1.15, 1]})

civ = d.cell.str.contains("civil")
ax.scatter(d.z[civ], d.worst_adj[civ], s=16, c=CIVIL, alpha=0.65, lw=0,
           label="CivilComments")
ax.scatter(d.z[~civ], d.worst_adj[~civ], s=18, c=BIOS, alpha=0.65, lw=0,
           marker="^", label="Bios")
xs = np.linspace(d.z.min(), d.z.max(), 2)
ax.plot(xs, d.worst.mean() + beta * xs, c=INK, lw=1.6)
ax.annotate("1 SD of reliance\n$\\approx$ 11 points of\nworst-group accuracy",
            xy=(1.3, d.worst.mean() + beta * 1.3), xycoords="data",
            xytext=(0.98, 0.98), textcoords="axes fraction",
            ha="right", va="top", ma="center", fontsize=8.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                      alpha=0.85),
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9,
                            shrinkA=4, shrinkB=3))
ax.set_xlabel("counterfactual cue reliance $\\Delta_{\\mathrm{cue}}$"
              "  (SD within cell)")
ax.set_ylabel("worst-group accuracy\n(cell-adjusted)")
ax.set_title("(a) 180 models, 10 populations:\nreliance $\\to$ failure",
             fontsize=10, color=INK, pad=6)
ax.legend(frameon=True, framealpha=0.85, edgecolor="none", fontsize=8.5,
          loc="lower left", handletextpad=0.2, borderaxespad=0.1)
ax.spines[["top", "right"]].set_visible(False)

names = ["$\\Delta_{\\mathrm{cue}}$\n(ask the model)",
         "linear probe\n(decodability)", "assigned training\ncorrelation $\\rho$"]
vals = [r2["cd"], r2["probe"], r2["rho"]]
cols = [GOOD, MUTED, BAD]
ypos = [2, 1, 0]
bx.barh(ypos, vals, color=cols, height=0.62)
bx.axvline(0, c=INK, lw=0.9)
for yp, v in zip(ypos, vals):
    # value labels always sit in the empty half-plane opposite the bar
    bx.text(v + 0.03 if v >= 0 else 0.03, yp, f"$R^2={v:+.2f}$",
            va="center", ha="left", fontsize=9.5, color=INK)
bx.set_yticks(ypos, names, fontsize=9)
bx.set_xlim(-0.32, 0.86)
bx.set_xlabel("extrapolation $R^2$ (held-out $\\rho$ level)", fontsize=9.5)
bx.set_title("(b) the measurement extrapolates,\nlinear extrapolation from $\\rho$ does not",
             fontsize=10, color=INK, pad=6)
bx.spines[["top", "right", "left"]].set_visible(False)
bx.tick_params(left=False)

fig.tight_layout(w_pad=2.2)
for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig2_main_result.{ext}", dpi=200, bbox_inches="tight")
print(f"hero figure written: beta={beta:+.4f}, R2={ {k: round(v,3) for k,v in r2.items()} }")
