"""
Figure 5. LEACE gain against cue reliance, with cell-by-rho means removed from
both. This is the first specification in Table 2.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

import load_runs

INK, MUTED = "#333333", "#767676"
CIVIL, BIOS = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

h = load_runs.load().reset_index(drop=True)
h = h.dropna(subset=["le_dw"]).reset_index(drop=True)
assert len(h) == 135, len(h)
h["z"] = h.groupby("cell").cd.transform(lambda s: (s - s.mean()) / s.std(ddof=1))
h["stratum"] = h.cell + "_" + h.rho.astype(str)
m = smf.ols("le_dw ~ z + C(stratum)", h).fit()
beta, p = m.params["z"], m.pvalues["z"]
# pinned to Table 2, row 1 (results/repair_targeting.txt)
assert abs(beta - 0.0417) < 5e-4 and abs(p - 0.0018) < 2e-4, (beta, p)
# Frisch-Waugh partial regression: a 2-D view of the fixed-effects fit
h["yr"] = h.le_dw - h.groupby("stratum").le_dw.transform("mean") + h.le_dw.mean()
h["xr"] = h.z - h.groupby("stratum").z.transform("mean")

fig, ax = plt.subplots(figsize=(3.2, 2.6))
civ = h.cell.str.contains("civil")
ax.scatter(h.xr[civ], h.yr[civ], s=15, c=CIVIL, alpha=0.65, lw=0,
           label="CivilComments")
ax.scatter(h.xr[~civ], h.yr[~civ], s=17, c=BIOS, alpha=0.65, lw=0,
           marker="^", label="Bios")
xs = np.linspace(h.xr.min(), h.xr.max(), 2)
ax.plot(xs, h.le_dw.mean() + beta * xs, c=INK, lw=1.6)
ax.axhline(0, c=MUTED, lw=0.8, ls=(0, (3, 3)))
ax.annotate(f"$\\beta={beta:+.3f}$, $p={p:.3f}$",
            xy=(0.03, 0.95), xycoords="axes fraction", va="top",
            fontsize=8.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none",
                      alpha=0.85))
ax.set_xlabel("$\\Delta_{\\mathrm{cue}}$ (SD within cell, stratum-adjusted)")
ax.set_ylabel("LEACE $\\Delta$ worst-group\n(stratum-adjusted)")
ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower center",
          bbox_to_anchor=(0.5, 1.0), handletextpad=0.15, columnspacing=1.0,
          borderaxespad=0.2, markerscale=1.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig5_repair_targeting.{ext}", dpi=200, bbox_inches="tight")
print(f"figure 5 written: beta={beta:+.4f} p={p:.4f}")
