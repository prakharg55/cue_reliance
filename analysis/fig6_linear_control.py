"""
Figure 6. (a) worst-group accuracy by rho in each population and on average.
(b) the coefficient of cue reliance under four controls for rho (Table 4).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

import fixed_effects

INK, MUTED = "#333333", "#767676"
BAD = "#d03b3b"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

d = fixed_effects.load()
assert len(d) == 180
# ladder estimates, pinned to results/fixed_effects.txt M1-M4
d["z"] = d.groupby("cell").cd.transform(lambda s: (s - s.mean()) / s.std(ddof=1))
d["cellrho"] = d.cell + "_" + d.rho.astype(str)
fits = [("none", "worst ~ z + C(cell)"),
        ("linear", "worst ~ z + C(cell) + rho"),
        ("indicators", "worst ~ z + C(cell) + C(rho)"),
        ("cell$\\times\\rho$ FE", "worst ~ z + C(cellrho)")]
est = [(name, m.params["z"], m.bse["z"], m.pvalues["z"])
       for name, f in fits for m in [smf.ols(f, d).fit()]]
for (name, b, se, p), (b0, se0) in zip(
        est, [(-0.1129, 0.0067), (-0.0796, 0.0134), (-0.0296, 0.0173), (-0.0288, 0.0167)]):
    assert abs(b - b0) < 5e-4 and abs(se - se0) < 5e-4, (name, b, se)

fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.2, 4.1),
                             gridspec_kw={"height_ratios": [1.15, 1]})
for cell, g in d.groupby("cell"):
    mm = g.groupby("rho").worst.mean()
    ax.plot(mm.index, mm.values, c=MUTED, lw=0.9, alpha=0.45)
mm = d.groupby("rho").worst.mean()
ax.plot(mm.index, mm.values, c=INK, lw=2.0, marker="o", ms=4)
ax.legend([plt.Line2D([], [], c=INK, lw=2.0, marker="o", ms=4),
           plt.Line2D([], [], c=MUTED, lw=0.9, alpha=0.45)],
          ["population mean", "the 10 cells"],
          frameon=False, fontsize=8, loc="lower left", handlelength=1.8,
          handletextpad=0.5, labelspacing=0.35, borderaxespad=0.25)
ax.set_xlabel("assigned training correlation $\\rho$")
ax.set_ylabel("worst-group accuracy")
ax.set_title("(a) $\\rho$'s effect is strongly nonlinear",
             fontsize=9.5, color=INK, pad=5)
ax.spines[["top", "right"]].set_visible(False)

ys = np.arange(len(est))[::-1]
for y, (name, b, se, p) in zip(ys, est):
    color = BAD if name == "linear" else INK
    bx.errorbar(b, y, xerr=1.96 * se, fmt="o", ms=5, color=color,
                ecolor=color, elinewidth=1.6, capsize=3)
    bx.text(b, y + 0.24, f"$p={'%.0e' % p if p < 1e-3 else '%.3f' % p}$",
            ha="center", fontsize=7.5, color=color)
bx.axvline(0, c=MUTED, lw=0.9)
bx.set_yticks(ys, [n for n, *_ in est], fontsize=8.5)
bx.set_xticks([-0.12, -0.08, -0.04, 0.0])
bx.set_xlabel("$\\beta_{\\Delta_{\\mathrm{cue}}}$ (95% CI)")
bx.set_title("(b) linear control makes the\neffect look significant",
             fontsize=9.5, color=INK, pad=5)
bx.spines[["top", "right", "left"]].set_visible(False)
bx.tick_params(left=False)
bx.set_ylim(-0.6, len(est) - 0.2)
fig.tight_layout(h_pad=2.0)
for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig6_linear_control.{ext}", dpi=200, bbox_inches="tight")

print("figure 6 written: coefficients pinned to results/fixed_effects.txt")
