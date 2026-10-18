"""
Figure 4. The placebo edit moves the models far less than the cue edit (a) and
does not predict worst-group accuracy (b). Plotted numbers are checked against
results/validity_controls.txt.
"""
import glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

INK, MUTED, CUE = "#333333", "#767676", "#2a78d6"
F = r"([-+\d.]+)"
rows = []
for f in sorted(glob.glob("logs/pbo_qwen_r*_s*_*.log")):
    t = open(f).read()
    m = re.search(rf"RESULT rho={F} seed=(\d+) worstgroup={F} advgroup={F} cueDelta={F}", t)
    p = re.search(rf"PLACEBO rho=[\d.]+ seed=\d+ placeboDelta={F} nEdited=(\d+)/(\d+)", t)
    if m and p:
        rows.append((float(m[5]), float(p[1]), float(m[3])))
cd, pb, w = (np.array(x) for x in zip(*rows))
ratio = np.abs(cd).mean() / np.abs(pb).mean()
r_cd, p_cd = pearsonr(cd, w)
r_pb, p_pb = pearsonr(pb, w)
# pin to results/validity_controls.txt C2
assert len(rows) == 18
assert abs(ratio - 40.3) < 0.15, ratio
assert abs(r_cd - (-0.905)) < 2e-3 and abs(r_pb - 0.080) < 2e-3, (r_cd, r_pb)

plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.35, 2.5),
                             gridspec_kw={"width_ratios": [0.72, 1]})

rng = np.random.default_rng(0)
for i, (vals, color, lab) in enumerate([(np.abs(cd), CUE, "cue edit"),
                                        (np.abs(pb), MUTED, "placebo")]):
    x = i + rng.uniform(-0.10, 0.10, len(vals))
    ax.scatter(x, vals, s=13, c=color, alpha=0.75, lw=0)
    ax.hlines(vals.mean(), i - 0.22, i + 0.22, color=color, lw=2)
ax.set_yscale("log")
ax.annotate("", xy=(0.5, np.abs(pb).mean()), xytext=(0.5, np.abs(cd).mean()),
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
ax.text(0.60, np.sqrt(np.abs(cd).mean() * np.abs(pb).mean()), "$40\\times$",
        fontsize=10, color=INK, va="center")
ax.set_xticks([0, 1], ["cue\nedit", "placebo\nedit"], fontsize=8.5)
ax.set_xlim(-0.55, 1.55)
ax.set_ylabel("|logit shift|  (log scale)")
ax.set_title("(a) the placebo\nbarely moves the model", fontsize=9, color=INK, pad=4)
ax.spines[["top", "right"]].set_visible(False)

zcd = (cd - cd.mean()) / cd.std(ddof=1)
zpb = (pb - pb.mean()) / pb.std(ddof=1)
bx.scatter(zpb, w, s=15, c=MUTED, alpha=0.7, lw=0, marker="^",
           label=f"placebo ($r{'{=}'}{r_pb:+.2f}$)")
bx.scatter(zcd, w, s=15, c=CUE, alpha=0.75, lw=0,
           label=f"cue edit ($r{'{=}'}{r_cd:+.2f}$)")
b1 = np.polyfit(zcd, w, 1)
x1 = np.linspace(zcd.min(), zcd.max(), 2)
bx.plot(x1, np.polyval(b1, x1), c=CUE, lw=1.6)
b0 = np.polyfit(zpb, w, 1)
x0 = np.linspace(zpb.min(), zpb.max(), 2)
bx.plot(x0, np.polyval(b0, x0), c=MUTED, lw=1.2, ls=(0, (3, 3)))
bx.set_xlabel("shift under edit (SD)")
bx.set_ylabel("worst-group accuracy")
bx.set_title("(b) and it predicts\nnothing", fontsize=9, color=INK, pad=4)
hnd, lab = bx.get_legend_handles_labels()
fig.legend(hnd[::-1], lab[::-1], loc="lower center", ncol=2, frameon=False,
           fontsize=7.5, bbox_to_anchor=(0.5, -0.09), handletextpad=0.15,
           columnspacing=1.0)
bx.spines[["top", "right"]].set_visible(False)

fig.tight_layout(w_pad=1.4)
for ext in ("pdf", "png"):
    fig.savefig(f"figures/fig4_placebo.{ext}", dpi=200, bbox_inches="tight")
print(f"placebo figure: n={len(rows)} ratio={ratio:.1f}x r_cd={r_cd:+.3f} r_pb={r_pb:+.3f}")
