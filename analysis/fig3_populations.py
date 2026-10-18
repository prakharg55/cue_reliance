"""
Figure 3. Worst-group accuracy against cue reliance in each of the ten
populations, shaded by rho. Each panel's correlation is checked against the
range given in the caption.
"""
import glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr

F = r"([-+\d.]+)"
CELLS = [  # (display title, task row, log glob)
    ("Qwen 0.5B",   "CivilComments", "logs/civ2_qwen_r*_s*_11489602.log"),
    ("Pythia 410m", "CivilComments", "logs/civ2_pythia_r*_s*_11494325.log"),
    ("Pythia 1.4b", "CivilComments", "logs/civ2_pythia14_r*_s*_11520739.log"),
    ("Qwen 1.5B",   "CivilComments", "logs/civ2_qwen15_r*_s*_*.log"),
    ("Qwen 0.5B",   "Bios",          "logs/biosv2_qwen_r*_s*_11498479.log"),
    ("Pythia 410m", "Bios",          "logs/biosv2_pythia_r*_s*_11502814.log"),
    ("Pythia 1.4b", "Bios",          "logs/biosv2_pythia14_r*_s*_11526403.log"),
    ("Qwen 1.5B",   "Bios",          "logs/biosv2_qwen15_r*_s*_*.log"),
    ("OLMo 1B",     "CivilComments", "logs/civ3_olmo_r*_s*_*.log"),
    ("OLMo 1B",     "Bios",          "logs/bios3_olmo_r*_s*_*.log"),
]
SEQ = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
INK, MUTED = "#333333", "#767676"
RHOS = [0.55, 0.65, 0.75, 0.85, 0.92, 0.97]

def parse(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        t = open(f).read()
        m = re.search(rf"RESULT rho={F} seed=(\d+) worstgroup={F} advgroup={F} cueDelta={F}", t)
        if not m: continue
        row = dict(rho=float(m[1]), worst=float(m[3]), cd=float(m[5]))
        for key, rx in [("le", rf"LEACE .* delta_worst={F}"),
                        ("pl", rf"PROSPECT_RESULT .* delta_worst={F}"),
                        ("fx", rf"FIXED(?:20)? .* delta_worst={F}")]:
            mm = re.search(rx, t); row[key] = float(mm[1]) if mm else np.nan
        rows.append(row)
    return rows

if __name__ == "__main__":
    data = [(ti, ta, parse(p)) for ti, ta, p in CELLS]
    data = [(ti, ta, r) for ti, ta, r in data if len(r) == 18]
    tasks = list(dict.fromkeys(ta for _, ta, _ in data))
    ncol = max(sum(1 for _, ta, _ in data if ta == t) for t in tasks)

    fig, axes = plt.subplots(len(tasks), ncol, figsize=(1.75 * ncol + 0.9, 3.9),
                             sharey="row", squeeze=False)
    plt.rcParams.update({"font.size": 12})
    for r, task in enumerate(tasks):
        cells = [(ti, rows) for ti, ta, rows in data if ta == task]
        for c in range(ncol):
            ax = axes[r][c]
            if c >= len(cells):
                ax.axis("off"); continue
            ti, rows = cells[c]
            cd = np.array([x["cd"] for x in rows]); w = np.array([x["worst"] for x in rows])
            rho = np.array([x["rho"] for x in rows])
            b = np.polyfit(cd, w, 1); xs = np.linspace(cd.min(), cd.max(), 2)
            ax.plot(xs, np.poly1d(b)(xs), color=MUTED, lw=1.2, zorder=1)
            ax.scatter(cd, w, c=[SEQ(RHOS.index(v) / 5) for v in rho], s=42,
                       edgecolors="white", linewidths=0.8, zorder=2)
            rr, pp = pearsonr(cd, w)
            # pin the caption claim: per-cell r in [-0.94, -0.68], all p < 0.002
            assert -0.94 - 5e-3 <= rr <= -0.68 + 5e-3, (ti, task, rr)
            assert pp < 0.002, (ti, task, pp)
            rlab = f"{rr:+.2f}".replace("0.", ".")
            ax.set_title(f"{ti}  (r = {rlab})", fontsize=11, color=INK, pad=3)
            ax.grid(color="#e8e8e8", lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            for s in ("left", "bottom"): ax.spines[s].set_color("#cccccc")
            ax.tick_params(colors=MUTED, labelsize=7)
            if c == 0: ax.set_ylabel(f"{task}\nworst-group accuracy", fontsize=11, color=INK)
            if r == len(tasks) - 1: ax.set_xlabel("cue reliance (cue$\\Delta$)", fontsize=12, color=INK)
    handles = [plt.Line2D([], [], marker="o", ls="", markersize=6, color=SEQ(i / 5),
                          markeredgecolor="white", label=f"{v:.2f}") for i, v in enumerate(RHOS)]
    fig.legend(handles=handles, title="training correlation $\\rho$", loc="upper center",
               ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.13), fontsize=11, title_fontsize=12)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/fig3_populations.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("cells rendered:", [f"{ti}·{ta}" for ti, ta, _ in data])
