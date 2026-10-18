"""
Figure 1, the study design and the measurement. (a) one population of 18
models, six assigned rho levels by three seeds. (b) the counterfactual edit
that defines cue reliance, which needs no labels. (c) worst-group accuracy
over the four label-by-cue groups, which needs group labels. The counts in (a)
are checked against the logs, and the canvas is the printed size.
"""
import glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

LOGS = os.environ.get("LOGS", "logs")
OUT = os.environ.get("OUT", "figures")

INK, MUTED, MARK = "#333333", "#767676", "#ffe08a"
SEQ = LinearSegmentedColormap.from_list(            # the rho ramp of the population figure
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DC = r"$\Delta_{\mathrm{cue}}$"
T_TITLE, T_TEXT, T_NOTE, T_TICK = 8.3, 7.9, 7.3, 7.0   # printed sizes, as in Fig 2
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42})

# ------------------------------------------------------------------ design, from the logs
CELLS = ["civ2_qwen_r*_s*_11489602", "civ2_pythia_r*_s*_11494325", "civ2_pythia14_r*_s*_11520739",
         "civ2_qwen15_r*_s*_*", "biosv2_qwen_r*_s*_11498479", "biosv2_pythia_r*_s*_11502814",
         "biosv2_pythia14_r*_s*_11526403", "biosv2_qwen15_r*_s*_*", "civ3_olmo_r*_s*_*",
         "bios3_olmo_r*_s*_*"]
designs = set()
for pat in CELLS:
    runs = [re.search(r"RESULT rho=([\d.]+) seed=(\d+)", open(f).read())
            for f in sorted(glob.glob(f"{LOGS}/{pat}.log"))]
    designs.add((len(runs), tuple(sorted({float(m[1]) for m in runs})),
                 tuple(sorted({int(m[2]) for m in runs}))))
assert len(designs) == 1, designs                   # every population has the same design
n_models, LEVELS, SEEDS = designs.pop()
assert n_models == 18 and len(LEVELS) == 6 and len(SEEDS) == 3, (n_models, LEVELS, SEEDS)
assert LEVELS == (0.55, 0.65, 0.75, 0.85, 0.92, 0.97)
assert len(CELLS) == 10

# ------------------------------------------------------------------ canvas
W = 455.24 / 72.27 * 0.97            # printed width in inches (0.97 textwidth)
H = W / 3.342                         # same height as before, so the page layout holds
fig = plt.figure(figsize=(W, H))
cv = fig.add_axes([0, 0, 1, 1]); cv.set_xlim(0, W); cv.set_ylim(0, H); cv.axis("off")
R = fig.canvas.get_renderer()


def text(x, y, s, size=T_TEXT, color=INK, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center")
    return cv.text(x, y, s, fontsize=size, color=color, **kw)


def title(x, s):
    text(x, H - 0.03, s, T_TITLE, va="top", ma="center", linespacing=1.15)



XA, XB, XC = 0.98, 3.06, 5.12        # column centres
Y_LOW, Y_MID = 0.21, 0.39             # shared baselines of the bottom rows

# (a) one population: six assigned rho levels by three seeds
title(XA, "(a) fine-tune at an assigned\ncue–label correlation $\\rho$")
S, G = 0.19, 0.05                                    # square size and gap
gx0 = XA - (6 * S + 5 * G) / 2
for j, v in enumerate(LEVELS):
    for k in range(3):
        cv.add_patch(FancyBboxPatch((gx0 + j * (S + G), 0.62 + k * (S + G)), S, S, lw=0,
                                    boxstyle="round,pad=0,rounding_size=0.025",
                                    fc=SEQ(j / 5)))
    text(gx0 + j * (S + G) + S / 2, 0.51, f"{v:.2f}".lstrip("0"), T_TICK)
text(gx0 - 0.10, 0.62 + (3 * S + 2 * G) / 2, "3 seeds", T_TICK, rotation=90)
text(XA, Y_MID, "weak to strong shortcut", T_NOTE)
text(XA, Y_LOW, f"{n_models} models × {len(CELLS)} populations", T_NOTE, MUTED)

# (b) the counterfactual edit, asked of every finished model
title(XB, "(b) ask each finished model\nwith a counterfactual edit")
left = cv.text(0, 1.27, "a comment with an ", fontsize=T_TEXT, color=INK, va="center")
mark = cv.text(0, 1.27, "identity term", fontsize=T_TEXT, color=INK, va="center",
               bbox=dict(boxstyle="round,pad=0.15,rounding_size=0.12", fc=MARK, ec="none"))
wl, wm = (t.get_window_extent(R).width / fig.dpi for t in (left, mark))
left.set_x(XB - (wl + wm) / 2); mark.set_x(XB - (wl + wm) / 2 + wl)
text(XB, 1.110, "↓  replace the term with “people”", T_NOTE, MUTED)
text(XB, 0.950, "the same comment, cue neutralized")
text(XB, 0.795, "↓  score both with the model", T_NOTE, MUTED)
text(XB, 0.555, DC + " = average signed change", T_TEXT)
text(XB, Y_MID, "in the decision score", T_TEXT)
text(XB, Y_LOW, "no labels needed", T_NOTE, MUTED)

# (c) the outcome: worst-group accuracy over the four label x cue groups
title(XC, "(c) measure worst-group\naccuracy on four groups")
CW, CH, CG = 0.66, 0.23, 0.04                        # cell width, height, gap
gx = 6.05 - 2 * CW - CG                              # left edge of the 2 x 2 grid
GROUPS = [("toxic", (False, True)), ("non-toxic", (True, False))]   # misleads? (term, no term)
for c, hdr in enumerate(("identity term", "no term")):
    text(gx + c * (CW + CG) + CW / 2, 1.31, hdr, T_NOTE,
         bbox=dict(boxstyle="round,pad=0.15,rounding_size=0.12", fc=MARK, ec="none") if c == 0 else None)
for r, (lab, misleads) in enumerate(GROUPS):
    y0 = 1.19 - CH - r * (CH + CG)
    text(gx - 0.06, y0 + CH / 2, lab, T_NOTE, ha="right")
    for c, m in enumerate(misleads):
        cv.add_patch(FancyBboxPatch((gx + c * (CW + CG), y0), CW, CH, lw=0,
                                    boxstyle="round,pad=0,rounding_size=0.025",
                                    fc="#c9c9c9" if m else "#efefef"))
        text(gx + c * (CW + CG) + CW / 2, y0 + CH / 2, "cue misleads" if m else "cue agrees",
             T_TICK - 0.4, INK if m else MUTED)
text(XC, 0.555, "worst-group accuracy", T_TEXT)
q0 = cv.texts[-1].get_window_extent(R).x0 / fig.dpi           # left end of "worst-group accuracy"
q1 = [t for t in cv.texts if t.get_text().endswith("average signed change")][0]
q1 = q1.get_window_extent(R).x1 / fig.dpi                     # right end of the cue-reliance line
cv.add_patch(FancyArrowPatch((q1 + 0.06, 0.555), (q0 - 0.06, 0.555), arrowstyle="-|>",
                             mutation_scale=7.5, lw=1.0, ls=(0, (2.5, 1.6)), color=INK,
                             shrinkA=0, shrinkB=0))
text((q0 + q1) / 2, 0.632, "predicts?", T_NOTE, INK, style="italic")   # the question of Fig 2
text(XC, Y_MID, "= accuracy on the hardest group", T_NOTE)
text(XC, Y_LOW, "needs group labels", T_NOTE, MUTED)

# ------------------------------------------------------------------ layout checks
fig.canvas.draw()
R = fig.canvas.get_renderer()
ext = [(t.get_text(), t.get_window_extent(R)) for t in cv.texts if t.get_text()]
for s, e in ext:
    assert e.x0 >= 0 and e.y0 >= 0 and e.x1 <= W * fig.dpi and e.y1 <= H * fig.dpi, s
pair = {"a comment with an ", "identity term"}
clash = [(p, q) for k, (p, ep) in enumerate(ext) for q, eq in ext[k + 1:]
         if ep.overlaps(eq) and {p, q} != pair]
assert not clash, clash
for s, e in ext:                                      # every column keeps a gutter
    xm = (e.x0 + e.x1) / 2 / fig.dpi
    if s == "predicts?":
        continue
    lo, hi = {0: (0.0, 2.00), 1: (2.02, 4.12), 2: (4.15, W)}[int(xm > 2.01) + int(xm > 4.13)]
    assert lo <= e.x0 / fig.dpi and e.x1 / fig.dpi <= hi, (s, e.x0 / fig.dpi, e.x1 / fig.dpi)

os.makedirs(OUT, exist_ok=True)
fig.savefig(f"{OUT}/fig1_overview.pdf")
fig.savefig(f"{OUT}/fig1_overview.png", dpi=300)
print(f"fig1_overview written: {len(CELLS)} populations x {n_models} models, levels {LEVELS}, "
      f"seeds {SEEDS}; no text overlaps")
