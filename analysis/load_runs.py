"""
Loads the high-seed populations from logs/.

load() returns one row per model for the three high-seed cells, with 15 seeds
at each of rho = 0.65, 0.85 and 0.97 (the original three seeds plus 12 added
ones). Several analyses and figures import it. Running it directly prints the
fixed-effects models and a power analysis for these models.
"""
import glob, re, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, norm

warnings.filterwarnings("ignore")
F = r"([-+\d.]+)"
HS_RHOS = [0.65, 0.85, 0.97]

# each cell: frozen glob + high-seed glob (same script, same cell, more seeds)
CELLS = [
    ("Qwen0.5B-civil",   ["logs/civ2_qwen_r*_s*_11489602.log",   "logs/hs_civ2_qwen_r*_s*_*.log"]),
    ("Pythia410m-civil", ["logs/civ2_pythia_r*_s*_11494325.log", "logs/hs_civ2_pythia_r*_s*_*.log"]),
    ("Qwen0.5B-bios",    ["logs/biosv2_qwen_r*_s*_11498479.log", "logs/hs_biosv2_qwen_r*_s*_*.log"]),
]


def load():
    rows = []
    for name, pats in CELLS:
        seen = set()
        for pat in pats:
            for f in sorted(glob.glob(pat)):
                t = open(f).read()
                m = re.search(rf"RESULT rho={F} seed=(\d+) worstgroup={F} advgroup={F} cueDelta={F}", t)
                if not m:
                    continue
                rho, seed = float(m[1]), int(m[2])
                if rho not in HS_RHOS or (rho, seed) in seen:
                    continue
                seen.add((rho, seed))
                pm = re.search(rf"PROBE_RESULT rho=[\d.]+ seed=\d+ maxAUROC={F}", t)
                lm = re.search(rf"LEACE rho=[\d.]+ seed=\d+ L=\d+ trainAUROC_after=[\d.]+ "
                               rf"worst=[\d.]+ overall=[\d.]+ delta_worst={F}", t)
                rows.append(dict(cell=name, rho=rho, seed=seed, worst=float(m[3]), cd=float(m[5]),
                                 probe=float(pm[1]) if pm else np.nan,
                                 le_dw=float(lm[1]) if lm else np.nan))
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    zs = lambda s: (s - s.mean()) / s.std(ddof=1) if s.std(ddof=1) > 0 else s * 0
    d["zcd"] = d.groupby("cell")["cd"].transform(zs)
    d["zprobe"] = d.groupby("cell")["probe"].transform(zs)
    d["stratum"] = d.cell + "_" + d.rho.astype(str)
    return d


def main():
    d = load()
    if d.empty:
        print("no high-seed logs yet"); return
    print("== seeds per stratum ==")
    tab = d.groupby(["cell", "rho"]).size().unstack(fill_value=0)
    print(tab.to_string())
    n_per = tab.values.mean()
    print(f"\nN={len(d)} models, {d.stratum.nunique()} strata, mean {n_per:.1f} seeds/stratum")

    print("\n== screening-off, now with power ==")
    for tag, formula in (
        ("M1 no rho control      ", "worst ~ zcd + C(cell)"),
        ("M2 rho LINEAR          ", "worst ~ zcd + C(cell) + rho"),
        ("M3 rho DUMMIES (exact) ", "worst ~ zcd + C(cell) + C(rho)"),
        ("M4 cell x rho FE       ", "worst ~ zcd + C(stratum)"),
    ):
        m = smf.ols(formula, d).fit()
        b, se, p = m.params["zcd"], m.bse["zcd"], m.pvalues["zcd"]
        print(f"  {tag}: beta={b:+.4f} se={se:.4f} p={p:.3e} "
              f"95%CI[{b-1.96*se:+.4f},{b+1.96*se:+.4f}] "
              f"[{'SIGNIFICANT' if p < 0.05 else 'not significant'}]")

    # minimum detectable effect for M4 at 80% power, alpha=.05 -- makes a null readable
    m4 = smf.ols("worst ~ zcd + C(stratum)", d).fit()
    mde = (norm.ppf(0.975) + norm.ppf(0.80)) * m4.bse["zcd"]
    sd_w = d.worst.std(ddof=1)
    print(f"  minimum detectable |beta| at 80% power (M4): {mde:.4f} "
          f"= {mde/sd_w:.2f} SD of worst-group accuracy")

    print("\n== within-stratum correlation (rho removed exactly) ==")
    zs, ns = [], []
    for s, g in d.groupby("stratum"):
        if len(g) >= 5 and g.cd.std(ddof=1) > 0:
            r = pearsonr(g.cd, g.worst)[0]
            zs.append(np.arctanh(np.clip(r, -0.999, 0.999))); ns.append(len(g) - 3)
            print(f"  {s:26s} n={len(g):2d}  r={r:+.3f}")
    if zs:
        zbar = np.sum(np.array(zs) * np.array(ns)) / np.sum(ns)
        se = 1 / np.sqrt(np.sum(ns))
        print(f"  Fisher-z pooled: r={np.tanh(zbar):+.3f} "
              f"95%CI[{np.tanh(zbar-1.96*se):+.3f},{np.tanh(zbar+1.96*se):+.3f}] "
              f"p={2*norm.sf(abs(zbar/se)):.3e}")

    print("\n== the other two claims, under exact rho control ==")
    for outcome, extra, desc in (("worst", "zprobe", "cueDelta vs probe (used-vs-decodable)"),
                                 ("le_dw", None, "cueDelta -> LEACE benefit (repair targeting)")):
        sub = d.dropna(subset=[outcome] + ([extra] if extra else []))
        if len(sub) < 30:
            print(f"  [{desc}: {len(sub)} rows -- skipped]"); continue
        f = f"{outcome} ~ zcd + C(stratum)" + (f" + {extra}" if extra else "")
        m = smf.ols(f, sub).fit()
        line = (f"  {desc} (N={len(sub)}): cueDelta beta={m.params['zcd']:+.4f} "
                f"p={m.pvalues['zcd']:.3e}")
        if extra:
            line += f" | probe beta={m.params[extra]:+.4f} p={m.pvalues[extra]:.3f}"
        print(line + ("  [SIGNIFICANT]" if m.pvalues["zcd"] < 0.05 else "  [not significant]"))


if __name__ == "__main__":
    main()
