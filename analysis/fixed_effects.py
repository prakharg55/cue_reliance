"""
Table 4 and Section 7. The effect of cue reliance on worst-group accuracy under
four controls for rho, across the 180 main-analysis models:

  none         worst ~ cue reliance + cell
  linear       worst ~ cue reliance + cell + rho
  indicators   worst ~ cue reliance + cell + one indicator per rho level
  cell x rho   worst ~ cue reliance + one indicator per cell and rho level

Also tests how far the effect of rho is from linear and reports the
correlation within each cell. load() is reused by Figures 2 and 6.
"""
import glob, re, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
F = r"([-+\d.]+)"

CELLS = [
    ("Qwen0.5B-civil",   "logs/civ2_qwen_r*_s*_11489602.log"),
    ("Pythia410m-civil", "logs/civ2_pythia_r*_s*_11494325.log"),
    ("Qwen0.5B-bios",    "logs/biosv2_qwen_r*_s*_11498479.log"),
    ("Pythia410m-bios",  "logs/biosv2_pythia_r*_s*_11502814.log"),
    ("Pythia1.4b-civil", "logs/civ2_pythia14_r*_s*_11520739.log"),
    ("Pythia1.4b-bios",  "logs/biosv2_pythia14_r*_s*_11526403.log"),
    ("Qwen1.5B-civil",   "logs/civ2_qwen15_r*_s*_*.log"),
    ("Qwen1.5B-bios",    "logs/biosv2_qwen15_r*_s*_*.log"),
    ("OLMo1B-civil-v3",  "logs/civ3_olmo_r*_s*_*.log"),
    ("OLMo1B-bios-v3",   "logs/bios3_olmo_r*_s*_*.log"),
]


def load():
    rows = []
    for name, pat in CELLS:
        got = []
        for f in sorted(glob.glob(pat)):
            t = open(f).read()
            m = re.search(rf"RESULT rho={F} seed=(\d+) worstgroup={F} advgroup={F} cueDelta={F}", t)
            if m:
                got.append(dict(cell=name, rho=float(m[1]), seed=int(m[2]),
                                worst=float(m[3]), cd=float(m[5])))
        if len(got) == 18:
            rows += got
    d = pd.DataFrame(rows)
    d["zcd"] = d.groupby("cell")["cd"].transform(lambda s: (s - s.mean()) / s.std(ddof=1))
    d["cellrho"] = d.cell + "_" + d.rho.astype(str)
    return d


def load_extra():
    rows = []
    for name, pat in CELLS:
        got = []
        for f in sorted(glob.glob(pat)):
            t = open(f).read()
            m = re.search(rf"RESULT rho={F} seed=(\d+) worstgroup={F} advgroup={F} cueDelta={F}", t)
            if not m: continue
            pm = re.search(rf"PROBE_RESULT rho=[\d.]+ seed=\d+ maxAUROC={F}", t)
            lm = re.search(rf"LEACE rho=[\d.]+ seed=\d+ L=\d+ trainAUROC_after=[\d.]+ "
                           rf"worst=[\d.]+ overall=[\d.]+ delta_worst={F}", t)
            got.append(dict(cell=name, rho=float(m[1]), worst=float(m[3]), cd=float(m[5]),
                            probe=float(pm[1]) if pm else np.nan,
                            le_dw=float(lm[1]) if lm else np.nan))
        if len(got) == 18:
            rows += got
    d = pd.DataFrame(rows)
    zs = lambda s: (s - s.mean()) / s.std(ddof=1) if s.std(ddof=1) > 0 else s * 0
    d["zcd"] = d.groupby("cell")["cd"].transform(zs)
    d["zprobe"] = d.groupby("cell")["probe"].transform(zs)
    return d


def report(d, label):
    print(f"\n== {label}: N={len(d)} models, {d.cell.nunique()} cells ==")
    fits = {
        "M1 cueD + cell FE                ": "worst ~ zcd + C(cell)",
        "M2 cueD + cell FE + rho LINEAR   ": "worst ~ zcd + C(cell) + rho",
        "M3 cueD + cell FE + rho DUMMIES  ": "worst ~ zcd + C(cell) + C(rho)",
        "M4 cueD + cell*rho FE (seed only)": "worst ~ zcd + C(cellrho)",
    }
    for name, formula in fits.items():
        m = smf.ols(formula, d).fit()
        b, se, p = m.params["zcd"], m.bse["zcd"], m.pvalues["zcd"]
        star = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"  {name}: beta={b:+.4f} se={se:.4f} p={p:.3e}  R2={m.rsquared:.3f}  [{star}]")

    # attribution for M1: cell FE alone vs cueDelta's partial R2 on top
    m1 = smf.ols("worst ~ zcd + C(cell)", d).fit()
    m0 = smf.ols("worst ~ C(cell)", d).fit()
    pr2 = (m1.rsquared - m0.rsquared) / (1 - m0.rsquared)
    print(f"  M1 attribution: cell-FE-only R2={m0.rsquared:.3f}; "
          f"partial R2 of cueDelta over cell FE = {pr2:.3f}")

    lin = smf.ols("worst ~ C(cell) + rho", d).fit()
    dum = smf.ols("worst ~ C(cell) + C(rho)", d).fit()
    from scipy.stats import f as fdist
    df1 = dum.df_model - lin.df_model
    fstat = ((lin.ssr - dum.ssr) / df1) / (dum.ssr / dum.df_resid)
    print(f"  rho nonlinearity (dummies vs linear, no cueDelta): F({int(df1)},{int(dum.df_resid)})"
          f"={fstat:.2f} p={fdist.sf(fstat, df1, dum.df_resid):.3e}"
          f"   R2 {lin.rsquared:.3f} -> {dum.rsquared:.3f}")

    print("  raw per-cell correlation of cueDelta with worst (pooled over rho):")
    from scipy.stats import pearsonr as _pr
    for cell, g in d.groupby("cell"):
        rr, pp = _pr(g.cd, g.worst)
        print(f"    {cell:18s} n={len(g)}  r={rr:+.4f}  p={pp:.2e}")

    print("  per-cell within-rho (seed-level) correlation of cueDelta with worst:")
    for cell, g in d.groupby("cell"):
        rw = g.groupby("rho").apply(lambda x: pd.DataFrame(
            {"cd": x.cd - x.cd.mean(), "w": x.worst - x.worst.mean()})).reset_index(drop=True)
        if rw.cd.std(ddof=1) > 0:
            r = np.corrcoef(rw.cd, rw.w)[0, 1]
            print(f"    {cell:18s} r={r:+.3f}  (n={len(g)}, 3 seeds x 6 rho)")




def stress_other_claims(d2, label):
    """Re-test the other two claims that were established with LINEAR rho control."""
    print(f"\n== {label} ==")
    for outcome, extra, desc in (
        ("worst", "zprobe", "used-vs-decodable: does cueDelta beat probe AUROC"),
        ("le_dw", None, "repair targeting: does cueDelta predict LEACE benefit"),
    ):
        sub = d2.dropna(subset=[outcome] + ([extra] if extra else []))
        if len(sub) < 40:
            print(f"  [{desc}: only {len(sub)} rows -- skipped]"); continue
        print(f"  {desc}  (N={len(sub)})")
        for tag, formula in (
            ("rho LINEAR ", f"{outcome} ~ zcd + C(cell) + rho" + (f" + {extra}" if extra else "")),
            ("rho DUMMIES", f"{outcome} ~ zcd + C(cell) + C(rho)" + (f" + {extra}" if extra else "")),
        ):
            m = smf.ols(formula, sub).fit()
            b, p = m.params["zcd"], m.pvalues["zcd"]
            line = f"    {tag}: cueDelta beta={b:+.4f} p={p:.3e}"
            if extra:
                line += f"   | {extra} beta={m.params[extra]:+.4f} p={m.pvalues[extra]:.3f}"
            print(line + ("  [SIGNIFICANT]" if p < 0.05 else "  [not significant]"))


def mediation(d, label):
    """rho is experimentally assigned, so rho -> cueDelta -> worst is a mediation
    chain. Quantify how much of the (nonparametric) rho effect runs through
    cueDelta, using the extreme contrast rho=0.97 vs rho=0.55."""
    print(f"\n== {label}: mediation of the rho effect through cueDelta ==")
    hi, lo = 0.97, 0.55
    def contrast(df, with_med):
        f = "worst ~ C(cell) + C(rho)" + (" + zcd" if with_med else "")
        m = smf.ols(f, df).fit()
        key = [k for k in m.params.index if k.startswith("C(rho)[T.") and str(hi) in k]
        return m.params[key[0]] if key else np.nan
    tot, dir_ = contrast(d, False), contrast(d, True)
    print(f"  total effect  rho={hi} vs {lo} on worst-group: {tot:+.4f}")
    print(f"  direct effect controlling cueDelta:            {dir_:+.4f}")
    print(f"  proportion mediated by cueDelta:               {1 - dir_/tot:.1%}")
    rng = np.random.default_rng(0); props = []
    cells = d.cell.unique()
    for _ in range(2000):
        samp = pd.concat([d[d.cell == c].sample(frac=1, replace=True, random_state=int(rng.integers(1e9)))
                          for c in cells])
        try:
            t_, d_ = contrast(samp, False), contrast(samp, True)
            if np.isfinite(t_) and abs(t_) > 1e-6: props.append(1 - d_/t_)
        except Exception:
            pass
    lo_ci, hi_ci = np.percentile(props, [2.5, 97.5])
    print(f"  bootstrap 95% CI on proportion mediated:       [{lo_ci:.1%}, {hi_ci:.1%}]  (B={len(props)})")


if __name__ == "__main__":
    d = load()
    report(d, "all complete cells")
    mediation(d, "all complete cells")
    stress_other_claims(load_extra(), "stress test of the other two linear-control claims")
    print("\nVerdict is read off M3/M4: if cueDelta's coefficient survives exact rho\n"
          "control it screens off the recipe; if it dies there but lives in M2, the\n"
          "earlier claim rested on linear-control leakage and must be reframed as\n"
          "MEDIATION (rho -> cue reliance -> failure), which is the causally correct\n"
          "story anyway given that rho is experimentally assigned.")
