"""
Section 5, Table 3 and Appendix E. The capability control, the placebo edit,
the split-half reliability of cue reliance, the instrumental-variable
analysis, early measurability, and leave-one-rho-out extrapolation.
"""
import glob, re, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
F = r"([-+\d.]+)"
RNG = np.random.default_rng(0)

BIOS = [("logs/biosv2_qwen_r*_s*_11498479.log", "Qwen-bios"),
        ("logs/biosv2_pythia_r*_s*_11502814.log", "Pythia-bios"),
        ("logs/biosv2_pythia14_r*_s*_11526403.log", "Pythia1.4b-bios"),
        ("logs/biosv2_qwen15_r*_s*_*.log", "Qwen1.5B-bios"),
        ("logs/hs_biosv2_qwen_r*_s*_*.log", "Qwen-bios")]


def load_bios():
    """bios logs carry all four group accuracies, so aligned/adversarial split is possible."""
    rows = []
    for pat, cell in BIOS:
        for f in sorted(glob.glob(pat)):
            m = re.search(rf"RESULT rho={F} seed=(\d+) worstgroup={F} advgroup={F} cueDelta={F} "
                          rf"\(G11={F} G10={F} G01={F} G00={F}\)", open(f).read())
            if m:
                G11, G10, G01, G00 = (float(m[i]) for i in (6, 7, 8, 9))
                rows.append(dict(cell=cell, rho=float(m[1]), seed=int(m[2]), worst=float(m[3]),
                                 cd=float(m[5]), overall=np.mean([G11, G10, G01, G00]),
                                 aligned=(G11 + G00) / 2, adversarial=(G10 + G01) / 2))
    d = pd.DataFrame(rows).drop_duplicates(subset=["cell", "rho", "seed"])
    d["zcd"] = d.groupby("cell")["cd"].transform(lambda s: (s - s.mean()) / s.std(ddof=1))
    d["stratum"] = d.cell + "_" + d.rho.astype(str)
    return d


def c1_capability(d):
    print("== C1 CAPABILITY ==")
    print("  The obvious controls are mechanically circular: overall accuracy CONTAINS")
    print("  worst-group (it is their mean), so conditioning worst on overall conditions")
    print("  the outcome on a quantity containing it. Under that control cueDelta dies:")
    for tag, f in (("worst ~ cueD + cell FE", "worst ~ zcd + C(cell)"),
                   ("  + overall accuracy   ", "worst ~ zcd + overall + C(cell)")):
        m = smf.ols(f, d).fit()
        print(f"    {tag}: beta={m.params['zcd']:+.4f} p={m.pvalues['zcd']:.3e}")
    print("\n  The CLEAN test uses DISJOINT example sets -- predict accuracy where the cue")
    print("  HURTS, controlling for accuracy where it HELPS. No shared examples:")
    for tag, f in (("adversarial ~ cueD + cell FE          ", "adversarial ~ zcd + C(cell)"),
                   ("adversarial ~ cueD + aligned + cell FE ", "adversarial ~ zcd + aligned + C(cell)"),
                   ("adversarial ~ cueD + aligned + strat FE", "adversarial ~ zcd + aligned + C(stratum)")):
        m = smf.ols(f, d).fit()
        print(f"    {tag}: beta={m.params['zcd']:+.4f} p={m.pvalues['zcd']:.3e}")
    full = smf.ols("adversarial ~ zcd + aligned + C(cell)", d).fit()
    base = smf.ols("adversarial ~ aligned + C(cell)", d).fit()
    print(f"    partial R2 of cueDelta over capability-only model = {full.rsquared - base.rsquared:+.4f}")
    print(f"    capability's own coefficient in that model: p={full.pvalues['aligned']:.3f}")


def c2_placebo():
    print("\n== C2 PLACEBO ==")
    v = []
    for f in sorted(glob.glob("logs/pbo_qwen_r*_s*_*.log")):
        t = open(f).read()
        m = re.search(rf"RESULT rho={F} seed=(\d+) worstgroup={F} advgroup={F} cueDelta={F}", t)
        p = re.search(rf"PLACEBO rho=[\d.]+ seed=\d+ placeboDelta={F} nEdited=(\d+)/(\d+)", t)
        if m and p:
            v.append((float(m[5]), float(p[1]), float(m[3]), int(p[2]) / int(p[3])))
    if not v:
        print("  (no placebo logs)"); return
    v = np.array(v)
    er = v[:, 3].mean()
    print(f"  N={len(v)}   |cueDelta| mean={np.abs(v[:,0]).mean():.3f}   "
          f"|placeboDelta| mean={np.abs(v[:,1]).mean():.3f}   ratio={np.abs(v[:,0]).mean()/np.abs(v[:,1]).mean():.1f}x")
    print(f"  placebo edits {er:.0%} of texts vs 100% for the identity edit; edit-rate-adjusted "
          f"ratio = {np.abs(v[:,0]).mean()/(np.abs(v[:,1]).mean()/er):.1f}x")
    print(f"  DECISIVE: r(cueDelta, worst)={pearsonr(v[:,0], v[:,2])[0]:+.3f} "
          f"(p={pearsonr(v[:,0], v[:,2])[1]:.1e})  vs  r(placeboDelta, worst)="
          f"{pearsonr(v[:,1], v[:,2])[0]:+.3f} (p={pearsonr(v[:,1], v[:,2])[1]:.2f})")
    print(f"  r(cueDelta, placeboDelta) = {pearsonr(v[:,0], v[:,1])[0]:+.3f}  (~0 => different constructs)")


def c3_reliability():
    print("\n== C3 RELIABILITY ==")
    per = {}
    for f in sorted(glob.glob("logs/civ3_*_r*_s*_*.log") + glob.glob("logs/bios3_*_r*_s*_*.log")):
        m = re.search(r"PEREX_CD ([\d\-. ]+)", open(f).read())
        if m:
            x = np.array([float(t) for t in m[1].split()])
            if len(x) >= 300:
                per[f] = x[:300]
    if len(per) < 10:
        print("  (insufficient per-example logs)"); return
    V = np.vstack(list(per.values()))
    rs = []
    for _ in range(500):
        idx = RNG.permutation(V.shape[1]); h = V.shape[1] // 2
        # per-model half estimates, correlated ACROSS models -- computing this WITHIN a
        # model gives exactly -1.000, because the halves partition a fixed total.
        rs.append(np.corrcoef(V[:, idx[:h]].mean(1), V[:, idx[h:]].mean(1))[0, 1])
    r = float(np.mean(rs)); sb = 2 * r / (1 + r)
    print(f"  {len(per)} models x 300 texts. split-half r across models = {r:.4f}, "
          f"Spearman-Brown = {sb:.4f}")
    print(f"  within-model SE={V.std(1, ddof=1).mean()/np.sqrt(V.shape[1]):.4f} vs "
          f"between-model SD={V.mean(1).std(ddof=1):.4f}")
    print(f"  => disattenuation moves a -0.0266 slope to {-0.0266/sb:+.4f}: measurement error")
    print("     is NOT what produced the small within-recipe effect.")


def c4_instrument():
    print("\n== C4 INSTRUMENT (rho is randomized) ==")
    import load_runs as P
    d = P.load()
    X = pd.get_dummies(d.cell, drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(d)), X])
    r = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    cd, w, z = r(d.cd.values), r(d.worst.values), r(d.rho.values)
    b_iv, b_ols = (z @ w) / (z @ cd), (cd @ w) / (cd @ cd)
    print(f"  first stage rho->cueDelta = {(z @ cd)/(z @ z):+.2f} (strong by construction)")
    print(f"  OLS={b_ols:+.4f}   2SLS={b_iv:+.4f}   ratio={b_iv/b_ols:.2f}")
    print("  CAVEAT: the exclusion restriction (rho affects worst-group ONLY through")
    print("  cueDelta) is almost certainly violated. Report as a bound, not an estimate --")
    print("  but do report it, since leaving a randomized instrument unused invites the ask.")


def c5_early_warning():
    print("\n== C5 EARLY WARNING -- does epoch-1 cueDelta beat free baselines? ==")
    rows = []
    for pat, cell in (("logs/ew_civ_qwen_r*_s*_*.log", "Qwen-civil"),
                      ("logs/ew_civ_pythia_r*_s*_*.log", "Pythia-civil")):
        for f in sorted(glob.glob(pat)):
            t = open(f).read()
            fin = re.search(rf"RESULT rho={F} seed=(\d+) worstgroup={F}", t)
            m1 = re.search(rf"EPOCH rho=[\d.]+ seed=\d+ epoch=1 cueDeltaTrain={F} cueDeltaEval={F} "
                           rf"worstSmall={F} overall={F} worstClass={F}", t)
            if fin and m1:
                rows.append(dict(cell=cell, rho=float(fin[1]), final=float(fin[3]),
                                 e1=float(m1[1]), ov=float(m1[4]), wc=float(m1[5])))
    if not rows:
        print("  (no early-warning logs)"); return
    d = pd.DataFrame(rows)
    print(f"  N={len(d)}. raw correlations with FINAL worst-group:")
    for k, lab in (("e1", "epoch-1 cueDelta"), ("ov", "epoch-1 overall accuracy (FREE)"),
                   ("wc", "epoch-1 worst-CLASS accuracy (FREE)")):
        rr, pp = pearsonr(d[k], d.final)
        print(f"    {lab:34s} r={rr:+.3f} p={pp:.1e}")
    print("  incremental validity of cueDelta over each free baseline:")
    for bl, lab in (("ov", "overall accuracy"), ("wc", "worst-CLASS accuracy")):
        for spec, f0, f1 in (("no controls          ", f"final ~ {bl}", f"final ~ e1 + {bl}"),
                             ("+ cell FE + rho DUMMIES", f"final ~ {bl} + C(cell) + C(rho)",
                              f"final ~ e1 + {bl} + C(cell) + C(rho)")):
            a, b = smf.ols(f0, d).fit(), smf.ols(f1, d).fit()
            print(f"    over {lab:22s} {spec}: dR2={b.rsquared-a.rsquared:+.4f} p={b.pvalues['e1']:.4f}")
    print("  => the incremental claim exists ONLY without proper controls. Early warning is")
    print("     DESCRIPTIVE ONLY: it does not beat a free baseline.")


def c6_out_of_regime():
    """R2: leave-one-rho-out extrapolation with cell FE -- train on 2 rho levels,
    predict the held-out level. The headline out-of-regime numbers live here;
    they previously existed only in a session transcript."""
    print("\n== C6 OUT-OF-REGIME EXTRAPOLATION (leave-one-rho-out, cell FE) ==")
    import load_runs as P
    d = P.load().reset_index(drop=True)
    rhos = sorted(d.rho.unique())
    for key in ("cd", "rho", "probe"):
        if d[key].isna().any():
            print(f"  {key}: NaNs, skipped"); continue
        yhat = np.full(len(d), np.nan)
        for hold in rhos:
            tr, te = d[d.rho != hold], d[d.rho == hold]
            m = smf.ols(f"worst ~ {key} + C(cell)", tr).fit()
            yhat[te.index] = m.predict(te)
        y = d.worst.values
        r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
        print(f"  from {key:6s}: R2={r2:+.3f}  MAE={np.mean(np.abs(y - yhat)):.3f}")
    yhat_s = np.full(len(d), np.nan)
    for hold in rhos:
        nearest = min([r for r in rhos if r != hold], key=lambda r: abs(r - hold))
        for cell in d.cell.unique():
            m = d[(d.rho == nearest) & (d.cell == cell)].worst.mean()
            yhat_s[d[(d.rho == hold) & (d.cell == cell)].index] = m
    y = d.worst.values
    print(f"  from rho-step (nearest observed level, cell-wise -- the monotone")
    print(f"  nonparametric alternative): R2={1 - np.sum((y - yhat_s) ** 2) / np.sum((y - y.mean()) ** 2):+.3f}"
          f"  MAE={np.mean(np.abs(y - yhat_s)):.3f}")
    print("  per-fold R2 for cueDelta vs the held-out level's own mean (the honest caveat):")
    for hold in rhos:
        tr, te = d[d.rho != hold], d[d.rho == hold]
        m = smf.ols("worst ~ cd + C(cell)", tr).fit()
        p = m.predict(te); yy = te.worst.values
        print(f"    rho={hold}: R2={1 - np.sum((yy-p)**2)/np.sum((yy-yy.mean())**2):+.3f}")




def c7_probe_ceiling():
    """Why 'used vs decodable' is uninformative at fixed rho: the probe barely
    varies within a stratum. Per-stratum SD of probe maxAUROC on the high-seed
    sample (the sample where fixed-rho comparisons are made)."""
    print("\n== C7 PROBE AT CEILING (per-stratum SD of probe AUROC, high-seed sample) ==")
    import load_runs as P
    d = P.load()
    sds = d.groupby(["cell", "rho"]).probe.std(ddof=1)
    for (cell, rho), s in sds.items():
        print(f"  {cell:14s} rho={rho}: SD(probeAUROC)={s:.4f}  (n={len(d[(d.cell==cell)&(d.rho==rho)])})")
    print(f"  range: {sds.min():.4f} -- {sds.max():.4f}   vs SD(cueDelta) within-stratum "
          f"{d.groupby(['cell','rho']).cd.std(ddof=1).min():.3f} -- {d.groupby(['cell','rho']).cd.std(ddof=1).max():.3f}")


if __name__ == "__main__":
    d = load_bios()
    print(f"bios cells loaded: N={len(d)}\n")
    c1_capability(d)
    c2_placebo()
    c3_reliability()
    c4_instrument()
    c5_early_warning()
    c6_out_of_regime()
    c7_probe_ceiling()
