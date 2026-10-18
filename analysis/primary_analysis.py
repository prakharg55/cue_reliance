"""
The pre-specified within-recipe analysis (Section 7, Tables 5 and 6).

Estimates how cue reliance relates to worst-group accuracy among models trained
with the same recipe, holding rho fixed exactly. The specification below was
fixed before the confirmation half of the high-seed data was analyzed.

  Estimand   the average association between cue reliance and worst-group
             accuracy within a recipe, with rho held fixed exactly.
  Primary    the equal-weighted mean of the nine within-stratum OLS slopes, in
             raw accuracy points per unit of cue reliance, tested with t(8).
             Equal weights are chosen in advance because pooled OLS would put
             76% of the weight on the rho = 0.97 strata, where cue reliance
             varies most.
  Companion  the pooled within-stratum correlation (Fisher z).
  Checks     heterogeneity (Q, I^2), leave-one-cell-out, leave-one-stratum-out,
             a within-stratum permutation test, and the high-seed-only
             replication.
"""
import glob, re, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, t as tdist, chi2, norm

warnings.filterwarnings("ignore")
import load_runs as P

RNG = np.random.default_rng(0)


def stratum_slopes(d):
    """One OLS slope and correlation per (cell x rho) stratum, raw units."""
    out = []
    for s, g in d.groupby("stratum"):
        if len(g) < 5 or g.cd.std(ddof=1) == 0:
            continue
        b, _ = np.polyfit(g.cd, g.worst, 1)
        r = pearsonr(g.cd, g.worst)[0]
        se = np.sqrt(np.sum((g.worst - np.poly1d(np.polyfit(g.cd, g.worst, 1))(g.cd)) ** 2)
                     / (len(g) - 2) / np.sum((g.cd - g.cd.mean()) ** 2))
        out.append(dict(stratum=s, cell=g.cell.iloc[0], rho=g.rho.iloc[0], n=len(g),
                        b=b, se=se, r=r, sd_cd=g.cd.std(ddof=1), sd_w=g.worst.std(ddof=1)))
    return pd.DataFrame(out)


def primary(sl, label="PRIMARY"):
    b = sl.b.values
    t = b.mean() / (b.std(ddof=1) / np.sqrt(len(b)))
    p = 2 * tdist.sf(abs(t), len(b) - 1)
    ci = tdist.ppf(.975, len(b) - 1) * b.std(ddof=1) / np.sqrt(len(b))
    print(f"  {label}: equal-weighted mean within-stratum slope = {b.mean():+.4f} "
          f"accuracy points per cueDelta unit")
    print(f"           t({len(b)-1})={t:+.2f}  p={p:.4f}  95% CI [{b.mean()-ci:+.4f}, {b.mean()+ci:+.4f}]")
    return b.mean(), p


def fisher(sl):
    z = np.arctanh(np.clip(sl.r.values, -.999, .999)); w = sl.n.values - 3
    zb = (z * w).sum() / w.sum(); se = 1 / np.sqrt(w.sum())
    Q = float((w * (z - zb) ** 2).sum()); df = len(z) - 1
    return (np.tanh(zb), np.tanh(zb - 1.96 * se), np.tanh(zb + 1.96 * se),
            2 * norm.sf(abs(zb / se)), Q, chi2.sf(Q, df), max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0)


def main():
    d = P.load()
    sl = stratum_slopes(d)
    print(f"=== data: {len(d)} models, {len(sl)} strata, {sl.n.min()}-{sl.n.max()} seeds each ===\n")

    print("== PRIMARY ESTIMATE ==")
    primary(sl)
    r, lo, hi, pf, Q, pQ, I2 = fisher(sl)
    print(f"  companion: Fisher-z pooled within-stratum r = {r:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  p={pf:.4f}")
    print(f"  heterogeneity: Q({len(sl)-1})={Q:.2f} p={pQ:.3f} I2={I2:.0f}%"
          f"  -> {'homogeneous; pooling licensed' if pQ > .1 else 'HETEROGENEOUS'}")

    print("\n== why equal weighting (the implicit FE-OLS weights) ==")
    w = (sl.n - 1) * sl.sd_cd ** 2
    for rho_, g in sl.assign(w=w).groupby("rho"):
        print(f"  rho={rho_}: OLS weight share {100*g.w.sum()/w.sum():5.1f}%   "
              f"mean SD(cueDelta)={g.sd_cd.mean():.3f}   mean r={g.r.mean():+.3f}")

    print("\n== standardization audit (all four, so the choice is visible) ==")
    d2 = d.copy()
    d2["zcd_cell"] = d2.groupby("cell")["cd"].transform(lambda s: (s - s.mean()) / s.std(ddof=1))
    d2["zcd_str"] = d2.groupby("stratum")["cd"].transform(lambda s: (s - s.mean()) / s.std(ddof=1))
    d2["zw_str"] = d2.groupby("stratum")["worst"].transform(lambda s: (s - s.mean()) / s.std(ddof=1))
    for tag, f, k in (("z within CELL + stratum FE  (MISMATCHED - do not use)", "worst ~ zcd_cell + C(stratum)", "zcd_cell"),
                      ("z within STRATUM + stratum FE", "worst ~ zcd_str + C(stratum)", "zcd_str"),
                      ("raw cueDelta + stratum FE", "worst ~ cd + C(stratum)", "cd"),
                      ("both z within stratum", "zw_str ~ zcd_str + C(stratum)", "zcd_str")):
        m = smf.ols(f, d2).fit()
        print(f"  {tag:46s} beta={m.params[k]:+.4f} p={m.pvalues[k]:.4f}")

    print("\n== robustness ==")
    for drop in sorted(sl.cell.unique()):
        s2 = sl[sl.cell != drop]
        b2, p2 = s2.b.mean(), 2 * tdist.sf(abs(s2.b.mean() / (s2.b.std(ddof=1) / np.sqrt(len(s2)))), len(s2) - 1)
        print(f"  leave-out cell {drop:18s}: slope={b2:+.4f} p={p2:.4f}  (Fisher r={fisher(s2)[0]:+.3f})")
    for i in range(len(sl)):
        s2 = sl.drop(sl.index[i])
        p2 = 2 * tdist.sf(abs(s2.b.mean() / (s2.b.std(ddof=1) / np.sqrt(len(s2)))), len(s2) - 1)
        if p2 > .05:
            print(f"  leave-out stratum {sl.stratum.iloc[i]:26s}: p={p2:.4f}  <- fragile to this one")

    # within-stratum permutation test: shuffle cueDelta inside each stratum
    obs = sl.b.mean(); cnt = 0; B = 20000
    arrs = [(g.cd.values, g.worst.values) for _, g in d.groupby("stratum")]
    for _ in range(B):
        bs = [np.polyfit(RNG.permutation(cd), w, 1)[0] for cd, w in arrs]
        cnt += abs(np.mean(bs)) >= abs(obs)
    print(f"  within-stratum permutation test (B={B}): p={cnt/B:.4f}  (assumption-free)")

    hs = d[d.seed >= 3]
    slh = stratum_slopes(hs)
    ph = 2 * tdist.sf(abs(slh.b.mean() / (slh.b.std(ddof=1) / np.sqrt(len(slh)))), len(slh) - 1)
    print(f"  high-seed-only replication (seeds 3-14, n={len(hs)}): slope={slh.b.mean():+.4f} "
          f"p={ph:.4f}  Fisher r={fisher(slh)[0]:+.3f}")

    # sealed confirmation half: fit nothing on it until now
    a, b_ = d[(d.seed >= 3) & (d.seed <= 8)], d[d.seed >= 9]
    for nm, part in (("exploratory half (seeds 3-8) ", a), ("SEALED half   (seeds 9-14)", b_)):
        s2 = stratum_slopes(part)
        p2 = 2 * tdist.sf(abs(s2.b.mean() / (s2.b.std(ddof=1) / np.sqrt(len(s2)))), len(s2) - 1)
        print(f"  {nm}: slope={s2.b.mean():+.4f} p={p2:.4f} Fisher r={fisher(s2)[0]:+.3f}")

    print("\n== per-stratum detail (report this table; it makes the mechanism visible) ==")
    print(sl[["stratum", "n", "sd_cd", "sd_w", "b", "r"]].to_string(index=False,
          float_format=lambda v: f"{v:+.4f}"))


if __name__ == "__main__":
    main()
