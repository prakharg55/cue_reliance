"""
Table 7. Equivalence tests (TOST) for the within-recipe effect over a range of
margins, following Braun et al. (2022). Uses the most conservative of the four
specifications in Table 6.
"""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
from scipy.stats import t as tdist
warnings.filterwarnings("ignore")
import load_runs as P

if __name__ == "__main__":
    d = P.load()
    d["zcd_cell"] = d.groupby("cell")["cd"].transform(lambda s: (s - s.mean()) / s.std(ddof=1))
    m = smf.ols("worst ~ zcd_cell + C(stratum)", d).fit()
    b, se, df = m.params["zcd_cell"], m.bse["zcd_cell"], int(m.df_resid)

    tot_sd = d.worst.std(ddof=1)
    within_sd = float(np.sqrt(np.mean(d.groupby("stratum")["worst"].var(ddof=1))))
    # the anchor a reader can actually interpret: the effect of the recipe itself
    hi = d[d.rho == 0.97].worst.mean(); lo = d[d.rho == 0.65].worst.mean()
    rho_effect = hi - lo

    print(f"within-rho slope: beta={b:+.4f} accuracy points per within-cell SD of cueDelta")
    print(f"  se={se:.4f}  df={df}  two-sided p={m.pvalues['zcd_cell']:.4f}")
    lo90, hi90 = b - tdist.ppf(.95, df) * se, b + tdist.ppf(.95, df) * se
    print(f"  90% CI (the interval TOST reads) = [{lo90:+.4f}, {hi90:+.4f}]")
    print(f"  -> we can reject |beta| >= {max(abs(lo90), abs(hi90)):.4f} accuracy points")
    print(f"\nscale anchors:  total SD(worst)={tot_sd:.4f}   within-stratum SD={within_sd:.4f}"
          f"   rho=0.97 vs 0.65 effect={rho_effect:+.4f}")

    print("\nmargin sensitivity (Braun et al. report this rather than one margin):")
    print(f"  {'margin':>8s}  {'as % of rho effect':>19s}  {'TOST p':>8s}  verdict")
    for margin in (0.02, 0.03, 0.04, 0.05, 0.0542, 0.06, 0.075, 0.10):
        p_lo = tdist.sf((b - (-margin)) / se, df)      # H0: beta <= -margin
        p_hi = tdist.cdf((b - margin) / se, df)        # H0: beta >= +margin
        p = max(p_lo, p_hi)
        print(f"  {margin:8.4f}  {100*margin/abs(rho_effect):18.1f}%  {p:8.4f}  "
              f"{'EQUIVALENT' if p < .05 else 'cannot conclude equivalence'}")

    print("\nprincipled margin candidates:")
    print(f"  1/10 within-stratum SD (Kruschke rule)      = {within_sd/10:.4f}")
    print(f"  10% of the recipe effect (interpretable)    = {abs(rho_effect)*0.10:.4f}")
    print(f"  the smallest repair effect we care about    = 0.0300  (LEACE mean gain ~0.09)")
