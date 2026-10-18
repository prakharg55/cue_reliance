"""
Table 2. Does cue reliance predict the worst-group gain from LEACE?

Regresses each high-seed model's LEACE gain on cue reliance under four
specifications: cue reliance standardized within cell or within stratum, with
or without the model's worst-group accuracy before repair as a control. Every
specification includes cell-by-rho fixed effects (n = 135).
"""
import statsmodels.formula.api as smf

import load_runs

h = load_runs.load().dropna(subset=["le_dw"]).reset_index(drop=True)
assert len(h) == 135, len(h)
h["stratum"] = h.cell + "_" + h.rho.astype(str)
h["z_cell"] = h.groupby("cell").cd.transform(lambda s: (s - s.mean()) / s.std(ddof=1))
h["z_stratum"] = h.groupby("stratum").cd.transform(lambda s: (s - s.mean()) / s.std(ddof=1))

print("== Table 2: LEACE worst-group gain ~ cue reliance + cell x rho fixed effects (n=135) ==")
print(f"  {'cue reliance standardized':27s} {'baseline control':17s} {'beta':>8s} {'p':>8s}")
for z, name in (("z_cell", "within cell"), ("z_stratum", "within stratum")):
    for control in (False, True):
        m = smf.ols(f"le_dw ~ {z} + C(stratum)" + (" + worst" if control else ""), h).fit()
        print(f"  {name:27s} {'yes' if control else 'no':17s} "
              f"{m.params[z]:+8.4f} {m.pvalues[z]:8.4f}")
