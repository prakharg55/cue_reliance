"""
Table 9 and Section 6. Chooses which models to repair under a budget. Within
each high-seed cell of 45 models, ranks models by cue reliance, by headroom
(one minus worst-group accuracy before repair), by assigned rho, at random,
and by the realized gain as an oracle. Reports the mean LEACE gain of the
models each policy picks.
"""
import numpy as np
from scipy.stats import pearsonr
import load_runs as P

if __name__ == "__main__":
    d = P.load().dropna(subset=["le_dw"]).reset_index(drop=True)
    assert len(d) == 135
    d["z"] = d.groupby("cell").cd.transform(lambda s: (s - s.mean()) / s.std(ddof=1))
    d["headroom"] = 1.0 - d.worst
    r_h, p_h = pearsonr(d.headroom, d.le_dw)
    print(f"headroom check: r(1-worst, LEACE gain) = {r_h:+.3f} (p={p_h:.1e}) on n={len(d)}")

    K = 15
    print(f"\n== prospective allocation: repair top {K} of 45 per held-out cell ==")
    print(f"{'held-out cell':16s} {'random':>8s} {'cueD':>8s} {'headroom':>9s} {'high-rho':>9s} {'oracle':>8s}")
    sums = np.zeros(5)
    for cell in sorted(d.cell.unique()):
        te = d[d.cell == cell]
        pol = {
            "random": te.le_dw.mean(),
            "cueD": te.nlargest(K, "z").le_dw.mean(),
            "headroom": te.nlargest(K, "headroom").le_dw.mean(),
            "high-rho": te[te.rho == te.rho.max()].le_dw.mean(),
            "oracle": te.nlargest(K, "le_dw").le_dw.mean(),
        }
        sums += np.array(list(pol.values()))
        print(f"{cell:16s} " + " ".join(f"{v:+8.3f}" for v in pol.values()))
    avg = sums / d.cell.nunique()
    print(f"{'MEAN':16s} " + " ".join(f"{v:+8.3f}" for v in avg))
    print(f"\n== budget sweep (mean selected gain over cells; rho policy uses")
    print(f"   stratum-mean expectations since ranking is undefined within a stratum) ==")
    print(f"{'k':>4s} {'random':>8s} {'cueD':>8s} {'headroom':>9s} {'high-rho':>9s} {'oracle':>8s}")
    for k in (5, 10, 15, 20, 30):
        acc = np.zeros(5)
        for cell in sorted(d.cell.unique()):
            te = d[d.cell == cell]
            strata = te.groupby("rho").le_dw.mean().sort_index(ascending=False)
            left, exp_rho = k, 0.0
            for mval in strata.values:
                take = min(left, 15); exp_rho += take * mval; left -= take
                if left == 0: break
            acc += np.array([te.le_dw.mean(),
                             te.nlargest(k, "z").le_dw.mean(),
                             te.nlargest(k, "headroom").le_dw.mean(),
                             exp_rho / k,
                             te.nlargest(k, "le_dw").le_dw.mean()])
        acc /= d.cell.nunique()
        print(f"{k:>4d} " + " ".join(f"{v:+8.3f}" for v in acc))

    print("\n== uncertainty: can random selection match the cueD ranking? ==")
    rng = np.random.default_rng(7)
    for k in (5, 10):
        exc = []
        for cell in sorted(d.cell.unique()):
            te = d[d.cell == cell]
            sel = te.nlargest(k, "z").le_dw.mean()
            draws = np.array([rng.choice(te.le_dw.values, size=k, replace=False).mean()
                              for _ in range(10000)])
            exc.append((draws >= sel).mean())
        print(f"  k={k}: P(random k-subset >= cueD selection) per cell = "
              + ", ".join(f"{e:.4f}" for e in exc))

    print("\nReading: ranking by the measurement is compared against selection by")
    print("visible failure (headroom = 1 - baseline worst) and by the assigned")
    print("recipe; oracle = selecting on realized gain, the ceiling.")
