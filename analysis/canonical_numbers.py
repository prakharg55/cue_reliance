"""
Per-cell statistics for the Qwen and Pythia populations, used for numbers
quoted in the text. Includes the correlations of cue reliance and rho with
worst-group accuracy, partial correlations with bootstrap intervals, the probe
comparison, and mean repair gains.
"""
import glob, re, sys
import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata, norm, chi2

LOGDIR = "logs"
RNG = np.random.default_rng(0)
B = 20_000

CELLS = [
    ("Qwen-civil",     f"{LOGDIR}/civ2_qwen_r*_s*_11489602.log"),
    ("Pythia-civil",   f"{LOGDIR}/civ2_pythia_r*_s*_11494325.log"),
    ("Qwen-bios",      f"{LOGDIR}/biosv2_qwen_r*_s*_11498479.log"),
    ("Pythia-bios",    f"{LOGDIR}/biosv2_pythia_r*_s*_11502814.log"),
]
GRID = [name for name, _ in CELLS]          # primary Fisher combination: the 2x2 only
SCALE_CELLS = [                             # third axis; each included when 18/18 logs exist
    ("Pythia1.4b-civil", f"{LOGDIR}/civ2_pythia14_r*_s*_11520739.log"),
    ("Pythia1.4b-bios",  f"{LOGDIR}/biosv2_pythia14_r*_s*_*.log"),
    ("Qwen1.5B-civil",   f"{LOGDIR}/civ2_qwen15_r*_s*_*.log"),
    ("Qwen1.5B-bios",    f"{LOGDIR}/biosv2_qwen15_r*_s*_*.log"),
]


def parse_cell(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        t = open(f).read()
        m = re.search(r"RESULT rho=([\d.]+) seed=(\d+) worstgroup=([\d.]+) "
                      r"advgroup=([\d.]+) cueDelta=([-\d.]+)", t)
        if not m:
            continue
        row = dict(rho=float(m[1]), seed=int(m[2]), worst=float(m[3]),
                   adv=float(m[4]), cd=float(m[5]))
        pm = re.search(r"PROBE_RESULT rho=[\d.]+ seed=\d+ maxAUROC=([\d.]+)", t)
        row["probe"] = float(pm[1]) if pm else np.nan
        lm = re.search(r"LEACE rho=[\d.]+ seed=\d+ L=\d+ trainAUROC_after=[\d.]+ "
                       r"worst=[\d.]+ overall=[\d.]+ delta_worst=([-+\d.]+)", t)
        row["leace_dw"] = float(lm[1]) if lm else np.nan
        sm = re.search(r"PROSPECT_RESULT rho=[\d.]+ seed=\d+ Lstar=\d+ worst=[\d.]+ "
                       r"overall=[\d.]+ delta_worst=([-+\d.]+)", t)
        row["prospect_dw"] = float(sm[1]) if sm else np.nan
        fm = re.search(r"FIXED(?:20)? rho=[\d.]+ seed=\d+ (?:L=\d+ )?worst=[\d.]+ "
                       r"overall=[\d.]+ delta_worst=([-+\d.]+)", t)
        row["fixed_dw"] = float(fm[1]) if fm else np.nan
        rows.append(row)
    return rows


def partial_p(x, y, z):
    rx = x - np.poly1d(np.polyfit(z, x, 1))(z)
    ry = y - np.poly1d(np.polyfit(z, y, 1))(z)
    return pearsonr(rx, ry)


def partial_stat(x, y, z):
    return partial_p(x, y, z)[0]


def bca_partial(x, y, z):
    n = len(x); th = partial_stat(x, y, z)
    bs = np.array([partial_stat(x[i], y[i], z[i]) for i in RNG.integers(0, n, (B, n))])
    bs = bs[np.isfinite(bs)]
    z0 = norm.ppf(min(max(np.mean(bs < th), 1e-9), 1 - 1e-9))
    jk = np.array([partial_stat(np.delete(x, i), np.delete(y, i), np.delete(z, i))
                   for i in range(n)])
    jm = jk.mean(); a = np.sum((jm - jk) ** 3) / (6 * (np.sum((jm - jk) ** 2) ** 1.5) + 1e-12)
    def q(al):
        zz = norm.ppf(al)
        return float(np.percentile(bs, 100 * norm.cdf(z0 + (z0 + zz) / (1 - a * (z0 + zz)))))
    return th, q(0.025), q(0.975)


def bca_r(x, y):
    n = len(x); th = pearsonr(x, y)[0]
    bs = np.array([pearsonr(x[i], y[i])[0] for i in RNG.integers(0, n, (B, n))])
    bs = bs[np.isfinite(bs)]
    z0 = norm.ppf(min(max(np.mean(bs < th), 1e-9), 1 - 1e-9))
    jk = np.array([pearsonr(np.delete(x, i), np.delete(y, i))[0] for i in range(n)])
    jm = jk.mean(); a = np.sum((jm - jk) ** 3) / (6 * (np.sum((jm - jk) ** 2) ** 1.5) + 1e-12)
    def q(al):
        zz = norm.ppf(al)
        return float(np.percentile(bs, 100 * norm.cdf(z0 + (z0 + zz) / (1 - a * (z0 + zz)))))
    return th, q(0.025), q(0.975)


def fisher(ps):
    ps = [p for p in ps if np.isfinite(p)]
    stat = -2 * sum(np.log(max(p, 1e-300)) for p in ps)
    return stat, chi2.sf(stat, 2 * len(ps))


def cell_battery(name, rows):
    w = np.array([r["worst"] for r in rows]); cd = np.array([r["cd"] for r in rows])
    rho = np.array([r["rho"] for r in rows]); pr = np.array([r["probe"] for r in rows])
    out = {"N": len(rows)}
    print(f"\n== {name} (N={len(rows)}) ==")
    th, lo, hi = bca_r(cd, w); _, p = pearsonr(cd, w)
    out["raw"] = (th, p, lo, hi)
    print(f"  raw r(cueD, worst)        = {th:+.4f}  p={p:.2e}  BCa[{lo:+.3f},{hi:+.3f}]")
    r2, p2 = pearsonr(rho, w)
    out["raw_rho"] = (r2, p2)
    print(f"  raw r(rho, worst)         = {r2:+.4f}  p={p2:.2e}")
    th, lo, hi = bca_partial(cd, w, rho); _, p3 = partial_p(cd, w, rho)
    out["fwd"] = (th, p3, lo, hi)
    print(f"  partial(cueD,worst|rho)   = {th:+.4f}  p={p3:.4f}  BCa[{lo:+.3f},{hi:+.3f}]")
    t2, lo2, hi2 = bca_partial(rho, w, cd); _, p4 = partial_p(rho, w, cd)
    out["rev"] = (t2, p4, lo2, hi2)
    print(f"  partial(rho,worst|cueD)   = {t2:+.4f}  p={p4:.4f}  BCa[{lo2:+.3f},{hi2:+.3f}]"
          f"  zero inside: {lo2 < 0 < hi2}")
    if np.isfinite(pr).all():
        r5, p5 = pearsonr(pr, w)
        t6 = partial_stat(cd, w, pr); _, p6 = partial_p(cd, w, pr)
        t7 = partial_stat(pr, w, cd); _, p7 = partial_p(pr, w, cd)
        out["probe_raw"], out["used"], out["decod"] = (r5, p5), (t6, p6), (t7, p7)
        print(f"  raw r(probeAUROC, worst)  = {r5:+.4f}  p={p5:.4f}")
        print(f"  partial(cueD,worst|probe) = {t6:+.4f}  p={p6:.4f}")
        print(f"  partial(probe,worst|cueD) = {t7:+.4f}  p={p7:.4f}")
    rs, ps = spearmanr(cd, w)
    tsp, psp = partial_p(rankdata(cd).astype(float), rankdata(w).astype(float),
                         rankdata(rho).astype(float))
    print(f"  Spearman raw / rank-part. = {rs:+.4f} (p={ps:.1e}) / {tsp:+.4f} (p={psp:.4f})")
    adv = np.array([r["adv"] for r in rows])
    r8, p8 = pearsonr(cd, adv)
    print(f"  secondary r(cueD, advgrp) = {r8:+.4f}  p={p8:.2e}")
    for k, lab in [("leace_dw", "LEACE"), ("prospect_dw", "prospect L*"),
                   ("fixed_dw", "fixed-depth")]:
        v = np.array([r[k] for r in rows])
        if np.isfinite(v).all():
            print(f"  mean delta_worst {lab:12s}= {v.mean():+.4f}  (sd {v.std(ddof=1):.3f})")
    return out


def main():
    results = {}
    for name, pattern in CELLS:
        rows = parse_cell(pattern)
        if len(rows) != 18:
            print(f"FATAL: {name} has N={len(rows)} != 18"); sys.exit(1)
        results[name] = cell_battery(name, rows)

    scale_done = []
    for name, pattern in SCALE_CELLS:
        rows = parse_cell(pattern)
        if len(rows) == 18:
            results[name] = cell_battery(name, rows)
            scale_done.append(name)
        else:
            print(f"\n[scale cell {name}: {len(rows)}/18 logs -- skipped]")

    for grid, lab_grid in [(GRID, "2x2 grid (primary)"),
                           (GRID + scale_done, f"all {len(GRID)+len(scale_done)} cells")]:
        print(f"\n== Fisher-combined across the {lab_grid} ==")
        for key, lab in [("fwd", "forward partial(cueD|rho)"),
                         ("rev", "reverse partial(rho|cueD)"),
                         ("used", "partial(cueD|probe)"),
                         ("decod", "partial(probe|cueD)")]:
            ps = [results[c][key][1] for c in grid if key in results[c]]
            stat, p = fisher(ps)
            print(f"  {lab:28s} chi2={stat:6.2f} df={2*len(ps)}  p={p:.2e}"
                  f"   cell p's: {', '.join(f'{q:.3f}' for q in ps)}")


if __name__ == "__main__":
    main()
