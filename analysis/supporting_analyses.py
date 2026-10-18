"""
Supporting analyses over the Qwen and Pythia populations. Covers how much rho
and cue reliance overlap in each cell, whether the probe is near its ceiling,
leave-one-rho-out prediction, repair targeting, and a per-cell comparison of
the repairs.
"""
import glob, re, sys
import numpy as np
from scipy.stats import pearsonr

CORE = [
    ("Qwen-civil",   "logs/civ2_qwen_r*_s*_11489602.log"),
    ("Pythia-civil", "logs/civ2_pythia_r*_s*_11494325.log"),
    ("Qwen-bios",    "logs/biosv2_qwen_r*_s*_11498479.log"),
    ("Pythia-bios",  "logs/biosv2_pythia_r*_s*_11502814.log"),
]
SCALE = [
    ("Pythia1.4b-civil", "logs/civ2_pythia14_r*_s*_11520739.log"),
    ("Pythia1.4b-bios",  "logs/biosv2_pythia14_r*_s*_*.log"),
    ("Qwen1.5B-civil",   "logs/civ2_qwen15_r*_s*_*.log"),
    ("Qwen1.5B-bios",    "logs/biosv2_qwen15_r*_s*_*.log"),
]
CELLS = CORE + SCALE
F = r"([-+\d.]+)"

def parse(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        t = open(f).read()
        m = re.search(rf"RESULT rho={F} seed=(\d+) worstgroup={F} advgroup={F} cueDelta={F}", t)
        pm = re.search(rf"PROBE_RESULT rho=[\d.]+ seed=\d+ maxAUROC={F} argmaxL=(\d+)", t)
        lm = re.search(rf"LEACE rho=[\d.]+ seed=\d+ L=\d+ trainAUROC_after={F} worst={F} overall={F} delta_worst={F}", t)
        sm = re.search(rf"PROSPECT_RESULT rho=[\d.]+ seed=\d+ Lstar=(\d+) worst={F} overall={F} delta_worst={F}", t)
        fm = re.search(rf"FIXED(?:20)? rho=[\d.]+ seed=\d+ (?:L=\d+ )?worst={F} overall={F} delta_worst={F}", t)
        if not (m and pm and lm and sm and fm):
            print(f"SKIP {f}: missing lines"); continue
        cm = re.search(rf"PROSPECT train-cueDelta base={F} curve (.+)", t)
        base = float(cm[1]); curve = [float(v) for v in cm[2].split()]
        Lstar = int(sm[1])
        fsel = 1 - curve[Lstar] / base if base > 0.2 else np.nan
        rows.append(dict(
            rho=float(m[1]), seed=int(m[2]), worst=float(m[3]), cd=float(m[5]),
            probe=float(pm[1]), probeL=int(pm[2]),
            le_dw=float(lm[4]), le_ov=float(lm[3]),
            pl_L=Lstar, pl_dw=float(sm[4]), pl_ov=float(sm[3]),
            fx_dw=float(fm[3]), fx_ov=float(fm[2]), fsel=fsel))
    return rows

def col(rows, k): return np.array([r[k] for r in rows])

def loo_rho_r2(rows, key):
    """Pooled out-of-sample R2/MAE predicting worst from `key`, leaving one rho out."""
    rhos = sorted({r["rho"] for r in rows})
    y = col(rows, "worst"); yhat = np.full(len(rows), np.nan)
    for hold in rhos:
        tr = [i for i, r in enumerate(rows) if r["rho"] != hold]
        te = [i for i, r in enumerate(rows) if r["rho"] == hold]
        x = col(rows, key)
        b = np.polyfit(x[tr], y[tr], 1)
        yhat[te] = np.poly1d(b)(x[te])
    ss_res = np.sum((y - yhat) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot, float(np.mean(np.abs(y - yhat)))

if __name__ == "__main__":
    data = {}
    for name, pat in CELLS:
        rows = parse(pat)
        if len(rows) == 18:
            data[name] = rows
        else:
            print(f"[{name}: {len(rows)}/18 logs -- skipped]")
            if name in dict(CORE): sys.exit(1)
    NTOT = 18 * len(data)

    print("== 1. cueDelta-rho collinearity ==")
    for name, rows in data.items():
        r, p = pearsonr(col(rows, "cd"), col(rows, "rho"))
        print(f"  {name:14s} r(cueD, rho) = {r:+.3f}  p={p:.1e}")

    print("\n== 2. probe decodability range ==")
    for name, rows in data.items():
        pr = col(rows, "probe")
        print(f"  {name:14s} maxAUROC mean={pr.mean():.3f} sd={pr.std(ddof=1):.3f} "
              f"range[{pr.min():.3f},{pr.max():.3f}]   cueD sd/mean={col(rows,'cd').std(ddof=1):.2f}/{col(rows,'cd').mean():.2f}")

    print("\n== 3. leave-one-rho-out prediction of worst-group ==")
    print(f"  {'cell':14s} {'from cueD R2/MAE':>20s} {'from rho R2/MAE':>20s} {'from probe R2/MAE':>20s}")
    for name, rows in data.items():
        s = f"  {name:14s}"
        for key in ("cd", "rho", "probe"):
            r2, mae = loo_rho_r2(rows, key)
            s += f"      {r2:+.3f}/{mae:.3f}  "
        print(s)

    print("\n== 4. repair targeting: r(x, LEACE delta_worst) ==")
    for name, rows in data.items():
        r1, p1 = pearsonr(col(rows, "cd"), col(rows, "le_dw"))
        r2, p2 = pearsonr(col(rows, "rho"), col(rows, "le_dw"))
        print(f"  {name:14s} cueD: {r1:+.3f} (p={p1:.3f})   rho: {r2:+.3f} (p={p2:.3f})")

    print("\n== 5. interventions per cell (mean delta_worst | mean post-overall) ==")
    print(f"  {'cell':14s} {'FIXED20':>18s} {'prospect-L*':>18s} {'LEACE':>18s}")
    for name, rows in data.items():
        parts = []
        for dk, ok in (("fx_dw", "fx_ov"), ("pl_dw", "pl_ov"), ("le_dw", "le_ov")):
            parts.append(f"{col(rows,dk).mean():+.3f} | {col(rows,ok).mean():.3f}")
        print(f"  {name:14s} {parts[0]:>18s} {parts[1]:>18s} {parts[2]:>18s}")
        ls = col(rows, "pl_L")
        print(f"  {'':14s} prospect L* chosen: min={ls.min()} med={int(np.median(ls))} max={ls.max()}"
              f"   probe argmaxL med={int(np.median(col(rows,'probeL')))}")

    print(f"\n== 6. pooled within-cell-standardized regression (N={NTOT}, {len(data)} cells) ==")
    Z = lambda v: (v - v.mean()) / v.std(ddof=1)
    zw, zc, zr = [], [], []
    for rows in data.values():
        zw.append(Z(col(rows, "worst"))); zc.append(Z(col(rows, "cd"))); zr.append(Z(col(rows, "rho")))
    zw, zc, zr = map(np.concatenate, (zw, zc, zr))
    r, p = pearsonr(zc, zw)
    print(f"  pooled r(z_cueD, z_worst) = {r:+.4f}  p={p:.2e}")
    X = np.column_stack([np.ones_like(zc), zc, zr])
    beta, *_ = np.linalg.lstsq(X, zw, rcond=None)
    res = zw - X @ beta; dof = len(zw) - 3
    se = np.sqrt(np.sum(res**2)/dof * np.linalg.inv(X.T @ X).diagonal())
    from scipy.stats import t as tdist
    for i, lab in ((1, "beta cueD | rho"), (2, "beta rho | cueD")):
        tv = beta[i]/se[i]
        print(f"  {lab}: {beta[i]:+.4f} (t={tv:+.2f}, p={2*tdist.sf(abs(tv), dof):.2e})")

    # 3 seeds share each rho level, so treat (cell x rho) as the sampling cluster:
    # resample the 6 rho-levels within every cell with replacement and refit the
    # pooled standardized OLS. CIs excluding zero = the OLS p's are not an artifact
    # of within-rho dependence.
    print(f"\n== 7. cluster bootstrap over rho-levels (B=4000, {len(data)} cells) ==")
    Zs = lambda v: (v - v.mean()) / v.std(ddof=1) if v.std(ddof=1) > 0 else v - v.mean()
    def pooled_beta(cells_rows, ykey):
        ys, xc, xr = [], [], []
        for rows in cells_rows:
            ys.append(Zs(col(rows, ykey))); xc.append(Zs(col(rows, "cd"))); xr.append(Zs(col(rows, "rho")))
        y = np.concatenate(ys)
        X = np.column_stack([np.ones_like(y), np.concatenate(xc), np.concatenate(xr)])
        return np.linalg.lstsq(X, y, rcond=None)[0][1:]
    rng = np.random.default_rng(1)
    RHOS = sorted({r["rho"] for r in next(iter(data.values()))})
    for ykey, lab in (("worst", "worst ~ cueD + rho"), ("le_dw", "LEACE-dw ~ cueD + rho")):
        point = pooled_beta(list(data.values()), ykey)
        bs = []
        for _ in range(4000):
            resampled = []
            for rows in data.values():
                pick = rng.choice(RHOS, size=len(RHOS), replace=True)
                resampled.append([r for p in pick for r in rows if r["rho"] == p])
            bs.append(pooled_beta(resampled, ykey))
        bs = np.array(bs)
        for j, nm in enumerate(("cueD", "rho ")):
            lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
            print(f"  {lab:24s} beta {nm}: {point[j]:+.4f}  cluster-CI[{lo:+.3f},{hi:+.3f}]"
                  f"  excludes 0: {lo > 0 or hi < 0}")

    # The prospective layer choice minimizes TRAIN cue-reliance; show that this
    # selection signal saturates in every cell (large achievable train-cueD drop)
    # while held-out damage diverges -- i.e. it senses cue suppression, not
    # collateral task damage. Limitations-section evidence.
    print("\n== 8. prospect selection signal vs held-out damage ==")
    for name, rows in data.items():
        f = np.array([r["fsel"] for r in rows]); dw = col(rows, "pl_dw")
        ok = np.isfinite(f)
        if ok.sum() > 2:
            r_, p_ = pearsonr(f[ok], dw[ok])
            print(f"  {name:16s} mean frac train-cueD removed at L* = {np.nanmean(f):.2f} "
                  f"(n={ok.sum()})   r(frac, delta_worst) = {r_:+.3f} (p={p_:.3f})")
