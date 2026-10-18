"""
Tables 1, 8 and 10 and Section 8. Mean worst-group gain and overall accuracy
for the five repairs, whether mediation predicts when hard ablation is safe,
how well PMI-discovered cues agree with the identity lexicon, and label
efficiency.
"""
import glob, re, sys
import numpy as np
from scipy.stats import pearsonr

F = r"([-+\d.]+)"
CELLS = [
    ("Qwen-civil-v3",   "logs/civ3_qwen_r*_s*_*.log",   "civil"),
    ("Pythia-civil-v3", "logs/civ3_pythia_r*_s*_*.log", "civil"),
    ("OLMo-civil-v3",   "logs/civ3_olmo_r*_s*_*.log",   "civil"),
    ("Qwen-bios-v3",    "logs/bios3_qwen_r*_s*_*.log",  "bios"),
    ("Pythia-bios-v3",  "logs/bios3_pythia_r*_s*_*.log","bios"),
    ("OLMo-bios-v3",    "logs/bios3_olmo_r*_s*_*.log",  "bios"),
]

def parse_log(t):
    row = {}
    m = re.search(rf"RESULT rho={F} seed=(\d+) worstgroup={F} advgroup={F} cueDelta={F}", t)
    if not m: return None
    row.update(rho=float(m[1]), seed=int(m[2]), worst=float(m[3]), cd=float(m[5]))
    for key, rx in [
        ("probe", rf"PROBE_RESULT rho=[\d.]+ seed=\d+ maxAUROC={F}"),
        ("le_dw", rf"LEACE rho=[\d.]+ seed=\d+ L=\d+ trainAUROC_after=[\d.]+ worst=[\d.]+ overall=[\d.]+ delta_worst={F}"),
        ("le_ov", rf"LEACE rho=[\d.]+ seed=\d+ L=\d+ trainAUROC_after=[\d.]+ worst=[\d.]+ overall={F}"),
        ("pl_dw", rf"PROSPECT_RESULT .* delta_worst={F}"),
        ("fx_dw", rf"FIXED(?:20)? rho=[\d.]+ seed=\d+ (?:L=(?:\d+) )?worst=[\d.]+ overall=[\d.]+ delta_worst={F}"),
        ("fx_ov", rf"FIXED(?:20)? rho=[\d.]+ seed=\d+ (?:L=(?:\d+) )?worst=[\d.]+ overall={F}"),
        ("dfr_dw", rf"DFR rho=[\d.]+ seed=\d+ worst=[\d.]+ overall=[\d.]+ delta_worst={F}"),
        ("dfr_ov", rf"DFR rho=[\d.]+ seed=\d+ worst=[\d.]+ overall={F}"),
        ("afr_dw", rf"AFR rho=[\d.]+ seed=\d+ worst=[\d.]+ overall=[\d.]+ delta_worst={F}"),
        ("afr_ov", rf"AFR rho=[\d.]+ seed=\d+ worst=[\d.]+ overall={F}"),
        ("med_peak", rf"MEDIATION_RESULT .* peak={F}"),
        ("med_auc", rf"MEDIATION_RESULT .* auc={F}"),
        ("cd_disc", rf"DISC .* cueDeltaDisc={F}"),
        ("disc_overlap", rf"DISC rho=[\d.]+ seed=\d+ K=\d+ overlap=(\d+)"),
    ]:
        mm = re.search(rx, t)
        row[key] = float(mm[1]) if mm else np.nan
    fm = re.search(rf"FIXED(?:20)? rho=[\d.]+ seed=\d+ L=(\d+) ", t)
    row["LFIX"] = int(fm[1]) if fm else np.nan
    mm = re.search(r"MEDIATION base=[-\d.]+ curve ([\d\-. ]+)", t)
    if mm:
        curve = np.array([float(v) for v in mm[1].split()])
        row["med_curve"] = curve
        nl = len(curve)
        if np.isfinite(row["LFIX"]):
            row["med_at_LFIX"] = float(curve[int(row["LFIX"])])
        pk = np.nanmax(curve)
        row["med_width"] = int(np.sum(curve > 0.5 * pk)) if pk > 0 else nl
        row["med_peak_depth"] = float(np.nanargmax(curve) / (nl - 1))
    mm = re.search(r"PEREX_CD ([\d\-. ]+)", t)
    row["perex_cd"] = np.array([float(v) for v in mm[1].split()]) if mm else None
    bits = {}
    for g in re.finditer(r"PEREX group=(\d)(\d) bits=([01]+)", t):
        bits[(int(g[1]), int(g[2]))] = np.frombuffer(g[3].encode(), dtype=np.uint8) - ord("0")
    row["bits"] = bits if len(bits) == 4 else None
    return row

def parse_cell(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        r = parse_log(open(f).read())
        if r: rows.append(r)
    return rows

if __name__ == "__main__":
    col = lambda rows, k: np.array([r[k] for r in rows])
    data = {}
    for name, pat, task in CELLS:
        rows = parse_cell(pat)
        if len(rows) == 18: data[name] = (rows, task)
        else: print(f"[{name}: {len(rows)}/18 -- skipped]")

    print("== 1. repair table (mean delta_worst | mean post-overall) ==")
    print(f"  {'cell':16s} {'FIXED':>15s} {'prospect':>15s} {'LEACE':>15s} {'DFR':>15s} {'AFR':>15s}")
    for name, (rows, task) in data.items():
        parts = []
        for dk, ok in (("fx_dw","fx_ov"),("pl_dw",None),("le_dw","le_ov"),("dfr_dw","dfr_ov"),("afr_dw","afr_ov")):
            d = col(rows, dk)
            parts.append(f"{np.nanmean(d):+.3f}" + (f"|{np.nanmean(col(rows,ok)):.3f}" if ok else "      "))
        print(f"  {name:16s} " + " ".join(f"{p:>15s}" for p in parts))

    print("\n== 2. mediation -> hard-ablation outcome ==")
    allm, allfx, allw, allpk = [], [], [], []
    for name, (rows, task) in data.items():
        mL = col(rows, "med_at_LFIX"); fx = col(rows, "fx_dw"); wd = col(rows, "med_width")
        ok = np.isfinite(mL) & np.isfinite(fx)
        if ok.sum() > 3:
            r1, p1 = pearsonr(mL[ok], fx[ok]); r2, p2 = pearsonr(wd[ok], fx[ok])
            print(f"  {name:16s} med@LFIX mean={np.nanmean(mL):.2f}  r(med@LFIX, fx_dw)={r1:+.3f} (p={p1:.3f})"
                  f"   r(width, fx_dw)={r2:+.3f} (p={p2:.3f})")
            allm += list(mL[ok]); allfx += list(fx[ok]); allw += list(wd[ok])
            allpk += list(col(rows, "med_peak_depth")[ok])
    if len(allm) > 10:
        r, p = pearsonr(allm, allfx)
        r2, p2 = pearsonr(allw, allfx)
        r3, p3 = pearsonr(allpk, allfx)
        print(f"  POOLED (N={len(allm)}): r(med@LFIX, fx_dw)={r:+.3f} (p={p:.1e})  "
              f"r(width, fx_dw)={r2:+.3f} (p={p2:.1e})  r(peak_depth, fx_dw)={r3:+.3f} (p={p3:.1e})")

    print("\n== 3. discovery agreement (civil cells) ==")
    for name, (rows, task) in data.items():
        if task != "civil": continue
        cd, cdd = col(rows, "cd"), col(rows, "cd_disc")
        ok = np.isfinite(cdd)
        if ok.sum() > 3:
            r1, p1 = pearsonr(cd[ok], cdd[ok]); r2, p2 = pearsonr(cdd[ok], col(rows, "worst")[ok])
            print(f"  {name:16s} r(disc, oracle)={r1:+.3f} (p={p1:.1e})  r(disc, worst)={r2:+.3f} (p={p2:.1e})"
                  f"  overlap mean={np.nanmean(col(rows,'disc_overlap')):.1f}/20")

    print("\n== 4. label-efficiency: r(estimate, true worst) across models ==")
    rng = np.random.default_rng(0)
    for name, (rows, task) in data.items():
        if any(r["bits"] is None or r["perex_cd"] is None for r in rows): 
            print(f"  {name}: missing perex -- skipped"); continue
        true_w = col(rows, "worst")
        line = f"  {name:16s}"
        for k in (5, 10, 25, 50):
            rs = []
            for _ in range(200):
                est = [min(np.mean(rng.choice(r["bits"][g], k, replace=False)) for g in r["bits"])
                       for r in rows]
                rs.append(pearsonr(est, true_w)[0])
            line += f"  k={k}:{np.mean(rs):+.2f}"
        for m in (25, 100, 300):
            rs = []
            for _ in range(200):
                est = [-np.mean(rng.choice(r["perex_cd"], min(m, len(r["perex_cd"])), replace=False))
                       for r in rows]
                rs.append(pearsonr(est, true_w)[0])
            line += f"  cd_m={m}:{np.mean(rs):+.2f}"
        print(line + "   (k = labeled/group, 4k labels; cd_m = unlabeled texts, 0 labels)")
