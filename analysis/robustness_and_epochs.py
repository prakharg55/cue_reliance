"""
Section 4 robustness and Table 11. Checks whether the results hold under full
fine-tuning and an instruction-tuned base, and whether measurements after the
first epoch predict final worst-group accuracy.
"""
import glob, re, warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")
F = r"([-+\d.]+)"

def res(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        t = open(f).read()
        m = re.search(rf"RESULT rho={F} seed=(\d+) worstgroup={F} advgroup={F} cueDelta={F}", t)
        if m: rows.append(dict(rho=float(m[1]), seed=int(m[2]), worst=float(m[3]), cd=float(m[5]), f=f))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print("=" * 76)
    print("A. ROBUSTNESS ARMS -- does the conclusion flip?")
    print("=" * 76)
    arms = [("LoRA / base  (frozen reference)", "logs/civ2_qwen_r*_s*_11489602.log"),
            ("FULL fine-tuning",                "logs/rob_full_qwen_r*_s*_*.log"),
            ("LoRA / INSTRUCT base",            "logs/rob_it_qwen_r*_s*_*.log")]
    for name, pat in arms:
        d = res(pat)
        if len(d) < 10: print(f"  {name:32s} only {len(d)} runs"); continue
        r, p = pearsonr(d.cd, d.worst)
        m = smf.ols("worst ~ cd + C(rho)", d).fit()   # exact rho control
        print(f"  {name:32s} N={len(d):2d}  r(cueD,worst)={r:+.3f} (p={p:.1e})"
              f"   mean worst={d.worst.mean():.3f}  cueD range=[{d.cd.min():.2f},{d.cd.max():.2f}]")

    print("\n" + "=" * 76)
    print("B. EARLY WARNING -- is cue reliance readable before training finishes?")
    print("=" * 76)
    rows = []
    for pat, cell in (("logs/ew_civ_qwen_r*_s*_*.log", "Qwen-civil"),
                      ("logs/ew_civ_pythia_r*_s*_*.log", "Pythia-civil")):
        for f in sorted(glob.glob(pat)):
            t = open(f).read()
            fin = re.search(rf"RESULT rho={F} seed=(\d+) worstgroup={F} advgroup={F} cueDelta={F}", t)
            if not fin: continue
            ep = {}
            for m in re.finditer(rf"EPOCH rho={F} seed=(\d+) epoch=(\d+) cueDeltaTrain={F} "
                                 rf"cueDeltaEval={F} worstSmall={F}(?: overall={F} worstClass={F} trainLoss={F})?", t):
                ep[int(m[3])] = dict(cd_tr=float(m[4]), cd_ev=float(m[5]), worst=float(m[6]),
                                     overall=float(m[7]) if m[7] else np.nan,
                                     wclass=float(m[8]) if m[8] else np.nan,
                                     loss=float(m[9]) if m[9] else np.nan)
            if 1 in ep and 4 in ep:
                rows.append(dict(cell=cell, rho=float(fin[1]), seed=int(fin[2]),
                                 final_worst=float(fin[3]), final_cd=float(fin[5]),
                                 e1_cd=ep[1]["cd_tr"], e1_worst=ep[1]["worst"],
                                 e1_overall=ep[1]["overall"], e1_wclass=ep[1]["wclass"],
                                 e1_loss=ep[1]["loss"]))
    d = pd.DataFrame(rows)
    print(f"  N={len(d)} models with full epoch traces ({d.cell.nunique()} cells)\n")
    print("  Predicting FINAL worst-group from epoch-1 signals (cell FE):")
    print(f"    {'predictor':28s} {'r':>8s} {'p':>10s}   {'+ exact rho control':>22s}")
    for key, lab in (("e1_cd", "epoch-1 cueDelta"), ("final_cd", "FINAL cueDelta (ceiling)"),
                     ("e1_worst", "epoch-1 worst-group (oracle, needs group labels)"),
                     ("e1_wclass", "epoch-1 worst-CLASS (label-free baseline)"),
                     ("e1_overall", "epoch-1 overall acc"), ("e1_loss", "epoch-1 train loss")):
        sub = d.dropna(subset=[key])
        if len(sub) < 10 or sub[key].std() == 0: continue
        r, p = pearsonr(sub[key], sub.final_worst)
        m = smf.ols(f"final_worst ~ {key} + C(cell) + C(rho)", sub).fit()
        print(f"    {lab:28.28s} {r:+8.3f} {p:10.2e}   beta={m.params[key]:+.4f} p={m.pvalues[key]:.4f}"
              + ("  [sig]" if m.pvalues[key] < .05 else "  [n.s.]"))
    r_e = pearsonr(d.e1_cd, d.final_worst)[0]; r_f = pearsonr(d.final_cd, d.final_worst)[0]
    print(f"\n  retention: R2(epoch-1)/R2(final) = {r_e**2:.3f}/{r_f**2:.3f} = {r_e**2/r_f**2:.1%}"
          "  <- how much of the end-of-training signal is already present after 1 of 4 epochs")
