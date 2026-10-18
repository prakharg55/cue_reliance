"""
Table 12. Eight public toxicity classifiers evaluated with the study's protocol
(training/eval_public.py). Overall accuracy tracks worst-group accuracy closely
(Spearman +0.98), and cue reliance does not predict it.

    python analysis/public_classifiers.py [logfile]
"""
import glob, re, sys
import numpy as np
from scipy.stats import pearsonr, spearmanr

F = r"([-+\d.]+)"
pat = sys.argv[1] if len(sys.argv) > 1 else "logs/pubredo*.log"
rows = []
for f in sorted(glob.glob(pat)):
    for m in re.finditer(rf"PUBLIC_RESULT model=(\S+) worst={F} overall={F} "
                         rf"auroc={F} cueDelta={F}", open(f).read()):
        rows.append((m.group(1), *(float(m.group(i)) for i in (2, 3, 4, 5))))

print(f"== PUBLIC CLASSIFIERS (n={len(rows)}) ==")
for name, w, o, a, cd in rows:
    print(f"  {name:55s} worst={w:.3f} overall={o:.3f} auroc={a:.3f} cueDelta={cd:+.3f}")
w = np.array([r[1] for r in rows]); o = np.array([r[2] for r in rows])
au = np.array([r[3] for r in rows]); cd = np.array([r[4] for r in rows])
rs, ps = spearmanr(o, w)
rsa, psa = spearmanr(au, w)
rcs, pcs = spearmanr(cd, w)
rp, pp = pearsonr(cd, w)
print(f"\n  capability: Spearman(overall, worst) = {rs:+.3f}  p={ps:.4f}")
print(f"  capability: Spearman(auroc, worst)   = {rsa:+.3f}  p={psa:.4f}")
print(f"  reliance:   Spearman(cueDelta, worst) = {rcs:+.3f}  p={pcs:.4f}")
print(f"  reliance:   Pearson(cueDelta, worst)  = {rp:+.3f}  p={pp:.4f}"
      f"   [{'wrong-signed' if rp > 0 else 'right-signed'}, "
      f"{'n.s.' if pp >= 0.05 else 'significant'}]")
print("\n  Reading: across uncontrolled public models, capability differences")
print("  dominate worst-group accuracy and the reliance measurement does not")
print("  separate from them. This is the scope limit the paper reports.")
