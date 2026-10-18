"""
Audits the two edits on the evaluation texts (Appendix B.2).

Counts how many texts each edit changes and how many tokens it replaces, and
checks whether the Bios swap undoes itself. Needs the pools built by
prepare_civil.py and prepare_bios.py.

    python training/audit_edits.py > results/audit_edits.txt
"""
import numpy as np
import common
from common import IDRE, neutralize
import train_bios
from prepare_bios import gender_swap, WRE

if __name__ == "__main__":
    print("== CivilComments: identity neutralization ==")
    gtr, gev = common.load_groups()
    print("  eval group sizes:", {k: len(v) for k, v in gev.items()},
          " train:", {k: len(v) for k, v in gtr.items()})
    idtexts = gev[(1, 1)] + gev[(0, 1)]
    hits = [len(IDRE.findall(t)) for t in idtexts]
    lens = [len(t.split()) for t in idtexts]
    changed = sum(1 for t in idtexts if neutralize(t) != t)
    print(f"  identity-present eval texts: n={len(idtexts)}; changed by neutralize: "
          f"{100*changed/len(idtexts):.1f}%")
    print(f"  identity tokens replaced/text: mean={np.mean(hits):.2f} median={np.median(hits):.0f} "
          f"max={max(hits)};  text length mean={np.mean(lens):.0f} words "
          f"(edit fraction {100*np.mean(hits)/np.mean(lens):.1f}%)")
    from collections import Counter
    top = Counter(w.lower() for t in idtexts for w in IDRE.findall(t)).most_common(8)
    print("  top terms:", ", ".join(f"{w}({c})" for w, c in top))
    print("  examples:")
    for t in [t for t in idtexts if 6 <= len(t.split()) <= 18][:3]:
        print(f"    - {t.strip()[:130]}")
        print(f"      -> {neutralize(t).strip()[:130]}")

    # exactness of the rho assignment: realized conditionals of an actual training
    # draw, with cue presence re-detected from the text (not the group key)
    for rho in (0.55, 0.97):
        items = common.make_train(gtr, rho=rho, n=4000, seed=0)
        y = np.array([lab for _, lab in items])
        c = np.array([bool(IDRE.search(t)) for t, _ in items])
        print(f"  realized rho={rho} draw (n=4000, seed 0): P(toxic|cue)={y[c].mean():.3f}  "
              f"P(toxic|no cue)={y[~c].mean():.3f}  P(toxic)={y.mean():.3f}")

    print("\n== Bios: gender-lexicon swap ==")
    gtr, gev = train_bios.load_groups()
    print(f"  labels: 1={train_bios.LABEL1} 0={train_bios.LABEL0}")
    print("  eval group sizes:", {k: len(v) for k, v in gev.items()})
    mtexts = gev[(1, 1)] + gev[(0, 1)]
    hits = [len(WRE.findall(t)) for t in mtexts]
    lens = [len(t.split()) for t in mtexts]
    changed = sum(1 for t in mtexts if gender_swap(t) != t)
    rt_bad = sum(1 for t in mtexts if gender_swap(gender_swap(t)) != t)
    print(f"  male-cue eval texts: n={len(mtexts)}; changed by swap: {100*changed/len(mtexts):.1f}%")
    print(f"  lexicon tokens swapped/text: mean={np.mean(hits):.2f} median={np.median(hits):.0f} "
          f"max={max(hits)};  length mean={np.mean(lens):.0f} words "
          f"(edit fraction {100*np.mean(hits)/np.mean(lens):.1f}%)")
    print(f"  round-trip swap(swap(x)) != x: {100*rt_bad/len(mtexts):.1f}%  "
          f"(his/her->her->his collapse; document in appendix)")
    top = Counter(w.lower() for t in mtexts for w in WRE.findall(t)).most_common(8)
    print("  top tokens:", ", ".join(f"{w}({c})" for w, c in top))
    print("  examples:")
    for t in [t for t in mtexts if 8 <= len(t.split()) <= 22][:3]:
        print(f"    - {t.strip()[:130]}")
        print(f"      -> {gender_swap(t).strip()[:130]}")
