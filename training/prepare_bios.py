"""
Builds the Bias-in-Bios pool and defines the gender-word swap.

Picks the profession pair with the largest balanced groups (professor and
physician), splits each profession by gendered words, and keeps up to 4,000
bios per group. gender_swap exchanges male and female words, and the Bios edit
applies it to male-worded bios. Writes $BIOS_POOL, by default
data/bios_pool.json. Needs network access.

    python training/prepare_bios.py
"""
import os
import json
import re
from collections import Counter
from datasets import load_dataset

OUT = os.environ.get("BIOS_POOL", "data/bios_pool.json")
PER_GROUP = 4000

# bidirectional gender lexicon (\b-bounded, lowercase matching; case preserved on 1st char)
PAIRS = [("he", "she"), ("him", "her"), ("his", "her"), ("himself", "herself"),
         ("mr", "ms"), ("man", "woman"), ("men", "women"), ("male", "female"),
         ("husband", "wife"), ("father", "mother"), ("son", "daughter"),
         ("brother", "sister"), ("gentleman", "lady"), ("boy", "girl")]
M2F = {}
F2M = {}
for m, f in PAIRS:
    M2F[m] = f
    if f not in F2M: F2M[f] = m   # her -> his (first mapping wins; ambiguity documented)
LEX = {**M2F, **F2M}
WRE = re.compile(r"\b(" + "|".join(sorted(LEX, key=len, reverse=True)) + r")\b", re.IGNORECASE)

def gender_swap(text):
    def rep(m):
        w = m.group(0); lo = w.lower()
        out = LEX[lo]
        return out.capitalize() if w[0].isupper() else out
    return WRE.sub(rep, text)

def has_gender_word(text):
    return WRE.search(text) is not None

PREFERRED = ["professor", "physician", "attorney", "teacher", "journalist",
             "photographer", "psychologist", "surgeon", "nurse", "architect"]

# standard bias_in_bios profession id -> name (De-Arteaga et al. 2019 ordering)
PROF_NAMES = ["accountant","architect","attorney","chiropractor","comedian","composer",
              "dentist","dietitian","dj","filmmaker","interior_designer","journalist",
              "model","nurse","painter","paralegal","pastor","personal_trainer",
              "photographer","physician","poet","professor","psychologist","rapper",
              "software_engineer","surgeon","teacher","yoga_teacher"]

def main():
    ds = load_dataset("LabHC/bias_in_bios")
    rows = []
    for split in ["train", "dev", "test"]:
        if split in ds: rows.extend(ds[split])
    print(f"total rows: {len(rows)}", flush=True)

    def prof_name(x):
        return PROF_NAMES[int(x["profession"])]

    counts = Counter((prof_name(x), x["gender"]) for x in rows if has_gender_word(x["hard_text"]))
    print("top (profession, gender) counts:", counts.most_common(20), flush=True)

    avail = [p for p in PREFERRED if counts.get((p, 0), 0) > 0 and counts.get((p, 1), 0) > 0]
    best, best_min = None, -1
    for i, p1 in enumerate(avail):
        for p0 in avail[i + 1:]:
            m = min(counts[(p1, 0)], counts[(p1, 1)], counts[(p0, 0)], counts[(p0, 1)])
            if m > best_min: best, best_min = (p1, p0), m
    p1, p0 = best
    print(f"CHOSEN PAIR: label1={p1} label0={p0}  min-cell={best_min}", flush=True)

    pool = []
    cell = Counter()
    for x in rows:
        pn = prof_name(x)
        if pn not in (p1, p0): continue
        t = x["hard_text"].strip()
        if not has_gender_word(t) or len(t) < 40: continue
        lab = 1 if pn == p1 else 0
        cue = 1 if x["gender"] == 0 else 0   # bias_in_bios: gender 0=male, 1=female
        if cell[(lab, cue)] >= PER_GROUP: continue
        cell[(lab, cue)] += 1
        pool.append({"text": t[:600], "label": lab, "cue": cue})
    print("final cells:", dict(cell), flush=True)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump({"label1": p1, "label0": p0, "pool": pool}, open(OUT, "w"))
    print(f"wrote {len(pool)} bios -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
