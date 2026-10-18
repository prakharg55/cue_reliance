"""
Builds the CivilComments pool that every CivilComments run samples from.

Streams google/civil_comments and marks a comment as containing the cue if it
uses one of 32 identity terms (Dixon et al., 2018). Keeps up to 4,000 comments
in each of the four groups (toxic or not, with or without an identity term).
Writes $CIVIL_POOL, by default data/civil_pool.json.

    python training/prepare_civil.py
"""
import os
import json, re, sys
from datasets import load_dataset

# Dixon et al. 2018 identity terms, focused on UNAMBIGUOUS identity mentions (drop generic
# words like white/black/american that aren't reliably identity references).
IDENTITY = ["muslim","islam","islamic","christian","christianity","jewish","jew","catholic",
            "hindu","buddhist","atheist","mormon","gay","lesbian","homosexual","bisexual",
            "transgender","queer","lgbt","lgbtq","latino","latina","hispanic","asian",
            "mexican","arab","immigrant","feminist","muslims","christians","jews","gays"]
IDRE = re.compile(r"\b(" + "|".join(IDENTITY) + r")\b", re.IGNORECASE)

PER_GROUP = 4000   # target per group (toxic/nontoxic x id/noid)


def has_identity(text):
    return IDRE.search(text) is not None


def main():
    ds = load_dataset("google/civil_comments", split="train", streaming=True)
    groups = {("tox", "id"): [], ("tox", "noid"): [], ("non", "id"): [], ("non", "noid"): []}
    seen = 0
    for r in ds:
        seen += 1
        t = (r["text"] or "").strip()
        if not (5 <= len(t) <= 600):           # keep moderate-length comments
            continue
        tox = "tox" if r["toxicity"] >= 0.5 else "non"
        idk = "id" if has_identity(t) else "noid"
        g = (tox, idk)
        if len(groups[g]) < PER_GROUP:
            groups[g].append({"text": t, "toxic": int(tox == "tox"), "identity": int(idk == "id")})
        if all(len(v) >= PER_GROUP for v in groups.values()):
            break
        if seen % 200000 == 0:
            print(f"  scanned {seen}: " + " ".join(f"{k}={len(v)}" for k, v in groups.items()), flush=True)
        if seen > 3_000_000:
            break
    pool = [x for v in groups.values() for x in v]
    out = os.environ.get("CIVIL_POOL", "data/civil_pool.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(pool, f)
    print(f"\nSCANNED {seen} comments. POOL sizes:")
    for k, v in groups.items():
        print(f"  {k}: {len(v)}")
    print(f"saved {len(pool)} examples -> {out}")
    # natural base rates among scanned
    print("EXAMPLES (toxic+identity):")
    for x in groups[("tox", "id")][:3]:
        print(f"  {x['text'][:110]!r}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
