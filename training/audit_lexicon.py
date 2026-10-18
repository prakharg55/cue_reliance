"""
Recomputes the document frequencies behind the placebo lexicon (Appendix B.3).

The placebo words are matched to the identity terms by how many comments in
the training pool contain each word. Needs the CivilComments pool.

    python training/audit_lexicon.py > results/audit_lexicon.txt
"""
import json
import os
import re

from common import IDENTITY, POOL

PLACEBO = os.environ.get("PLACEBO_WORDS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "placebo_words.json"))


def doc_freq(words, texts):
    """Number of documents containing each word, whole-word and case-insensitive."""
    out = {}
    for w in words:
        rx = re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE)
        out[w] = sum(1 for t in texts if rx.search(t))
    return out


if __name__ == "__main__":
    texts = [x["text"] for x in json.load(open(POOL))]
    placebo = json.load(open(PLACEBO))
    di, dp = doc_freq(IDENTITY, texts), doc_freq(placebo, texts)

    print(f"== LEXICON FREQUENCY AUDIT (recomputed from the training pool, "
          f"n={len(texts)} docs) ==")
    print(f"  identity lexicon: {len(IDENTITY)} terms, "
          f"total document frequency = {sum(di.values())}")
    print(f"  placebo  lexicon: {len(placebo)} words, "
          f"total document frequency = {sum(dp.values())}")
    for name, d in (("identity", di), ("placebo ", dp)):
        ranked = sorted(d.items(), key=lambda kv: -kv[1])
        print(f"  per-term {name} df: " + ", ".join(f"{w}:{c}" for w, c in ranked))
