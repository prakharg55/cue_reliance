"""
Evaluates a public toxicity classifier with the study's protocol (Table 12).

Loads a classifier from the Hugging Face Hub, measures worst-group accuracy on
the same evaluation groups, and measures cue reliance with the same edit,
using the classifier's toxicity logit as the decision score. The eight
classifiers are listed in public_models.json. Some of them may have been
trained on CivilComments, including comments scored here.

    python training/eval_public.py --model unitary/toxic-bert
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from common import load_groups, neutralize

# filled in from the model survey; label_key names the class treated as "toxic"
REGISTRY = {}
REG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public_models.json")


def load_registry():
    global REGISTRY
    try:
        REGISTRY = json.load(open(REG_PATH))
    except FileNotFoundError:
        REGISTRY = {}
    return REGISTRY


def toxic_index(model, want=None):
    """Index of the label meaning 'toxic'. Uses config.id2label so we do not
    hardcode per-model conventions (they disagree)."""
    id2l = {int(k): str(v).lower() for k, v in (model.config.id2label or {}).items()}
    if want:
        for i, lab in id2l.items():
            if str(want).lower() == lab:
                return i, id2l
    for i, lab in id2l.items():
        if any(k in lab for k in ("toxic", "hate", "offensive", "abusive", "label_1")):
            return i, id2l
    return (1 if len(id2l) > 1 else 0), id2l


def is_multilabel(model):
    """Multi-label heads (toxic-bert: toxic/severe_toxic/obscene/threat/insult/
    identity_hate) are independent sigmoids, so 'toxic minus the other classes'
    is meaningless -- the rare classes sit at large negative logits and every
    text scores positive. Detected this the hard way: worst-group came out 0.000
    with AUROC 0.846, i.e. a pure thresholding artifact."""
    if getattr(model.config, "problem_type", None) == "multi_label_classification":
        return True
    labs = {str(v).lower() for v in (model.config.id2label or {}).values()}
    return len(labs & {"severe_toxic", "obscene", "threat", "insult", "identity_hate"}) >= 2


@torch.no_grad()
def scores(model, tok, texts, device, tox_i, multilabel, bs=64, max_len=128):
    """Scalar toxicity score. Multi-label: the toxic logit alone. Single-label:
    toxic logit minus the mean of the competing classes (= the binary logit
    difference when there are two)."""
    out = []
    for s in range(0, len(texts), bs):
        ch = texts[s:s + bs]
        enc = tok(ch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(device)
        lg = model(**enc).logits.float()
        if lg.shape[-1] == 1 or multilabel:
            out.extend(lg[:, tox_i].cpu().tolist())
        else:
            others = [j for j in range(lg.shape[-1]) if j != tox_i]
            out.extend((lg[:, tox_i] - lg[:, others].mean(-1)).cpu().tolist())
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", default=None, help="override which label means toxic")
    ap.add_argument("--per", type=int, default=400)
    ap.add_argument("--n_cd", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    reg = load_registry()
    if args.list or not args.model:
        print(json.dumps(reg, indent=2) if reg else f"no registry at {REG_PATH}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = reg.get(args.model, {})
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
    tox_i, id2l = toxic_index(model, args.label or spec.get("label"))
    print(f"PUBLIC model={args.model} labels={id2l} toxic_index={tox_i} device={device}", flush=True)

    gtr, gev = load_groups()        # identical eval split to the main study
    multilabel = is_multilabel(model)
    rng = np.random.default_rng(args.seed)

    # Public models differ wildly in calibration, so a fixed threshold would
    # confound "relies on the cue" with "is mis-calibrated for this dataset".
    # Pick each model's threshold on TRAIN-side text (never the eval split) to
    # maximize balanced accuracy, then evaluate worst-group at that threshold.
    cal_t, cal_y = [], []
    for key in [(1, 1), (1, 0), (0, 1), (0, 0)]:
        sel = list(rng.choice(gtr[key], size=min(200, len(gtr[key])), replace=False))
        cal_t += sel; cal_y += [key[0]] * len(sel)
    cal_s = scores(model, tok, cal_t, device, tox_i, multilabel)
    cal_y = np.array(cal_y)
    cands = np.quantile(cal_s, np.linspace(0.02, 0.98, 97))
    bal = [np.mean([( (cal_s > c).astype(int)[cal_y == k] == k).mean() for k in (0, 1)])
           for c in cands]
    thr = float(cands[int(np.argmax(bal))])
    print(f"PUBLIC_CAL model={args.model} multilabel={multilabel} threshold={thr:.3f} "
          f"balanced_acc={max(bal):.3f}", flush=True)

    # threshold-free where possible: a score of 0 is the natural decision point
    # for a logit difference; report accuracy at that point plus the AUROC-style
    # rank quantities that do not depend on calibration.
    ga, allsc, ally = {}, [], []
    for key in [(1, 1), (1, 0), (0, 1), (0, 0)]:
        sel = list(rng.choice(gev[key], size=min(args.per, len(gev[key])), replace=False))
        sc = scores(model, tok, sel, device, tox_i, multilabel)
        ga[key] = float(((sc > thr).astype(int) == key[0]).mean())
        allsc.append(sc); ally.append(np.full(len(sc), key[0]))
    worst = min(ga.values()); overall = float(np.mean(list(ga.values())))

    from sklearn.metrics import roc_auc_score
    auroc = float(roc_auc_score(np.concatenate(ally), np.concatenate(allsc)))

    rng2 = np.random.default_rng(args.seed + 2)
    cd_texts = list(rng2.choice(gev[(1, 1)] + gev[(0, 1)], size=args.n_cd, replace=False))
    s_orig = scores(model, tok, cd_texts, device, tox_i, multilabel)
    s_neut = scores(model, tok, [neutralize(t) for t in cd_texts], device, tox_i, multilabel)
    cd = float(np.mean(s_orig - s_neut))

    print(f"PUBLIC_RESULT model={args.model} worst={worst:.3f} overall={overall:.3f} "
          f"auroc={auroc:.3f} cueDelta={cd:.3f} "
          f"G11={ga[(1,1)]:.3f} G10={ga[(1,0)]:.3f} G01={ga[(0,1)]:.3f} G00={ga[(0,0)]:.3f}",
          flush=True)
    print("PUBLIC DONE", flush=True)


if __name__ == "__main__":
    main()
