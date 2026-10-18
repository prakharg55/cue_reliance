"""
Trains and measures one Bias-in-Bios model.

The Bias-in-Bios counterpart of train_civil.py: professor versus physician,
with gendered words as the cue. Cue reliance is measured on male-worded bios
by swapping their gendered words.

    python training/train_bios.py --model qwen --rho 0.97 --seed 0
"""
import os
import argparse, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import local_model_path, MODELS
from prepare_bios import gender_swap

POOL = os.environ.get("BIOS_POOL", "data/bios_pool.json")
A_ID, B_ID = None, None
LABEL1, LABEL0 = "professor", "physician"   # overwritten from pool metadata


def prompt(text):
    return f"Bio: {text}\nQuestion: Is this person a {LABEL1} or a {LABEL0}?\nAnswer:"


def load_groups(split=0.8):
    """(g_train, g_eval): disjoint deduplicated 80/20 partition; keys (label, cue)."""
    meta = json.load(open(POOL))
    global LABEL1, LABEL0
    LABEL1, LABEL0 = meta["label1"], meta["label0"]
    g = {(1, 1): [], (1, 0): [], (0, 1): [], (0, 0): []}
    for x in meta["pool"]:
        g[(x["label"], x["cue"])].append(x["text"])
    gtr, gev = {}, {}
    for k, v in g.items():
        v = list(dict.fromkeys(v))
        cut = int(len(v) * split); gtr[k] = v[:cut]; gev[k] = v[cut:]
    return gtr, gev


def make_train(g, rho, n, seed):
    """cue (male-worded) is a rho-reliable shortcut for label 1: P(label1|male)=rho,
    P(label1|female)=1-rho; label balanced. Identical structure to common.make_train."""
    rng = np.random.default_rng(seed)
    def draw(key, k): return list(rng.choice(g[key], size=k, replace=len(g[key]) < k))
    half = n // 2
    l1c = int(round(half * rho)); l0c = half - l1c
    l1n = half - l1c; l0n = l1c
    items = [(t, 1) for t in draw((1, 1), l1c)] + [(t, 0) for t in draw((0, 1), l0c)] \
          + [(t, 1) for t in draw((1, 0), l1n)] + [(t, 0) for t in draw((0, 0), l0n)]
    rng.shuffle(items); return items


def build_ids(tok, items, max_len=160):
    n = len(items)
    ids = torch.full((n, max_len), tok.pad_token_id, dtype=torch.long)
    lab = torch.full((n, max_len), -100, dtype=torch.long)
    am = torch.zeros((n, max_len), dtype=torch.long)
    for i, (text, y) in enumerate(items):
        full = tok(prompt(text) + (" A" if y else " B"))["input_ids"][:max_len]
        ids[i, :len(full)] = torch.tensor(full); am[i, :len(full)] = 1; lab[i, len(full) - 1] = full[-1]
    return ids, lab, am


@torch.no_grad()
def group_acc(model, tok, g, seed, device, per=400, bs=48):
    rng = np.random.default_rng(seed); out = {}
    for key in [(1, 1), (1, 0), (0, 1), (0, 0)]:
        texts = list(rng.choice(g[key], size=min(per, len(g[key])), replace=False))
        hit = 0
        for s in range(0, len(texts), bs):
            ch = texts[s:s + bs]
            enc = tok([prompt(t) for t in ch], return_tensors="pt", padding=True,
                      truncation=True, max_length=160).to(device)
            lg = model(**enc).logits[torch.arange(len(ch)), enc["attention_mask"].sum(1) - 1]
            pred = (lg[:, A_ID] > lg[:, B_ID]).cpu().numpy().astype(int)
            hit += int((pred == key[0]).sum())
        out[key] = hit / len(texts)
    return out


@torch.no_grad()
def batched_logit_diff(model, tok, texts, device, bs=48):
    out = []
    for s in range(0, len(texts), bs):
        ch = texts[s:s + bs]
        enc = tok([prompt(t) for t in ch], return_tensors="pt", padding=True,
                  truncation=True, max_length=160).to(device)
        lg = model(**enc).logits[torch.arange(len(ch)), enc["attention_mask"].sum(1) - 1]
        out.extend((lg[:, A_ID] - lg[:, B_ID]).float().cpu().tolist())
    return np.array(out)


def cue_delta_batched(model, tok, texts, device):
    ld = batched_logit_diff(model, tok, texts, device)
    ls = batched_logit_diff(model, tok, [gender_swap(t) for t in texts], device)
    return float(np.mean(ld - ls))


def get_layers(model):
    m = model.base_model.model if hasattr(model, "base_model") else model
    if hasattr(m, "model") and hasattr(m.model, "layers"): return m.model.layers
    if hasattr(m, "gpt_neox"): return m.gpt_neox.layers
    raise ValueError("unknown architecture")


@torch.no_grad()
def cue_directions(model, tok, g, device, n=400):
    """unit direction per layer: mean[h(x) - h(gender_swap(x))] at the decision position,
    over TRAIN male-cue texts (paired counterfactual, sharper than civil's unpaired)."""
    rng = np.random.default_rng(0)
    texts = list(rng.choice(g[(1, 1)] + g[(0, 1)], size=n, replace=False))
    nl = model.config.num_hidden_layers
    acc = [None] * (nl + 1)
    for t in texts:
        for sgn, txt in ((1, t), (-1, gender_swap(t))):
            enc = tok(prompt(txt), return_tensors="pt", truncation=True, max_length=160).to(device)
            hs = model(**enc, output_hidden_states=True).hidden_states
            for L in range(nl + 1):
                d = sgn * hs[L][0, -1].float()
                acc[L] = d if acc[L] is None else acc[L] + d
    dirs = []
    for L in range(nl + 1):
        v = acc[L] / n
        dirs.append((v / (v.norm() + 1e-8)).to(device))
    return dirs


def make_projection_hook(dhat):
    def hook(mod, inp, out):
        istuple = isinstance(out, tuple)
        h = out[0] if istuple else out
        h = h - (h @ dhat).unsqueeze(-1) * dhat
        return ((h,) + tuple(out[1:])) if istuple else h
    return hook


def ablate_eval(model, tok, g, layer_idx, dhat, seed, device):
    h = get_layers(model)[layer_idx].register_forward_hook(make_projection_hook(dhat))
    try:
        return group_acc(model, tok, g, seed, device)
    finally:
        h.remove()


@torch.no_grad()
def collect_resid(model, tok, texts, device, bs=24):
    outs = []
    for s in range(0, len(texts), bs):
        ch = texts[s:s + bs]
        enc = tok([prompt(t) for t in ch], return_tensors="pt", padding=True,
                  truncation=True, max_length=160).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        idx = enc["attention_mask"].sum(1) - 1
        outs.append(torch.stack([h[torch.arange(len(ch)), idx] for h in hs], dim=1).float().cpu())
    return torch.cat(outs).numpy()


def main():
    from train_civil import leace_eraser, make_leace_hook
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.8)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="qwen", choices=list(MODELS))
    ap.add_argument("--n_prospect", type=int, default=96)
    ap.add_argument("--n_probe", type=int, default=400)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MODELS[args.model]
    mp = local_model_path(cfg["subdir"])
    tok = AutoTokenizer.from_pretrained(mp)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    global A_ID, B_ID
    A_ID = tok.encode(" A")[-1]; B_ID = tok.encode(" B")[-1]
    model = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.float32).to(device)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=cfg["targets"]))
    gtr, gev = load_groups()
    ids, lab, am = build_ids(tok, make_train(gtr, args.rho, args.n_train, args.seed))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    n = ids.shape[0]; spe = n // args.batch
    print(f"BIOSV2 rho={args.rho} seed={args.seed} model={args.model} "
          f"labels={LABEL1}/{LABEL0} device={device}", flush=True)
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n)
        for it in range(spe):
            idx = perm[it*args.batch:(it+1)*args.batch]
            o = model(input_ids=ids[idx].to(device), attention_mask=am[idx].to(device), labels=lab[idx].to(device))
            opt.zero_grad(); o.loss.backward(); opt.step()
    model.eval()

    def worst(ga): return min(ga.values())
    def overall(ga): return float(np.mean(list(ga.values())))

    ga = group_acc(model, tok, gev, args.seed + 1, device)
    rng = np.random.default_rng(args.seed + 3)
    cdtexts = list(rng.choice(gev[(1, 1)] + gev[(0, 1)], size=300, replace=False))
    cd = cue_delta_batched(model, tok, cdtexts, device)
    wg = worst(ga)
    print(f"RESULT rho={args.rho} seed={args.seed} worstgroup={wg:.3f} "
          f"advgroup={(ga[(1,0)]+ga[(0,1)])/2:.3f} cueDelta={cd:.3f} "
          f"(G11={ga[(1,1)]:.2f} G10={ga[(1,0)]:.2f} G01={ga[(0,1)]:.2f} G00={ga[(0,0)]:.2f})", flush=True)

    layers = get_layers(model)
    nl = model.config.num_hidden_layers
    dirs = cue_directions(model, tok, gtr, device)

    # PROBE: gender decodability per layer (male-cue vs female-cue eval texts)
    rng = np.random.default_rng(args.seed + 7)
    tr_pos = list(rng.choice(gtr[(1, 1)] + gtr[(0, 1)], size=args.n_probe, replace=False))
    tr_neg = list(rng.choice(gtr[(1, 0)] + gtr[(0, 0)], size=args.n_probe, replace=False))
    ev_pos = list(rng.choice(gev[(1, 1)] + gev[(0, 1)], size=args.n_probe, replace=False))
    ev_neg = list(rng.choice(gev[(1, 0)] + gev[(0, 0)], size=args.n_probe, replace=False))
    Rtr = collect_resid(model, tok, tr_pos + tr_neg, device)
    Rev = collect_resid(model, tok, ev_pos + ev_neg, device)
    ytr = np.array([1] * len(tr_pos) + [0] * len(tr_neg))
    yev = np.array([1] * len(ev_pos) + [0] * len(ev_neg))
    aucs = []
    for L in range(nl + 1):
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(Rtr[:, L], ytr)
        aucs.append(float(roc_auc_score(yev, clf.decision_function(Rev[:, L]))))
    print("PROBE per-layer AUROC " + " ".join(f"{a:.3f}" for a in aucs), flush=True)
    print(f"PROBE_RESULT rho={args.rho} seed={args.seed} maxAUROC={max(aucs):.4f} "
          f"argmaxL={int(np.argmax(aucs))}", flush=True)

    # PROSPECT: train-only layer choice
    ptexts = list(rng.choice(gtr[(1, 1)] + gtr[(0, 1)], size=args.n_prospect, replace=False))
    base_cd = cue_delta_batched(model, tok, ptexts, device)
    curve = []
    for L in range(nl):
        h = layers[L].register_forward_hook(make_projection_hook(dirs[L + 1]))
        try:
            curve.append(cue_delta_batched(model, tok, ptexts, device))
        finally:
            h.remove()
    Lstar = int(np.argmin(curve))
    print("PROSPECT train-cueDelta base={:.3f} curve ".format(base_cd)
          + " ".join(f"{c:.3f}" for c in curve), flush=True)
    ga_p = ablate_eval(model, tok, gev, Lstar, dirs[Lstar + 1], args.seed + 1, device)
    print(f"PROSPECT_RESULT rho={args.rho} seed={args.seed} Lstar={Lstar} "
          f"worst={worst(ga_p):.3f} overall={overall(ga_p):.3f} delta_worst={worst(ga_p)-wg:+.3f}", flush=True)

    LFIX = round(20 * nl / 24)
    ga_20 = ablate_eval(model, tok, gev, LFIX, dirs[LFIX + 1], args.seed + 1, device)
    print(f"FIXED rho={args.rho} seed={args.seed} L={LFIX} worst={worst(ga_20):.3f} "
          f"overall={overall(ga_20):.3f} delta_worst={worst(ga_20)-wg:+.3f}", flush=True)

    mu, A = leace_eraser(Rtr[:, LFIX + 1], ytr)
    Xer = Rtr[:, LFIX + 1] - (Rtr[:, LFIX + 1] - mu) @ A.T
    clf = LogisticRegression(max_iter=2000, C=0.1).fit(Xer, ytr)
    auc_er = float(roc_auc_score(ytr, clf.decision_function(Xer)))
    h = layers[LFIX].register_forward_hook(make_leace_hook(mu, A, device))
    try:
        ga_le = group_acc(model, tok, gev, args.seed + 1, device)
    finally:
        h.remove()
    print(f"LEACE rho={args.rho} seed={args.seed} L={LFIX} trainAUROC_after={auc_er:.3f} "
          f"worst={worst(ga_le):.3f} overall={overall(ga_le):.3f} delta_worst={worst(ga_le)-wg:+.3f}", flush=True)
    print("V2 DONE", flush=True)


if __name__ == "__main__":
    main()
