"""
Trains and measures one CivilComments model.

Fine-tunes the chosen model with LoRA on 4,000 comments in which identity
terms come with the toxic label at rate rho. It then prints worst-group
accuracy and cue reliance (the RESULT line), a linear probe for the cue at
every layer, and three repairs: directional ablation at a fixed layer and at a
layer chosen from training data alone, and LEACE.

    python training/train_civil.py --model qwen --rho 0.97 --seed 0

Models are qwen, qwen15, pythia and pythia14 (see MODELS in common.py).
"""
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import common as CF
from common import (local_model_path, prompt, neutralize, load_groups, make_train,
                      build_ids, group_acc, MODELS)
from directions import get_layers, identity_directions, ablate_eval


@torch.no_grad()
def batched_logit_diff(model, tok, texts, device, bs=32):
    """logit(A)-logit(B) at the last real token for a list of prompted texts."""
    out = []
    for s in range(0, len(texts), bs):
        ch = texts[s:s + bs]
        enc = tok([prompt(t) for t in ch], return_tensors="pt", padding=True,
                  truncation=True, max_length=96).to(device)
        lg = model(**enc).logits[torch.arange(len(ch)), enc["attention_mask"].sum(1) - 1]
        out.extend((lg[:, CF.A_ID] - lg[:, CF.B_ID]).float().cpu().tolist())
    return np.array(out)


@torch.no_grad()
def cue_delta_batched(model, tok, texts, device):
    """cueDelta over given identity-present texts (batched; same definition as
    common.cue_delta but usable under hooks and on arbitrary text lists)."""
    ld_orig = batched_logit_diff(model, tok, texts, device)
    ld_neut = batched_logit_diff(model, tok, [neutralize(t) for t in texts], device)
    return float(np.mean(ld_orig - ld_neut))


def make_projection_hook(dhat):
    def hook(mod, inp, out):
        istuple = isinstance(out, tuple)
        h = out[0] if istuple else out
        h = h - (h @ dhat).unsqueeze(-1) * dhat
        return ((h,) + tuple(out[1:])) if istuple else h
    return hook


@torch.no_grad()
def collect_resid(model, tok, texts, device, bs=16):
    """last-real-token hidden states at every layer -> (n, n_layers+1, d)."""
    outs = []
    for s in range(0, len(texts), bs):
        ch = texts[s:s + bs]
        enc = tok([prompt(t) for t in ch], return_tensors="pt", padding=True,
                  truncation=True, max_length=96).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states   # (L+1) x (b, T, d)
        idx = enc["attention_mask"].sum(1) - 1
        stack = torch.stack([h[torch.arange(len(ch)), idx] for h in hs], dim=1)
        outs.append(stack.float().cpu())
    return torch.cat(outs).numpy()


def leace_eraser(X, z, ridge=1e-3):
    """Closed-form LEACE (Belrose et al. 2023) for a binary concept z on rows of X.
    Returns (mu, A) with eraser x' = x - A @ (x - mu)."""
    X = np.asarray(X, np.float64); z = np.asarray(z, np.float64).reshape(-1, 1)
    mu = X.mean(0)
    Xc = X - mu; zc = z - z.mean()
    d = X.shape[1]
    sigma = Xc.T @ Xc / len(X) + ridge * np.eye(d)
    evals, evecs = np.linalg.eigh(sigma)
    evals = np.clip(evals, 1e-8, None)
    W = evecs @ np.diag(evals ** -0.5) @ evecs.T          # sigma^{-1/2}
    Winv = evecs @ np.diag(evals ** 0.5) @ evecs.T        # sigma^{+1/2}
    sxz = Xc.T @ zc / len(X)                              # (d, 1)
    M = W @ sxz
    M = M / (np.linalg.norm(M) + 1e-12)
    P = M @ M.T                                           # rank-1 projector, whitened space
    A = Winv @ P @ W                                      # oblique eraser in raw space
    return mu, A


def make_leace_hook(mu, A, device):
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    A_t = torch.tensor(A, dtype=torch.float32, device=device)
    def hook(mod, inp, out):
        istuple = isinstance(out, tuple)
        h = out[0] if istuple else out
        h = h - (h - mu_t) @ A_t.T
        return ((h,) + tuple(out[1:])) if istuple else h
    return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.8)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="qwen", choices=list(MODELS))
    ap.add_argument("--n_prospect", type=int, default=96)
    ap.add_argument("--n_probe", type=int, default=400)
    args = ap.parse_args()

    # ---- training block: byte-identical order to common.main for determinism ----
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MODELS[args.model]
    mp = local_model_path(cfg["subdir"])
    tok = AutoTokenizer.from_pretrained(mp)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    CF.A_ID = tok.encode(" A")[-1]; CF.B_ID = tok.encode(" B")[-1]
    model = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.float32).to(device)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=cfg["targets"]))
    gtr, gev = load_groups()
    ids, lab, am = build_ids(tok, make_train(gtr, args.rho, args.n_train, args.seed))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    n = ids.shape[0]; spe = n // args.batch
    print(f"V2 rho={args.rho} seed={args.seed} model={args.model} device={device}", flush=True)
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n)
        for it in range(spe):
            idx = perm[it*args.batch:(it+1)*args.batch]
            o = model(input_ids=ids[idx].to(device), attention_mask=am[idx].to(device), labels=lab[idx].to(device))
            opt.zero_grad(); o.loss.backward(); opt.step()
    model.eval()

    def worst(ga): return min(ga.values())
    def overall(ga): return float(np.mean(list(ga.values())))

    # ---- baseline: worst-group accuracy and cue reliance (the RESULT line) ----
    ga = group_acc(model, tok, gev, args.seed + 1, device)
    cd = CF.cue_delta(model, tok, gev, args.seed + 3, device)
    wg = worst(ga); adv = (ga[(1, 0)] + ga[(0, 1)]) / 2
    print(f"RESULT rho={args.rho} seed={args.seed} worstgroup={wg:.3f} advgroup={adv:.3f} cueDelta={cd:.3f}", flush=True)

    layers = get_layers(model)
    nl = model.config.num_hidden_layers
    dirs = identity_directions(model, tok, gtr, device)          # train comments only

    # ---- PROBE: per-layer decodability of identity presence ----
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
    print(f"PROBE_RESULT rho={args.rho} seed={args.seed} maxAUROC={max(aucs):.4f} argmaxL={int(np.argmax(aucs))}", flush=True)

    # ---- PROSPECT: choose the ablation layer from TRAIN data only ----
    ptexts = list(rng.choice(gtr[(1, 1)] + gtr[(0, 1)], size=args.n_prospect, replace=False))
    cd_train_base = cue_delta_batched(model, tok, ptexts, device)
    curve = []
    for L in range(nl):
        h = layers[L].register_forward_hook(make_projection_hook(dirs[L + 1]))
        try:
            curve.append(cue_delta_batched(model, tok, ptexts, device))
        finally:
            h.remove()
    Lstar = int(np.argmin(curve))
    print("PROSPECT train-cueDelta base={:.3f} curve ".format(cd_train_base)
          + " ".join(f"{c:.3f}" for c in curve), flush=True)

    ga_p = ablate_eval(model, tok, gev, Lstar, dirs[Lstar + 1], args.seed + 1, device)
    print(f"PROSPECT_RESULT rho={args.rho} seed={args.seed} Lstar={Lstar} "
          f"worst={worst(ga_p):.3f} overall={overall(ga_p):.3f} "
          f"delta_worst={worst(ga_p)-wg:+.3f}", flush=True)

    # ---- FIXED: fixed relative depth (layer 20 of 24), scaled for deeper models ----
    LFIX = round(20 * nl / 24)
    ga_20 = ablate_eval(model, tok, gev, LFIX, dirs[LFIX + 1], args.seed + 1, device)
    print(f"FIXED rho={args.rho} seed={args.seed} L={LFIX} worst={worst(ga_20):.3f} "
          f"overall={overall(ga_20):.3f} delta_worst={worst(ga_20)-wg:+.3f}", flush=True)

    # ---- LEACE baseline at the same layer (concept = identity presence, train resid) ----
    mu, A = leace_eraser(Rtr[:, LFIX + 1], ytr)
    # sanity: probe on erased train representations should fall to chance
    Xer = Rtr[:, LFIX + 1] - (Rtr[:, LFIX + 1] - mu) @ A.T
    clf = LogisticRegression(max_iter=2000, C=0.1).fit(Xer, ytr)
    auc_er = float(roc_auc_score(ytr, clf.decision_function(Xer)))
    h = layers[LFIX].register_forward_hook(make_leace_hook(mu, A, device))
    try:
        ga_le = group_acc(model, tok, gev, args.seed + 1, device)
    finally:
        h.remove()
    print(f"LEACE rho={args.rho} seed={args.seed} L={LFIX} trainAUROC_after={auc_er:.3f} "
          f"worst={worst(ga_le):.3f} overall={overall(ga_le):.3f} "
          f"delta_worst={worst(ga_le)-wg:+.3f}", flush=True)
    print("V2 DONE", flush=True)


if __name__ == "__main__":
    main()
