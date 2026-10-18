"""
Cue directions and directional ablation, used by the repair runs.

The cue direction at each layer is the mean change in the last-position
residual stream when the identity terms are edited out (identity_directions).
ablate_eval projects that direction out at one layer and measures the four
groups again. Running this file directly sweeps the ablation over layers for
one model, which is exploratory and not used in the paper.
"""
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import common as CF
from common import (local_model_path, prompt, neutralize, load_groups, make_train,
                      build_ids, group_acc, cue_delta, MODELS)


def get_layers(model):
    m = model.base_model.model if hasattr(model, "base_model") else model
    if hasattr(m, "model") and hasattr(m.model, "layers"):   # Llama/Qwen
        return m.model.layers
    if hasattr(m, "gpt_neox"):                               # GPT-NeoX / Pythia
        return m.gpt_neox.layers
    raise ValueError("unknown architecture")


@torch.no_grad()
def identity_directions(model, tok, g, device, n=400):
    """Unit shortcut direction at each residual layer, from the cueDelta interchange."""
    rng = np.random.default_rng(0)
    texts = list(rng.choice(g[(1, 1)] + g[(0, 1)], size=n, replace=False))
    nl = model.config.num_hidden_layers
    acc = [None] * (nl + 1)
    for t in texts:
        e1 = tok(prompt(t), return_tensors="pt", truncation=True, max_length=96).to(device)
        e2 = tok(prompt(neutralize(t)), return_tensors="pt", truncation=True, max_length=96).to(device)
        h1 = model(**e1, output_hidden_states=True).hidden_states
        h2 = model(**e2, output_hidden_states=True).hidden_states
        for L in range(nl + 1):
            d = (h1[L][0, -1] - h2[L][0, -1]).float()
            acc[L] = d if acc[L] is None else acc[L] + d
    dirs = []
    for L in range(nl + 1):
        v = acc[L] / n
        dirs.append((v / (v.norm() + 1e-8)).to(device))
    return dirs   # dirs[L] aligns with hidden_states[L] = output of decoder layer L-1


def ablate_eval(model, tok, g, layer_idx, dhat, seed, device):
    """project dhat out of decoder layer `layer_idx` output (all positions), then eval groups."""
    layers = get_layers(model)
    def hook(mod, inp, out):
        istuple = isinstance(out, tuple)
        h = out[0] if istuple else out
        h = h - (h @ dhat).unsqueeze(-1) * dhat
        return ((h,) + tuple(out[1:])) if istuple else h
    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        ga = group_acc(model, tok, g, seed, device)
    finally:
        handle.remove()
    return ga


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.95)   # a shortcut-reliant model
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate_layers", default=None, help="comma list, e.g. 18,20,22; default=sweep all")
    ap.add_argument("--model", default="qwen", choices=list(MODELS))
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MODELS[args.model]
    mp = local_model_path(cfg["subdir"])
    tok = AutoTokenizer.from_pretrained(mp)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    CF.A_ID = tok.encode(" A")[-1]; CF.B_ID = tok.encode(" B")[-1]   # per-model label tokens
    model = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.float32).to(device)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=cfg["targets"]))
    gtr, gev = load_groups()                       # DISJOINT train/eval split (no contamination)
    ids, lab, am = build_ids(tok, make_train(gtr, args.rho, args.n_train, args.seed))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    n = ids.shape[0]; spe = n // args.batch
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n)
        for it in range(spe):
            idx = perm[it*args.batch:(it+1)*args.batch]
            o = model(input_ids=ids[idx].to(device), attention_mask=am[idx].to(device), labels=lab[idx].to(device))
            opt.zero_grad(); o.loss.backward(); opt.step()
    model.eval()
    base = group_acc(model, tok, gev, args.seed+1, device)     # all eval/diagnosis on held-out
    def worst(ga): return min(ga.values())
    def overall(ga): return float(np.mean(list(ga.values())))
    cd = cue_delta(model, tok, gev, args.seed+3, device)
    print(f"BASELINE rho={args.rho} cueDelta={cd:.3f} worst={worst(base):.3f} overall={overall(base):.3f} "
          f"(TI={base[(1,1)]:.2f} TN={base[(1,0)]:.2f} NI={base[(0,1)]:.2f} NN={base[(0,0)]:.2f})", flush=True)
    dirs = identity_directions(model, tok, gtr, device)        # derive direction from TRAIN comments...
    nl = model.config.num_hidden_layers
    print(f"  {'ablate_L':>9}{'worst':>8}{'overall':>9}", flush=True)
    best = (None, worst(base), None)
    cand = [int(x) for x in args.ablate_layers.split(",")] if args.ablate_layers else range(nl)
    for li in cand:
        ga = ablate_eval(model, tok, gev, li, dirs[li + 1], args.seed+1, device)
        print(f"  {li:>9}{worst(ga):>8.3f}{overall(ga):>9.3f}", flush=True)
        if worst(ga) > best[1]:
            best = (li, worst(ga), overall(ga))
    print(f"RESULT rho={args.rho} cueDelta={cd:.3f} baseline_worst={worst(base):.3f} "
          f"best_ablate_L={best[0]} fixed_worst={best[1]:.3f} fixed_overall={best[2]:.3f} "
          f"delta_worst={best[1]-worst(base):+.3f}", flush=True)


if __name__ == "__main__":
    main()
