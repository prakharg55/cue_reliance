"""
Measures cue reliance after every training epoch (Appendix E.2).

Uses the same training as train_civil_repairs.py and prints cue reliance and a
quick worst-group estimate after each epoch. Its final RESULT line matches the
other scripts for the same model, rho and seed.

    python training/train_civil_epochs.py --model qwen --rho 0.97 --seed 0
"""
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

import common as CF
from common import (local_model_path, prompt, neutralize, load_groups,
                      make_train, build_ids, group_acc, MODELS)
from train_civil_repairs import batched_logit_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.8)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="qwen", choices=list(MODELS))
    ap.add_argument("--n_epoch_cd", type=int, default=150)
    args = ap.parse_args()

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

    # fixed measurement samples (drawn once, before training; separate generators)
    rtr = np.random.default_rng(args.seed + 101)
    rev = np.random.default_rng(args.seed + 103)
    tr_texts = list(rtr.choice(gtr[(1, 1)] + gtr[(0, 1)], size=args.n_epoch_cd, replace=False))
    ev_texts = list(rev.choice(gev[(1, 1)] + gev[(0, 1)], size=args.n_epoch_cd, replace=False))

    last_loss = [float('nan')]

    def measure(ep):
        model.eval()
        with torch.no_grad():
            cd_tr = float(np.mean(batched_logit_diff(model, tok, tr_texts, device)
                                  - batched_logit_diff(model, tok, [neutralize(t) for t in tr_texts], device)))
            cd_ev = float(np.mean(batched_logit_diff(model, tok, ev_texts, device)
                                  - batched_logit_diff(model, tok, [neutralize(t) for t in ev_texts], device)))
            ga = group_acc(model, tok, gev, args.seed + 1, device, per=150)
        overall = float(np.mean(list(ga.values())))
        # worst-CLASS accuracy: group-label-free selection criterion (Yang et al.,
        # ICML 2023) -- the baseline cueDelta has to beat to be worth anything.
        worstcls = min((ga[(c, 0)] + ga[(c, 1)]) / 2 for c in (0, 1))
        print(f"EPOCH rho={args.rho} seed={args.seed} epoch={ep} cueDeltaTrain={cd_tr:.3f} "
              f"cueDeltaEval={cd_ev:.3f} worstSmall={min(ga.values()):.3f} "
              f"overall={overall:.3f} worstClass={worstcls:.3f} trainLoss={last_loss[0]:.4f}", flush=True)
        model.train()

    print(f"V3E rho={args.rho} seed={args.seed} model={args.model} device={device}", flush=True)
    measure(0)
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n); ep_loss = 0.0
        for it in range(spe):
            idx = perm[it*args.batch:(it+1)*args.batch]
            o = model(input_ids=ids[idx].to(device), attention_mask=am[idx].to(device), labels=lab[idx].to(device))
            opt.zero_grad(); o.loss.backward(); opt.step()
            ep_loss += float(o.loss.item())
        last_loss[0] = ep_loss / max(spe, 1)
        measure(ep + 1)
    model.eval()

    ga = group_acc(model, tok, gev, args.seed + 1, device)
    cd = CF.cue_delta(model, tok, gev, args.seed + 3, device)
    print(f"RESULT rho={args.rho} seed={args.seed} worstgroup={min(ga.values()):.3f} "
          f"advgroup={(ga[(1,0)]+ga[(0,1)])/2:.3f} cueDelta={cd:.3f}", flush=True)
    print("V3E DONE", flush=True)


if __name__ == "__main__":
    main()
