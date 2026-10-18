"""
Shared code for the CivilComments runs.

Defines the models (MODELS), the prompt, the identity-term edit (neutralize),
the four evaluation groups, training sets at an assigned cue-label correlation
rho (make_train), worst-group accuracy (group_acc) and cue reliance
(cue_delta). The train_*.py scripts import it. Running it directly trains and
evaluates one model with the same training loop.
"""
import argparse, os, glob, json, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

POOL = os.environ.get("CIVIL_POOL", "data/civil_pool.json")
A_ID, B_ID = 362, 425   # " A"=toxic, " B"=not (Qwen default; overridden per-model in main)
MODELS = {
    "qwen":   dict(subdir="models--Qwen--Qwen2.5-0.5B",
                   targets=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]),
    "pythia": dict(subdir="models--EleutherAI--pythia-410m",
                   targets=["query_key_value","dense","dense_h_to_4h","dense_4h_to_h"]),
    "qwen15": dict(subdir="models--Qwen--Qwen2.5-1.5B",
                   targets=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]),
    "pythia14": dict(subdir="models--EleutherAI--pythia-1.4b",
                   targets=["query_key_value","dense","dense_h_to_4h","dense_4h_to_h"]),
    "qwen_it": dict(subdir="models--Qwen--Qwen2.5-0.5B-Instruct",
                   targets=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]),
    "olmo1": dict(subdir="models--allenai--OLMo-2-0425-1B",
                   targets=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]),
}
IDENTITY = ["muslim","islam","islamic","christian","christianity","jewish","jew","catholic",
            "hindu","buddhist","atheist","mormon","gay","lesbian","homosexual","bisexual",
            "transgender","queer","lgbt","lgbtq","latino","latina","hispanic","asian",
            "mexican","arab","immigrant","feminist","muslims","christians","jews","gays"]
IDRE = re.compile(r"\b(" + "|".join(IDENTITY) + r")\b", re.IGNORECASE)


def local_model_path(subdir="models--Qwen--Qwen2.5-0.5B"):
    base = os.path.join(os.environ.get("HF_HOME", "hf_cache"),
                        "hub", subdir, "snapshots")
    s = sorted(glob.glob(base + "/*")); return s[-1] if s else subdir


def prompt(text):
    return f"Comment: {text}\nQuestion: Is this comment toxic?\nAnswer:"


def neutralize(text):
    return IDRE.sub("people", text)


def load_groups(split=0.8):
    """Return (g_train, g_eval): DISJOINT 80/20 partition of each group so evaluation never
    sees a training comment (the pool is streamed-order ≈ random, so first-80/last-20 is a
    valid random split). keys (toxic, identity)."""
    pool = json.load(open(POOL))
    g = {(1,1):[],(1,0):[],(0,1):[],(0,0):[]}
    for x in pool: g[(x["toxic"], x["identity"])].append(x["text"])
    gtr, gev = {}, {}
    for k, v in g.items():
        v = list(dict.fromkeys(v))                 # dedupe (preserve order) so no text spans both splits
        cut = int(len(v) * split); gtr[k] = v[:cut]; gev[k] = v[cut:]
    return gtr, gev


def make_train(g, rho, n, seed):
    """identity is a rho-reliable shortcut: P(toxic|identity)=rho, P(toxic|no-id)=1-rho; label balanced."""
    rng = np.random.default_rng(seed)
    def draw(key, k): return list(rng.choice(g[key], size=k, replace=len(g[key]) < k))
    half = n // 2
    ti = int(round(half * rho)); ni = half - ti           # identity: rho toxic, (1-rho) nontoxic
    tn = half - ti; nn = ti                                # no-id: (1-rho) toxic, rho nontoxic
    items = [(t,1) for t in draw((1,1),ti)] + [(t,0) for t in draw((0,1),ni)] \
          + [(t,1) for t in draw((1,0),tn)] + [(t,0) for t in draw((0,0),nn)]
    rng.shuffle(items); return items


def build_ids(tok, items, max_len=96):
    n = len(items)
    ids = torch.full((n, max_len), tok.pad_token_id, dtype=torch.long)
    lab = torch.full((n, max_len), -100, dtype=torch.long)
    am = torch.zeros((n, max_len), dtype=torch.long)
    for i,(text,tox) in enumerate(items):
        full = tok(prompt(text) + (" A" if tox else " B"))["input_ids"][:max_len]
        ids[i,:len(full)] = torch.tensor(full); am[i,:len(full)] = 1; lab[i,len(full)-1] = full[-1]
    return ids, lab, am


@torch.no_grad()
def group_acc(model, tok, g, seed, device, per=400, bs=64):
    rng = np.random.default_rng(seed); out = {}
    for key in [(1,1),(1,0),(0,1),(0,0)]:
        texts = list(rng.choice(g[key], size=min(per,len(g[key])), replace=False))
        hit = 0
        for s in range(0,len(texts),bs):
            ch = texts[s:s+bs]
            enc = tok([prompt(t) for t in ch], return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
            lg = model(**enc).logits[torch.arange(len(ch)), enc["attention_mask"].sum(1)-1]
            pred = (lg[:,A_ID] > lg[:,B_ID]).cpu().numpy().astype(int)   # 1=toxic
            hit += int((pred == key[0]).sum())
        out[key] = hit/len(texts)
    return out


@torch.no_grad()
def cue_delta(model, tok, g, seed, device, n=300):
    """identity-interchange: original vs identity-neutralized; logit-diff(A,B) shift = shortcut reliance."""
    rng = np.random.default_rng(seed)
    texts = list(rng.choice(g[(1,1)]+g[(0,1)], size=n, replace=False))   # identity-present comments
    def ld(t):
        enc = tok(prompt(t), return_tensors="pt", truncation=True, max_length=96).to(device)
        lg = model(**enc).logits[0,-1]; return float(lg[A_ID]-lg[B_ID])
    return float(np.mean([ld(t) - ld(neutralize(t)) for t in texts]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.8)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="qwen", choices=list(MODELS))
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MODELS[args.model]
    mp = local_model_path(cfg["subdir"])
    tok = AutoTokenizer.from_pretrained(mp)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    global A_ID, B_ID
    A_ID = tok.encode(" A")[-1]; B_ID = tok.encode(" B")[-1]   # per-model label tokens
    model = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.float32).to(device)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=cfg["targets"]))
    model.print_trainable_parameters()
    gtr, gev = load_groups()                       # DISJOINT train/eval split (no contamination)
    ids, lab, am = build_ids(tok, make_train(gtr, args.rho, args.n_train, args.seed))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    n = ids.shape[0]; spe = n // args.batch
    print(f"rho={args.rho} seed={args.seed} device={device}", flush=True)
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n); run = 0.0
        for it in range(spe):
            idx = perm[it*args.batch:(it+1)*args.batch]
            o = model(input_ids=ids[idx].to(device), attention_mask=am[idx].to(device), labels=lab[idx].to(device))
            opt.zero_grad(); o.loss.backward(); opt.step(); run += o.loss.item()
        model.eval()
        ga = group_acc(model, tok, gev, args.seed+1, device)   # eval on HELD-OUT split
        wg = min(ga.values()); adv = (ga[(1,0)]+ga[(0,1)])/2   # shortcut-adversarial groups
        print(f"ep{ep+1}/{args.epochs} loss={run/spe:.3f} groups TI={ga[(1,1)]:.2f} TN={ga[(1,0)]:.2f} NI={ga[(0,1)]:.2f} NN={ga[(0,0)]:.2f} worst={wg:.2f} adv={adv:.2f}", flush=True)
    cd = cue_delta(model, tok, gev, args.seed+3, device)        # interchange on held-out split
    print(f"RESULT rho={args.rho} seed={args.seed} worstgroup={wg:.3f} advgroup={adv:.3f} cueDelta={cd:.3f}", flush=True)


if __name__ == "__main__":
    main()
