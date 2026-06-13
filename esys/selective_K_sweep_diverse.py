"""K-sweep on the 8 DIVERSE domains under reasoning: minimal K for field+selective@K to recover.

Same question as selective_K_sweep.py but on the diverse_tasks suite (retail_refund, airline_cancel,
deploy_guard, bank_withdraw, doc_access, rx_safety, customs_route, oncall_route) with a 2-way
correct/stale decision after a CoT (behavioral: which action word the model states after </think>).
Run: MECH_ATTN=eager python esys/selective_K_sweep_diverse.py --model Qwen/Qwen3-8B
"""
import argparse, os, sys, json
import torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2"))
from mech_suite import load, clone, prefill, step, wilson
from align import align_pair
import diverse_tasks as DT

KLIST = [0, 4, 8, 16, 32, 64]


def chat(tok, s, think=True):
    try:
        return tok.apply_chat_template([{"role": "user", "content": s}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=think)
    except TypeError:
        return tok.apply_chat_template([{"role": "user", "content": s}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def dattn_rank(model, co, last, dpos, a):
    att = torch.stack([x[0, :, -1, :] for x in step(model, clone(co, dpos), last, dpos).attentions]).mean(1).mean(0)
    return sorted(range(a, dpos), key=lambda i: float(att[i]), reverse=True)


def patched(co, cn, positions, L):
    w = clone(co, L)
    if positions:
        p = torch.tensor(positions, device="cuda")
        for i in range(len(w.layers)):
            w.layers[i].keys[:, :, p, :] = cn.layers[i].keys[:, :, p, :]
            w.layers[i].values[:, :, p, :] = cn.layers[i].values[:, :, p, :]
    return w


@torch.no_grad()
def cot_decide(model, tok, cache0, nid, L, seed, correct, stale, max_new, temp=0.7):
    """Generate a CoT then read the stated decision (which action word appears first after </think>)."""
    g = torch.Generator(device="cuda"); g.manual_seed(seed)
    cache = clone(cache0, L - 1); cur = int(nid[0, L - 1]); pos = L - 1; gen = []
    eos = tok.eos_token_id
    for _ in range(max_new):
        out = step(model, cache, cur, pos); pos += 1
        p = torch.softmax(out.logits[0, -1].float() / temp, -1)
        nx = int(torch.multinomial(p, 1, generator=g)); gen.append(nx); cur = nx
        if "</think>" in tok.decode(gen[-16:]) or nx == eos:
            break
    # post-CoT: greedily decode the answer, find which action word is stated first
    ans = []
    for _ in range(16):
        out = step(model, cache, cur, pos); pos += 1
        nx = int(out.logits[0, -1].argmax()); ans.append(nx); cur = nx
        if nx == eos:
            break
    txt = tok.decode(ans).lower()
    ci = txt.find(correct.lower()); si = txt.find(stale.lower())
    if ci >= 0 and (si < 0 or ci < si):
        return "correct"
    if si >= 0:
        return "stale"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B"); ap.add_argument("--tag", default=None)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--max_new", type=int, default=320)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok, model = load(args.model)
    METH = [f"field+sel@{k}" for k in KLIST] + ["full", "stale"]
    corr = {m: 0 for m in METH}; n = 0
    for d, t in DT.TASKS.items():
        correct, stale = t["correct"], t["stale"]
        al = align_pair(tok, chat(tok, DT.build(d, t["vold"])), chat(tok, DT.build(d, t["vnew"])))
        oid, nid = al["old_ids"], al["new_ids"]; a, b = al["field_span"]
        L = oid.shape[1]; dpos = L - 1
        co = prefill(model, oid); cn = prefill(model, nid)
        order = dattn_rank(model, co, int(oid[0, dpos]), dpos, a)
        fld = list(range(a, b)); extra = [p for p in order if p not in range(a, b)]
        for s in range(args.K):
            for k in KLIST:
                pc = patched(co, cn, fld + extra[:k], L)
                corr[f"field+sel@{k}"] += (cot_decide(model, tok, pc, nid, L, 11 + s * 5 + k, correct, stale, args.max_new) == "correct")
            corr["full"] += (cot_decide(model, tok, cn, nid, L, 700 + s, correct, stale, args.max_new) == "correct")
            corr["stale"] += (cot_decide(model, tok, co, oid, L, 800 + s, correct, stale, args.max_new) == "correct")
            n += 1
        print(f"  {d} done ({n})", flush=True)
    out = {"model": args.model, "n": n, "K_correct": {}}
    full = corr["full"] / n if n else 0
    print(f"\n==== DIVERSE K-SWEEP (field+selective under REASONING) — {args.model} (n={n}) ====")
    for k in KLIST:
        p = corr[f"field+sel@{k}"] / n if n else 0
        out["K_correct"][k] = {"P_correct": round(p, 3), "ci": wilson(corr[f"field+sel@{k}"], n)}
        print(f"  field+sel@{k:<3d} P_correct={p:.2f} CI{wilson(corr[f'field+sel@{k}'], n)}")
    out["full_P_correct"] = round(full, 3); out["stale_P_correct"] = round(corr["stale"] / n, 3)
    kstar = next((k for k in KLIST if corr[f"field+sel@{k}"] / n >= full - 1e-9), None)
    out["K_star_full"] = kstar
    print(f"  full={full:.2f}  stale={out['stale_P_correct']:.2f}  => K* (>=full) = {kstar}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"ksweep_diverse_{tag}.json"), "w"), indent=2)
    print("DIVERSE_KSWEEP_DONE")


if __name__ == "__main__":
    main()
