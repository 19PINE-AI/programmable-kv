"""Selective-recompute (decision-attention) vs the GOLDEN erratum — sweep over k, models, benchmarks.

For each model and each of the 8 diverse gating benchmarks, we compare decision recovery / P(correct)
for: stale (floor), full reprefill (=oracle), the ERRATUM (append-at-end; the golden method), and
SELECTIVE recompute of the top-k downstream tokens ranked by DECISION-ATTENTION (cheap, profilable on
the base context), for k in {4,8,16,32,64}. Reports, per method, P(correct decision) over the tasks
where the oracle is correct, and the mean recovery ratio. Use MECH_ATTN=eager (decision attentions).
Run: MECH_ATTN=eager python esys/selective_sweep.py --model Qwen/Qwen3-8B
"""
import argparse, os, sys, json
import torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2"))
from align import align_pair
from mech_suite import load          # handles gemma-3 text-only, quantized, eager
import diverse_tasks as DT
from transformers.cache_utils import DynamicCache

KS = [4, 8, 16, 32, 64]
ERR = "[STATE UPDATE] {f} has changed to {v}; this overrides any earlier value AND any earlier conclusion.\n\n"


def clone(c, upto):
    d = DynamicCache()
    for i, l in enumerate(c.layers):
        d.update(l.keys[:, :, :upto, :].clone(), l.values[:, :, :upto, :].clone(), i)
    return d


@torch.no_grad()
def score(model, cache, last, pos, tc, ts):
    out = model(input_ids=torch.tensor([[int(last)]], device="cuda"), past_key_values=clone(cache, pos),
                cache_position=torch.tensor([pos], device="cuda"), use_cache=True)
    lg = out.logits[0, -1].float()
    return float(lg[tc] - lg[ts]), ("correct" if lg[tc] >= lg[ts] else "stale")


@torch.no_grad()
def score_full(model, tok, text, tc, ts):
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
    lg = model(input_ids=ids, use_cache=False).logits[0, -1].float()
    return float(lg[tc] - lg[ts]), ("correct" if lg[tc] >= lg[ts] else "stale")


@torch.no_grad()
def patch_score(model, co, cn, positions, dpos, last, tc, ts):
    w = clone(co, dpos)
    if positions:
        p = torch.tensor(positions, device="cuda")
        for i in range(len(w.layers)):
            w.layers[i].keys[:, :, p, :] = cn.layers[i].keys[:, :, p, :]
            w.layers[i].values[:, :, p, :] = cn.layers[i].values[:, :, p, :]
    out = model(input_ids=torch.tensor([[int(last)]], device="cuda"), past_key_values=w,
                cache_position=torch.tensor([dpos], device="cuda"), use_cache=True)
    lg = out.logits[0, -1].float()
    return float(lg[tc] - lg[ts]), ("correct" if lg[tc] >= lg[ts] else "stale")


@torch.no_grad()
def decision_attention_rank(model, co, last, dpos, a):
    out = model(input_ids=torch.tensor([[int(last)]], device="cuda"), past_key_values=clone(co, dpos),
                cache_position=torch.tensor([dpos], device="cuda"), use_cache=True, output_attentions=True)
    att = torch.stack([x[0] for x in out.attentions])[:, :, -1, :].mean(1).mean(0)
    return sorted(range(a, dpos), key=lambda i: float(att[i]), reverse=True)


def chat(tok, s, thinking=False):
    try:
        return tok.apply_chat_template([{"role": "user", "content": s}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=thinking)
    except TypeError:
        return tok.apply_chat_template([{"role": "user", "content": s}], tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok, model = load(args.model)
    methods = ["stale", "full", "erratum"] + [f"selective@{k}" for k in KS]
    corr = {m: 0 for m in methods}; rec = {m: [] for m in methods}; n = 0
    for d, t in DT.TASKS.items():
        tc = tok(t["correct"], add_special_tokens=False)["input_ids"][0]
        ts = tok(t["stale"], add_special_tokens=False)["input_ids"][0]
        al = align_pair(tok, chat(tok, DT.build(d, t["vold"])), chat(tok, DT.build(d, t["vnew"])))
        oid, nid = al["old_ids"], al["new_ids"]; a, b = al["field_span"]
        L = oid.shape[1]; dpos = L - 1; last = int(nid[0, dpos])
        co = model(input_ids=oid.to("cuda"), use_cache=True).past_key_values
        cn = model(input_ids=nid.to("cuda"), use_cache=True).past_key_values
        s_old, _ = score(model, co, int(oid[0, dpos]), dpos, tc, ts)
        s_new, full_dec = score(model, cn, last, dpos, tc, ts)
        if full_dec != "correct" or abs(s_new - s_old) < 1e-4:
            print(f"  {d}: oracle not correct / non-flipping, skip", flush=True); continue
        denom = s_new - s_old
        def R(s):
            return (s - s_old) / denom
        # methods
        rec["full"].append(R(s_new)); corr["full"] += 1
        rec["stale"].append(R(s_old)); corr["stale"] += (s_old >= 0 and tc and False) or 0  # stale is the floor
        ss, sd = score(model, co, int(oid[0, dpos]), dpos, tc, ts); corr["stale"] += (sd == "correct"); rec["stale"][-1] = R(ss)
        # erratum (append at end, golden)
        er_text = chat(tok, DT.build(d, t["vold"], erratum_value=t["vnew"]))
        es, ed = score_full(model, tok, er_text, tc, ts)
        rec["erratum"].append(R(es)); corr["erratum"] += (ed == "correct")
        # selective@k by decision-attention
        order = decision_attention_rank(model, co, last, dpos, a)
        for k in KS:
            ps, pd = patch_score(model, co, cn, order[:k], dpos, last, tc, ts)
            rec[f"selective@{k}"].append(R(ps)); corr[f"selective@{k}"] += (pd == "correct")
        n += 1
        print(f"  {d}: erratum={ed} | " + " ".join(f"sel@{k}={patch_score(model,co,cn,order[:k],dpos,last,tc,ts)[1][:1]}" for k in [8,32]), flush=True)

    out = {"model": args.model, "n_tasks": n, "methods": {}}
    print(f"\n==== SELECTIVE vs ERRATUM ({args.model}, n={n} benchmarks) ====")
    print(f"  {'method':14s} {'P_correct':>10s} {'mean_recovery':>14s}")
    for m in methods:
        pc = corr[m] / n if n else 0; mr = sum(rec[m]) / len(rec[m]) if rec[m] else 0
        out["methods"][m] = {"P_correct": round(pc, 3), "mean_recovery": round(mr, 3)}
        print(f"  {m:14s} {pc:>10.2f} {mr:>14.2f}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"selective_sweep_{tag}.json"), "w"), indent=2)
    print("SELECTIVE_SWEEP_DONE")


if __name__ == "__main__":
    main()
