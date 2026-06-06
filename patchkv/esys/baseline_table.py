"""Head-to-head baseline table: correctness x cost for every KV-update strategy.

Answers the reviewer's "why not just hoist to end?" and positions vs prior selective-recompute
work (CacheBlend). On 8 diverse gating tasks (non-reasoning — the regime where strategies differ;
reasoning is covered in §5e), for each strategy we report P(correct decision) and the recompute
fraction (tokens recomputed / total), plus a POISONED-context column (a stale self-conclusion
asserting the old value) where the erratum's "overrides any earlier conclusion" clause matters.

Strategies:
  full_reprefill  : recompute everything (oracle ceiling)         cost 1.00
  stale           : reuse everything, no edit (floor)             cost 0.00
  in_place        : recompute only the field's KV                 cost ~field/L
  cacheblend@15%  : recompute field + top-15% highest-KV-deviation downstream (prior work)  cost ~0.15
  hoist_to_end    : field PLACED at the end; recompute field+decision suffix (needs prompt rewrite)
  erratum         : leave stale; append [STATE UPDATE ...]; recompute that span + decision
  field+erratum   : in_place AND erratum
Run: MECH_ATTN=sdpa python esys/baseline_table.py
"""
import argparse, os, sys, json
import torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from align import align_pair
import diverse_tasks as DT
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

FILLER = "\n".join(f"NOTE {i+1}. Standard operating note; no special handling." for i in range(12))
ERR = "[STATE UPDATE] {f} has changed to {v}; this overrides any earlier value AND any earlier conclusion.\n"
POISON = "\nPRIOR NOTE (resolved): based on {f}={ov}, the request was approved; proceed as concluded.\n"


def tnl(role, field, val, rule, request, where="early", erratum=None, poison=None):
    sess = f"SESSION\n{field}: {val}\n" if where == "early" else "SESSION\n(see current value below)\n"
    tail_field = f"CURRENT {field}: {val}\n" if where == "end" else ""
    p = (poison or "")
    e = (erratum or "")
    return (f"{role}\n\n{sess}{p}\n{rule}\n{FILLER}\n\n{e}TASK\n{request}\n{tail_field}Decision:")


def prefill(model, ids):
    return model(input_ids=ids.to("cuda"), use_cache=True).past_key_values


def clone(c, upto):
    d = DynamicCache()
    for i, l in enumerate(c.layers):
        d.update(l.keys[:, :, :upto, :].clone(), l.values[:, :, :upto, :].clone(), i)
    return d


@torch.no_grad()
def decide_from(model, cache, last, pos, tc, ts):
    out = model(input_ids=torch.tensor([[int(last)]], device="cuda"), past_key_values=cache,
                cache_position=torch.tensor([pos], device="cuda"), use_cache=True)
    lg = out.logits[0, -1].float()
    return "correct" if lg[tc] >= lg[ts] else "stale"


@torch.no_grad()
def decide_full(model, tok, text, tc, ts):
    ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]])
    out = model(input_ids=ids.to("cuda"), use_cache=False)
    lg = out.logits[0, -1].float()
    return "correct" if lg[tc] >= lg[ts] else "stale", ids.shape[1]


@torch.no_grad()
def kv_dev_rank(co, cn, a, dpos):
    """Rank downstream positions [a,dpos) by total KV deviation cn-co (CacheBlend selection)."""
    devs = torch.zeros(dpos, device="cuda")
    for lc, ln in zip(co.layers, cn.layers):
        devs[:dpos] += (ln.keys[0, :, :dpos] - lc.keys[0, :, :dpos]).norm(dim=(0, 2))
        devs[:dpos] += (ln.values[0, :, :dpos] - lc.values[0, :, :dpos]).norm(dim=(0, 2))
    order = sorted(range(a, dpos), key=lambda i: float(devs[i]), reverse=True)
    return order


@torch.no_grad()
def patch_decide(model, co, cn, positions, dpos, last, tc, ts):
    w = clone(co, dpos)
    pos = torch.tensor(positions, device="cuda")
    for i in range(len(w.layers)):
        w.layers[i].keys[:, :, pos, :] = cn.layers[i].keys[:, :, pos, :]
        w.layers[i].values[:, :, pos, :] = cn.layers[i].values[:, :, pos, :]
    return decide_from(model, w, last, dpos, tc, ts)


@torch.no_grad()
def append_decide(model, tok, base_ids, dpos, insert_text, tc, ts):
    """Erratum-style: reuse cache of base_ids[:dpos] (stale prefix up to the decision cue), then
    forward [insert_text + the decision cue token] and decide. Cost = inserted tokens."""
    co = prefill(model, base_ids[:, :dpos])
    ins = tok(insert_text, add_special_tokens=False)["input_ids"]
    seq = ins + [int(base_ids[0, dpos])]            # inserted text then the 'Decision:'-ending token
    ids = torch.tensor([seq], device="cuda")
    out = model(input_ids=ids[:, :-1], past_key_values=co,
                cache_position=torch.arange(dpos, dpos + ids.shape[1] - 1, device="cuda"), use_cache=True)
    cache = out.past_key_values; pos = dpos + ids.shape[1] - 1
    return decide_from(model, cache, int(ids[0, -1]), pos, tc, ts), len(ins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--cb_frac", type=float, default=0.15)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa", trust_remote_code=True).eval()
    METHODS = ["full_reprefill", "stale", "in_place", f"cacheblend@{int(args.cb_frac*100)}%",
               "hoist_to_end", "erratum", "field+erratum"]
    agg = {m: {"correct": 0, "cost": []} for m in METHODS}
    agg_poison = {m: {"correct": 0} for m in ["full_reprefill", "hoist_to_end", "erratum", "field+erratum"]}
    n = 0
    for d, t in DT.TASKS.items():
        role = t["role"]; field = t["field"]; ov, nv = t["vold"], t["vnew"]
        rule = t["rule"]; req = t["request"]
        tc = tok(t["correct"], add_special_tokens=False)["input_ids"][0]
        ts = tok(t["stale"], add_special_tokens=False)["input_ids"][0]
        # aligned old/new natural sequences (for patch-based methods)
        t_old = tnl(role, field, ov, rule, req, "early"); t_new = tnl(role, field, nv, rule, req, "early")
        al = align_pair(tok, t_old, t_new); oid, nid = al["old_ids"], al["new_ids"]; a, b = al["field_span"]
        L = oid.shape[1]; dpos = L - 1; last_new = int(nid[0, dpos])
        co = prefill(model, oid); cn = prefill(model, nid)
        # methods
        res = {}
        res["full_reprefill"] = (decide_from(model, clone(cn, dpos), last_new, dpos, tc, ts), 1.0)
        res["stale"] = (decide_from(model, clone(co, dpos), int(oid[0, dpos]), dpos, tc, ts), 0.0)
        res["in_place"] = (patch_decide(model, co, cn, list(range(a, b)), dpos, last_new, tc, ts), (b - a) / L)
        ndown = dpos - b; k = max(1, int(args.cb_frac * ndown))
        cb_pos = list(range(a, b)) + kv_dev_rank(co, cn, b, dpos)[:k]
        res[f"cacheblend@{int(args.cb_frac*100)}%"] = (patch_decide(model, co, cn, cb_pos, dpos, last_new, tc, ts), len(cb_pos) / L)
        # hoist: field at end -> prefix cached, recompute field+decision suffix
        th = tnl(role, field, nv, rule, req, "end")
        hd, hlen = decide_full(model, tok, th, tc, ts)
        # cost of hoist = tokens from where the field block starts to the end (field + 'Decision:')
        hfield_tok = len(tok(f"CURRENT {field}: {nv}\nDecision:", add_special_tokens=False)["input_ids"])
        res["hoist_to_end"] = (hd, hfield_tok / hlen)
        # erratum: append the [STATE UPDATE..] before the decision cue, reuse stale prefix
        erc, erlen = append_decide(model, tok, oid, dpos, ERR.format(f=field, v=nv), tc, ts)
        res["erratum"] = (erc, erlen / L)
        # field+erratum: patch field in co, then append erratum (approx via patch + append on patched cache)
        # measure decision: patch field into co clone, then append erratum + decide
        wco = clone(co, dpos)
        posf = torch.tensor(list(range(a, b)), device="cuda")
        for i in range(len(wco.layers)):
            wco.layers[i].keys[:, :, posf, :] = cn.layers[i].keys[:, :, posf, :]
            wco.layers[i].values[:, :, posf, :] = cn.layers[i].values[:, :, posf, :]
        ins = tok(ERR.format(f=field, v=nv), add_special_tokens=False)["input_ids"]
        ids = torch.tensor([ins + [int(oid[0, dpos])]], device="cuda")
        out = model(input_ids=ids[:, :-1], past_key_values=clone(wco, dpos),
                    cache_position=torch.arange(dpos, dpos + ids.shape[1] - 1, device="cuda"), use_cache=True)
        fe = decide_from(model, out.past_key_values, int(ids[0, -1]), dpos + ids.shape[1] - 1, tc, ts)
        res["field+erratum"] = (fe, ((b - a) + len(ins)) / L)
        for m in METHODS:
            dec, cost = res[m]
            agg[m]["correct"] += (dec == "correct"); agg[m]["cost"].append(cost)
        # POISONED variant: prior note asserts the OLD value -> approve; only the conclusion-override erratum should resist
        poison = POISON.format(f=field, ov=ov)
        tp_full = tnl(role, field, nv, rule, req, "early", poison=poison)
        pf, _ = decide_full(model, tok, tp_full, tc, ts); agg_poison["full_reprefill"]["correct"] += (pf == "correct")
        tp_hoist = tnl(role, field, nv, rule, req, "end", poison=poison)
        ph, _ = decide_full(model, tok, tp_hoist, tc, ts); agg_poison["hoist_to_end"]["correct"] += (ph == "correct")
        tp_err = tnl(role, field, ov, rule, req, "early", poison=poison, erratum=ERR.format(f=field, v=nv))
        pe, _ = decide_full(model, tok, tp_err, tc, ts); agg_poison["erratum"]["correct"] += (pe == "correct")
        agg_poison["field+erratum"]["correct"] += (pe == "correct")
        n += 1
        print(f"  {d}: " + " ".join(f"{m.split('@')[0][:7]}={res[m][0][:1]}" for m in METHODS), flush=True)

    out = {"model": args.model, "n_tasks": n, "methods": {}}
    print(f"\n==== BASELINE TABLE ({args.model}, n={n} tasks, non-reasoning) ====")
    print(f"  {'method':16s} {'P_correct':>10s} {'recompute%':>12s}  {'poison P_correct':>16s}")
    for m in METHODS:
        pc = agg[m]["correct"] / n; cost = sum(agg[m]["cost"]) / n
        pois = (agg_poison[m]["correct"] / n) if m in agg_poison else None
        out["methods"][m] = {"P_correct": round(pc, 3), "recompute_frac": round(cost, 4),
                             "poison_P_correct": round(pois, 3) if pois is not None else None}
        print(f"  {m:16s} {pc:>10.2f} {100*cost:>11.2f}%  {('%.2f'%pois) if pois is not None else '   -':>16s}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"baseline_table_{args.tag}.json"), "w"), indent=2)
    print("BASELINE_TABLE_DONE")


if __name__ == "__main__":
    main()
