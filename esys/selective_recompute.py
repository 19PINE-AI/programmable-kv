"""Profile-guided SELECTIVE recompute: recompute only the most-affected downstream tokens.

User's idea: instead of appending an erratum or full-reprefilling, (1) make the surgical field
edit, (2) find which downstream tokens' representations change most, (3) record those positions
OFFLINE, (4) in production recompute KV only for those positions and leave the rest stale.

We test feasibility on a ~1B model:
  A) RECOVERY vs k for several selection criteria — which CHEAP, decision-agnostic criterion
     selects positions that recover the decision with k << all?
       kv_change   : ||cn.K[i]-co.K[i]|| + ||cn.V[i]-co.V[i]||  (CacheBlend-style; profilable)
       hidden_change: ||h_new[i]-h_old[i]||  (residual-stream change; profilable)
       decision_oracle: per-position causal decision-recovery (EXPENSIVE upper bound, from D1)
       suffix      : just the last-k positions (no profiling)
       random      : random k (floor)
  B) GENERALIZATION — profile the affected-set on (old->new1), apply that FIXED set to a DIFFERENT
     value (old->new2); does it still recover? (the crux of "record offline, reuse in production").
Run: MECH_ATTN=sdpa python esys/selective_recompute.py --model Qwen/Qwen3-1.7B
"""
import argparse, os, sys, json, random
import torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from align import align_pair
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# gating scenarios with MULTIPLE new values (all flip the decision) for the generalization test
SCEN = {
 "account_role": dict(role="You are a retail support agent.",
   rule="POLICY: If account_role is admin, perform refunds. Otherwise you MUST deny and escalate.",
   request="The user asks for a $40 refund.", field="account_role", old="admin",
   news=["suspended", "restricted", "flagged"], correct="escalate", stale="refund"),
 "order_status": dict(role="You are an order management agent.",
   rule="POLICY: An order may be cancelled only if order_status is pending; otherwise deny.",
   request="The user asks to cancel the order.", field="order_status", old="pending",
   news=["processed", "delivered", "shipped"], correct="deny", stale="cancel"),
}
FILLER = "\n".join(f"NOTE {i+1}. Standard operating note; no special handling required." for i in range(16))


def build(s, val):
    body = (f"{s['role']}\n\nSESSION CONTEXT\n{s['field']}: {val}\nchannel: api\n\n{s['rule']}\n{FILLER}\n\n"
            f"TASK\n{s['request']}\nAnswer with exactly one word.\nDecision:")
    return body


def prefill(model, ids, hidden=False):
    out = model(input_ids=ids.to("cuda"), use_cache=True, output_hidden_states=hidden)
    return (out.past_key_values, out.hidden_states) if hidden else out.past_key_values


def clone(c, upto):
    d = DynamicCache()
    for i, l in enumerate(c.layers):
        d.update(l.keys[:, :, :upto, :].clone(), l.values[:, :, :upto, :].clone(), i)
    return d


@torch.no_grad()
def dec_score(model, cache, last, pos, tc, ts):
    out = model(input_ids=torch.tensor([[int(last)]], device="cuda"), past_key_values=clone(cache, pos),
                cache_position=torch.tensor([pos], device="cuda"), use_cache=True)
    lg = out.logits[0, -1].float()
    return float(lg[tc] - lg[ts])


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
    return float(lg[tc] - lg[ts])


@torch.no_grad()
def kv_change_rank(co, cn, a, dpos):
    d = torch.zeros(dpos, device="cuda")
    for lc, ln in zip(co.layers, cn.layers):
        d[:dpos] += (ln.keys[0, :, :dpos] - lc.keys[0, :, :dpos]).norm(dim=(0, 2))
        d[:dpos] += (ln.values[0, :, :dpos] - lc.values[0, :, :dpos]).norm(dim=(0, 2))
    return sorted(range(a, dpos), key=lambda i: float(d[i]), reverse=True), d


def hidden_change_rank(ho, hn, a, dpos):
    H = torch.stack([(hn[l][0, :dpos] - ho[l][0, :dpos]).norm(dim=-1) for l in range(len(ho))]).sum(0)
    return sorted(range(a, dpos), key=lambda i: float(H[i]), reverse=True)


@torch.no_grad()
def decision_attention_rank(model, co, last, dpos, a):
    """Offline-profilable, DECISION-AWARE: which downstream positions the decision token attends to
    (computed on the OLD context). Requires eager attentions."""
    out = model(input_ids=torch.tensor([[int(last)]], device="cuda"), past_key_values=clone(co, dpos),
                cache_position=torch.tensor([dpos], device="cuda"), use_cache=True, output_attentions=True)
    att = torch.stack([x[0] for x in out.attentions])[:, :, -1, :].mean(1).mean(0)  # [dpos+1], mean heads&layers
    return sorted(range(a, dpos), key=lambda i: float(att[i]), reverse=True)


def setup(model, tok, C, s, newval):
    tc = tok(s["correct"], add_special_tokens=False)["input_ids"][0]
    ts = tok(s["stale"], add_special_tokens=False)["input_ids"][0]
    al = align_pair(tok, C(build(s, s["old"])), C(build(s, newval)))
    oid, nid = al["old_ids"], al["new_ids"]; a, b = al["field_span"]
    L = oid.shape[1]; dpos = L - 1; last = int(nid[0, dpos])
    co, ho = prefill(model, oid, hidden=True); cn, hn = prefill(model, nid, hidden=True)
    s_old = dec_score(model, co, int(oid[0, dpos]), dpos, tc, ts)
    s_new = dec_score(model, cn, last, dpos, tc, ts)
    return dict(co=co, cn=cn, ho=ho, hn=hn, a=a, b=b, dpos=dpos, last=last, tc=tc, ts=ts,
                s_old=s_old, s_new=s_new, denom=(s_new - s_old))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--tag", default="qwen3_1p7b")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="eager", trust_remote_code=True).eval()
    def C(s):
        try:
            return tok.apply_chat_template([{"role": "user", "content": s}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template([{"role": "user", "content": s}], tokenize=False, add_generation_prompt=True)

    rng = random.Random(0)
    KS = [1, 2, 4, 8, 16, 32, 64]
    crit_names = ["kv_change", "hidden_change", "decision_attention", "decision_oracle", "suffix", "random"]
    agg = {c: {k: [] for k in KS} for c in crit_names}
    gen = {"profiled_on_other": [], "own_set": [], "ks": []}

    for name, s in SCEN.items():
        st = setup(model, tok, C, s, s["news"][0])
        if abs(st["denom"]) < 1e-4:
            print(f"  {name}: non-flipping, skip", flush=True); continue
        co, cn, dpos, last, tc, ts = st["co"], st["cn"], st["dpos"], st["last"], st["tc"], st["ts"]
        a = st["a"]; ndown = dpos - a
        def rec(P):
            return (patch_score(model, co, cn, P, dpos, last, tc, ts) - st["s_old"]) / st["denom"]
        # rankings
        kv_order, _ = kv_change_rank(co, cn, a, dpos)
        hid_order = hidden_change_rank(st["ho"], st["hn"], a, dpos)
        att_order = decision_attention_rank(model, co, last, dpos, a)
        # decision-oracle: per-position recovery (expensive)
        per = {i: rec([i]) for i in range(a, dpos)}
        dec_order = sorted(range(a, dpos), key=lambda i: per[i], reverse=True)
        suffix_order = list(range(dpos - 1, a - 1, -1))
        rand_order = list(range(a, dpos)); rng.shuffle(rand_order)
        orders = {"kv_change": kv_order, "hidden_change": hid_order, "decision_oracle": dec_order,
                  "decision_attention": att_order, "suffix": suffix_order, "random": rand_order}
        for c, order in orders.items():
            for k in KS:
                if k <= len(order):
                    agg[c][k].append(rec(order[:k]))
        print(f"  {name}: ndown={ndown} | kv@16={rec(kv_order[:16]):.2f} hidden@16={rec(hid_order[:16]):.2f} "
              f"oracle@16={rec(dec_order[:16]):.2f} suffix@16={rec(suffix_order[:16]):.2f}", flush=True)

        # GENERALIZATION: profile kv_change set on news[0]; apply that FIXED set to news[1]
        st2 = setup(model, tok, C, s, s["news"][1])
        if abs(st2["denom"]) > 1e-4:
            co2, cn2, dpos2, last2 = st2["co"], st2["cn"], st2["dpos"], st2["last"]
            # the profiled set (positions) from news[0], clipped to news[1] length
            def rec2(P):
                P = [p for p in P if p < dpos2]
                return (patch_score(model, co2, cn2, P, dpos2, last2, st2["tc"], st2["ts"]) - st2["s_old"]) / st2["denom"]
            own_order = decision_attention_rank(model, co2, last2, dpos2, st2["a"])
            for k in [8, 16, 32]:
                gen["profiled_on_other"].append(rec2(att_order[:k]))   # decision-attention set (base context, value-independent)
                gen["own_set"].append(rec2(own_order[:k]))            # decision-attention set from news[1]'s own base context
                gen["ks"].append(k)
            print(f"    generalization {name}: profiled-on-{s['news'][0]} set applied to {s['news'][1]}: "
                  f"@16={rec2(kv_order[:16]):.2f} vs own@16={rec2(own_order[:16]):.2f}", flush=True)

    out = {"model": args.model, "recovery_vs_k": {}, "generalization": {}}
    print(f"\n==== SELECTIVE RECOMPUTE ({args.model}) ====")
    print(f"  recovery vs #tokens recomputed (mean over scenarios):")
    print(f"  {'k':>4}  " + "  ".join(f"{c:>14}" for c in crit_names))
    for k in KS:
        row = {}
        for c in crit_names:
            vs = agg[c][k]; row[c] = round(sum(vs) / len(vs), 2) if vs else None
        out["recovery_vs_k"][k] = row
        print(f"  {k:>4}  " + "  ".join(f"{(row[c] if row[c] is not None else 0):>14.2f}" for c in crit_names))
    for k in sorted(set(gen["ks"])):
        po = [v for v, kk in zip(gen["profiled_on_other"], gen["ks"]) if kk == k]
        ow = [v for v, kk in zip(gen["own_set"], gen["ks"]) if kk == k]
        out["generalization"][k] = {"profiled_on_other": round(sum(po) / len(po), 3),
                                    "own_set": round(sum(ow) / len(ow), 3)}
    print(f"  GENERALIZATION (offline set from a DIFFERENT value): {out['generalization']}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"selective_recompute_{args.tag}.json"), "w"), indent=2)
    print("SELECTIVE_RECOMPUTE_DONE")


if __name__ == "__main__":
    main()
