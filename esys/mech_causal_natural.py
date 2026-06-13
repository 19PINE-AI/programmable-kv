"""Mechanism battery (D1 causal patching) on the 8 NATURAL diverse-domain tasks — breadth.

The §5.1 memoization map was measured on 3 templated scenarios (n=12). Here we re-run the causal
KV-patching on the 8 diverse, naturalistic domains (retail/airline/devops/banking/access/clinical/
customs/oncall), deployment-realistic chat template, with the correct-vs-stale action token as the
decision. We report field-only recovery (= in_place) and the suffix/prefix recovery curves per
domain and aggregated (bootstrap 95% CI), to show the mechanism holds across natural tasks.
Run: MECH_ATTN=sdpa python esys/mech_causal_natural.py
"""
import argparse, os, sys, json, random
import torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from align import align_pair
import diverse_tasks as DT
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def boot_ci(xs, B=10000, seed=0):
    n = len(xs)
    if n == 0:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(B))
    return [round(means[int(0.025 * B)], 3), round(means[int(0.975 * B)], 3)]


def prefill(model, ids):
    return model(input_ids=ids.to("cuda"), use_cache=True).past_key_values


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
    return float(lg[tc] - lg[ts])


@torch.no_grad()
def patched_score(model, co, cn, positions, dpos, last, tc, ts):
    w = clone(co, dpos)
    pos = torch.tensor(positions, device="cuda")
    for i in range(len(w.layers)):
        w.layers[i].keys[:, :, pos, :] = cn.layers[i].keys[:, :, pos, :]
        w.layers[i].values[:, :, pos, :] = cn.layers[i].values[:, :, pos, :]
    out = model(input_ids=torch.tensor([[int(last)]], device="cuda"), past_key_values=w,
                cache_position=torch.tensor([dpos], device="cuda"), use_cache=True)
    lg = out.logits[0, -1].float()
    return float(lg[tc] - lg[ts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa", trust_remote_code=True).eval()
    def C(s):
        try:
            return tok.apply_chat_template([{"role": "user", "content": s}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template([{"role": "user", "content": s}], tokenize=False, add_generation_prompt=True)

    fracs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    per = {}
    fo_all = []; suf = {f: [] for f in fracs}; pre = {f: [] for f in fracs}
    for d, t in DT.TASKS.items():
        tc = tok(t["correct"], add_special_tokens=False)["input_ids"][0]
        ts = tok(t["stale"], add_special_tokens=False)["input_ids"][0]
        al = align_pair(tok, C(DT.build(d, t["vold"])), C(DT.build(d, t["vnew"])))
        oid, nid = al["old_ids"], al["new_ids"]; a, b = al["field_span"]
        L = oid.shape[1]; dpos = L - 1; last = int(nid[0, dpos])
        co = prefill(model, oid); cn = prefill(model, nid)
        s_old = score(model, co, int(oid[0, dpos]), dpos, tc, ts)
        s_new = score(model, cn, last, dpos, tc, ts)
        denom = s_new - s_old
        if abs(denom) < 1e-4:
            print(f"  {d}: non-flipping, skip", flush=True); continue
        def rec(P):
            return (patched_score(model, co, cn, P, dpos, last, tc, ts) - s_old) / denom
        fo = rec(list(range(a, b)))                       # field-only = in_place
        full = rec(list(range(a, dpos)))                  # sanity ~1.0
        ndown = dpos - a
        sufd, pred = {}, {}
        for f in fracs:
            k = max(1, int(round(f * ndown)))
            sufd[f] = rec(list(range(dpos - k, dpos)))     # suffix
            pred[f] = rec(list(range(a, a + k)))           # prefix
        per[d] = {"field_only": round(fo, 3), "full_downstream": round(full, 2),
                  "suffix": {f: round(sufd[f], 3) for f in fracs}}
        fo_all.append(fo)
        for f in fracs:
            suf[f].append(sufd[f]); pre[f].append(pred[f])
        print(f"  {d:14s} field_only={fo:.3f} full={full:.2f} suffix@0.1={sufd[0.1]:.2f} suffix@0.2={sufd[0.2]:.2f}", flush=True)

    agg = {"model": args.model, "n_domains": len(fo_all),
           "field_only_recovery": {"mean": round(sum(fo_all) / len(fo_all), 3), "ci": boot_ci(fo_all)},
           "suffix_recovery": {f: {"mean": round(sum(suf[f]) / len(suf[f]), 3), "ci": boot_ci(suf[f])} for f in fracs},
           "prefix_recovery": {f: round(sum(pre[f]) / len(pre[f]), 3) for f in fracs}}
    json.dump({"agg": agg, "per_domain": per}, open(os.path.join(os.path.dirname(__file__), "..",
              "results", f"mech_causal_natural_{args.tag}.json"), "w"), indent=2)
    print(f"\n==== D1 ON NATURAL TASKS ({len(fo_all)} domains, {args.model}) ====")
    print(f"  field-only recovery (= in_place): {agg['field_only_recovery']['mean']} CI{agg['field_only_recovery']['ci']}")
    print(f"  suffix recovery: @0.1={agg['suffix_recovery'][0.1]['mean']} @0.2={agg['suffix_recovery'][0.2]['mean']} "
          f"@0.5={agg['suffix_recovery'][0.5]['mean']} | prefix@0.5={agg['prefix_recovery'][0.5]}")
    print("MECH_CAUSAL_NATURAL_DONE")


if __name__ == "__main__":
    main()
