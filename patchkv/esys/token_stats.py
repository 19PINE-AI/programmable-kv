"""STATISTICAL token-level interpretability: where does the memoized conclusion reside?

For each model x each of the 8 diverse gating tasks, we compute the per-downstream-position causal
decision-recovery and attribute the total (positive) recovery MASS to interpretable buckets:
  region:  field / session / rule (the gating policy) / filler / decision (TASK..Decision: + scaffold)
  type:    DELIMITER (newline/punctuation/whitespace = "aggregator" tokens) vs CONTENT (word/number)
Aggregated across tasks (and models) with bootstrap 95% CIs -> statistical significance for the claim
that the conclusion lives in the rule + pre-decision AGGREGATOR (delimiter) tokens, not the field.
Run: MECH_ATTN=sdpa python esys/token_stats.py --model Qwen/Qwen3-8B
"""
import argparse, os, sys, json, random, string
import torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from align import align_pair
from mech_suite import load
from transformers.cache_utils import DynamicCache
import diverse_tasks as DT


def boot_ci(xs, B=10000, seed=0):
    n = len(xs)
    if n == 0:
        return [0.0, 0.0]
    rng = random.Random(seed)
    m = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(B))
    return [round(m[int(0.025 * B)], 3), round(m[int(0.975 * B)], 3)]


def clone(c, upto):
    d = DynamicCache()
    for i, l in enumerate(c.layers):
        d.update(l.keys[:, :, :upto, :].clone(), l.values[:, :, :upto, :].clone(), i)
    return d


@torch.no_grad()
def dscore(model, cache, last, pos, tc, ts):
    o = model(input_ids=torch.tensor([[int(last)]], device="cuda"), past_key_values=clone(cache, pos),
              cache_position=torch.tensor([pos], device="cuda"), use_cache=True)
    return float(o.logits[0, -1, tc] - o.logits[0, -1, ts])


@torch.no_grad()
def patch1(model, co, cn, i, dpos, last, tc, ts):
    w = clone(co, dpos)
    for L in range(len(w.layers)):
        w.layers[L].keys[:, :, i, :] = cn.layers[L].keys[:, :, i, :]
        w.layers[L].values[:, :, i, :] = cn.layers[L].values[:, :, i, :]
    o = model(input_ids=torch.tensor([[int(last)]], device="cuda"), past_key_values=w,
              cache_position=torch.tensor([dpos], device="cuda"), use_cache=True)
    return float(o.logits[0, -1, tc] - o.logits[0, -1, ts])


def is_delim(s):
    t = s.strip()
    return (t == "") or all(c in string.punctuation for c in t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok, model = load(args.model)

    def chat(s):
        try:
            return tok.apply_chat_template([{"role": "user", "content": s}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template([{"role": "user", "content": s}], tokenize=False, add_generation_prompt=True)

    REGIONS = ["field", "session", "rule", "filler", "decision"]
    region_mass = {r: [] for r in REGIONS}     # per-task fraction of recovery mass
    delim_mass = {"delimiter": [], "content": []}
    top_token_cat = {}                          # category of the single top-recovery token, counted
    n = 0
    for d, t in DT.TASKS.items():
        tc = tok(t["correct"], add_special_tokens=False)["input_ids"][0]
        ts = tok(t["stale"], add_special_tokens=False)["input_ids"][0]
        oldtxt = chat(DT.build(d, t["vold"]))
        enc = tok(oldtxt, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        al = align_pair(tok, oldtxt, chat(DT.build(d, t["vnew"])))
        oid, nid = al["old_ids"], al["new_ids"]; a, b = al["field_span"]
        L = oid.shape[1]; dpos = L - 1; last = int(nid[0, dpos])
        co = model(input_ids=oid.to("cuda"), use_cache=True).past_key_values
        cn = model(input_ids=nid.to("cuda"), use_cache=True).past_key_values
        s_old = dscore(model, co, int(oid[0, dpos]), dpos, tc, ts)
        s_new = dscore(model, cn, last, dpos, tc, ts)
        if abs(s_new - s_old) < 1e-4:
            print(f"  {d}: non-flipping, skip", flush=True); continue
        # region char anchors
        ra = oldtxt.find(t["rule"][:24]); fa = oldtxt.find("NOTE 1."); ta = oldtxt.find("TASK\n")
        def region(i):
            if a <= i < b:
                return "field"
            c = offs[i][0] if i < len(offs) else len(oldtxt)
            if ta >= 0 and c >= ta:
                return "decision"
            if fa >= 0 and c >= fa:
                return "filler"
            if ra >= 0 and c >= ra:
                return "rule"
            return "session"
        # per-position positive recovery mass
        rmass = {r: 0.0 for r in REGIONS}; dm = {"delimiter": 0.0, "content": 0.0}
        per = {}
        for i in range(a, dpos):
            r = (patch1(model, co, cn, i, dpos, last, tc, ts) - s_old) / (s_new - s_old)
            per[i] = r
            if r > 0:
                rmass[region(i)] += r
                dm["delimiter" if is_delim(tok.decode([int(oid[0, i])])) else "content"] += r
        tot = sum(rmass.values()) or 1e-9
        for r in REGIONS:
            region_mass[r].append(rmass[r] / tot)
        td = dm["delimiter"] + dm["content"] or 1e-9
        delim_mass["delimiter"].append(dm["delimiter"] / td); delim_mass["content"].append(dm["content"] / td)
        topi = max(per, key=per.get); cat = region(topi)
        topdelim = "delim" if is_delim(tok.decode([int(oid[0, topi])])) else "content"
        key = f"{cat}/{topdelim}"; top_token_cat[key] = top_token_cat.get(key, 0) + 1
        print(f"  {d}: top-recovery token={tok.decode([int(oid[0, topi])])!r} ({cat},{topdelim}) | "
              f"mass rule={rmass['rule']/tot:.2f} decision={rmass['decision']/tot:.2f} field={rmass['field']/tot:.2f} "
              f"| delimiter={dm['delimiter']/td:.2f}", flush=True)
        n += 1

    out = {"model": args.model, "n_tasks": n,
           "region_recovery_mass": {r: {"mean": round(sum(region_mass[r]) / n, 3), "ci": boot_ci(region_mass[r])} for r in REGIONS},
           "delimiter_vs_content": {k: {"mean": round(sum(v) / n, 3), "ci": boot_ci(v)} for k, v in delim_mass.items()},
           "top_token_category_counts": top_token_cat}
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"token_stats_{tag}.json"), "w"), indent=2)
    print(f"\n==== TOKEN-LEVEL STATS ({args.model}, n={n} tasks) ====")
    print("  recovery MASS by region:")
    for r in REGIONS:
        print(f"    {r:9s} {out['region_recovery_mass'][r]['mean']:.3f} CI{out['region_recovery_mass'][r]['ci']}")
    print(f"  DELIMITER (aggregator) vs CONTENT recovery mass: "
          f"delimiter={out['delimiter_vs_content']['delimiter']['mean']} CI{out['delimiter_vs_content']['delimiter']['ci']} | "
          f"content={out['delimiter_vs_content']['content']['mean']}")
    print(f"  top-recovery token category counts: {top_token_cat}")
    print("TOKEN_STATS_DONE")


if __name__ == "__main__":
    main()
