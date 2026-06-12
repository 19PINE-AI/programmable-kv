"""EXP4 - Specificity: are the notes on SPECIFIC aggregator tokens, or diffuse?

Dose-response shows recovery grows with the number of patched tokens, but that alone
is consistent with a diffuse code ("any downstream tokens would do"). The discriminating
control: at matched count k, patch the TOP-k positions (ranked by individual causal
recovery) vs k RANDOM downstream positions. If the conclusion is carried by specific
aggregator/delimiter tokens, top-k >> random-k. If diffuse, the two curves coincide.

We use the original gated task (mech_suite.build, field flip OLD->NEW) and the patching
recovery metric (transplant NEW's KV into OLD at a position set).

Run: MECH_ATTN=sdpa python esys/mechd_specificity.py --model Qwen/Qwen3-8B --tag qwen3_8b
"""
import argparse, json, os, sys
import torch
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mech_suite import load, clone, prefill, ftok, build, step, META, TOK_WORDS, make_instances
from align import align_pair


def boot_ci(xs, B=2000):
    n = len(xs)
    if n == 0: return (0.0, 0.0)
    if n == 1: return (round(xs[0], 3), round(xs[0], 3))
    means = [sum(xs[(bsi * 2654435761 + j * 40503) % n] for j in range(n)) / n for bsi in range(B)]
    means.sort()
    return (round(means[int(0.025 * B)], 3), round(means[int(0.975 * B)], 3))


@torch.no_grad()
def score(model, cache, last, dpos, toi):
    lg = step(model, clone(cache, dpos), last, dpos).logits[0, -1].float()
    return float(lg[toi["safe"]] - lg[toi["unsafe"]])


@torch.no_grad()
def patched_score(model, co, cn, positions, last, dpos, toi):
    w = clone(co, dpos)
    pos = torch.tensor(positions, device=w.layers[0].keys.device)
    for i in range(len(w.layers)):
        w.layers[i].keys[:, :, pos, :] = cn.layers[i].keys[:, :, pos, :]
        w.layers[i].values[:, :, pos, :] = cn.layers[i].values[:, :, pos, :]
    lg = step(model, w, last, dpos).logits[0, -1].float()
    return float(lg[toi["safe"]] - lg[toi["unsafe"]])


def run(model, tok, scn, oid):
    m = META[scn]
    toi = {"safe": ftok(tok, TOK_WORDS[m["safe"]]), "unsafe": ftok(tok, TOK_WORDS[m["unsafe"]])}
    t_old = build(tok, scn, oid, m["vold"], False, True)
    t_new = build(tok, scn, oid, m["vnew"], False, True)
    al = align_pair(tok, t_old, t_new)
    oid_ids, nid_ids = al["old_ids"], al["new_ids"]
    a, b = al["field_span"]; L = oid_ids.shape[1]; dpos = L - 1
    last = int(nid_ids[0, dpos])
    co = prefill(model, oid_ids); cn = prefill(model, nid_ids)
    s_old = score(model, co, last, dpos, toi); s_new = score(model, cn, last, dpos, toi)
    denom = s_new - s_old
    if abs(denom) < 0.5:
        return None
    def rec(positions):
        return (patched_score(model, co, cn, positions, last, dpos, toi) - s_old) / denom

    down = list(range(b, dpos))                     # post-field downstream positions
    # rank by individual single-position recovery
    indiv = {p: rec([p]) for p in down}
    ranked = sorted(down, key=lambda p: indiv[p], reverse=True)
    rng = np.random.default_rng(1234 + hash((scn, oid)) % 9999)
    ks = [1, 2, 4, 8, 16, 32, 64]
    topk, randk = {}, {}
    for k in ks:
        if k > len(down):
            continue
        topk[k] = rec(ranked[:k])
        # average random-k over a few draws for stability
        rs = []
        for _ in range(5):
            sel = rng.choice(len(down), size=k, replace=False)
            rs.append(rec([down[i] for i in sel]))
        randk[k] = float(np.mean(rs))
    return {"scn": scn, "oid": oid, "n_down": len(down),
            "topk": {k: round(v, 3) for k, v in topk.items()},
            "randk": {k: round(v, 3) for k, v in randk.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--max_instances", type=int, default=9)
    args = ap.parse_args()
    tok, model = load(args.model)
    recs = []
    for scn, oid in make_instances()[:args.max_instances]:
        r = run(model, tok, scn, oid)
        if r is None:
            print(f"  [{scn}/{oid}] non-flipping, skipped", flush=True); continue
        recs.append(r)
        print(f"  [{scn}/{oid}] top@8={r['topk'].get(8)} rand@8={r['randk'].get(8)} "
              f"top@16={r['topk'].get(16)} rand@16={r['randk'].get(16)}", flush=True)
    ks = [1, 2, 4, 8, 16, 32, 64]
    agg = {"n": len(recs), "topk": {}, "randk": {}, "gap": {}}
    for k in ks:
        tv = [r["topk"][k] for r in recs if k in r["topk"]]
        rv = [r["randk"][k] for r in recs if k in r["randk"]]
        if tv and rv:
            agg["topk"][k] = {"mean": round(np.mean(tv), 3), "ci": boot_ci(tv)}
            agg["randk"][k] = {"mean": round(np.mean(rv), 3), "ci": boot_ci(rv)}
            agg["gap"][k] = round(np.mean(tv) - np.mean(rv), 3)
    out = {"model": args.model, "agg": agg, "instances": recs}
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"mechd_specificity_{args.tag}.json"), "w"), indent=2)
    print("\n==== EXP4 SPECIFICITY (top-k vs random-k downstream, matched count) ====")
    print(f"n={agg['n']} instances")
    print(f"{'k':>4} | {'top-k recovery':>20} | {'random-k recovery':>20} | gap")
    for k in ks:
        if k in agg["topk"]:
            t = agg["topk"][k]; r = agg["randk"][k]
            print(f"{k:>4} | {t['mean']:>7.3f} CI{str(t['ci']):>12} | "
                  f"{r['mean']:>7.3f} CI{str(r['ci']):>12} | {agg['gap'][k]:+.3f}")
    print("MECHD_SPECIFICITY_DONE", flush=True)


if __name__ == "__main__":
    main()
