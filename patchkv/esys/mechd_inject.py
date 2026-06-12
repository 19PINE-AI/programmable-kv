"""EXP5 - Counterfactual note injection: writing a FALSE conclusion.

If the cache is a notebook of memoized conclusions, we should be able to *write a false
entry* and have the model act on it. We start from a fully consistent prefill whose LIVE
field implies conclusion C, then overwrite ONLY the downstream notes' KV with the notes
from a context whose conclusion is C' (the opposite). The field token itself, and the
whole prefix before it, are left untouched and still imply C.

  base = NEW prefill (live field -> SAFE, self-consistent)
  inject OLD context's downstream notes (-> UNSAFE) at downstream positions
  -> does the decision follow the INJECTED note (UNSAFE, false vs live field) or the
     LIVE field (SAFE)?

We report, both directions:
  - recovery_toward_injected: continuous (1 = fully adopts injected conclusion)
  - P(decision follows injected note): categorical argmax flip rate
  - dose-response: inject the top-k note positions; how few suffice to flip the belief
This turns "notebook you can write to" from metaphor into a measured capability and ties
the mechanism back to the editing result (editing = overwriting this same note).

Run: MECH_ATTN=sdpa python esys/mechd_inject.py --model Qwen/Qwen3-8B --tag qwen3_8b
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
def margins(model, cache, last, dpos, toi):
    lg = step(model, clone(cache, dpos), last, dpos).logits[0, -1].float()
    return float(lg[toi["safe"]] - lg[toi["unsafe"]])


@torch.no_grad()
def inject_margin(model, base_cache, src_cache, positions, last, dpos, toi):
    """Clone base; overwrite (K,V) at `positions` from src (the injected notes)."""
    w = clone(base_cache, dpos)
    pos = torch.tensor(positions, device=w.layers[0].keys.device)
    for i in range(len(w.layers)):
        w.layers[i].keys[:, :, pos, :] = src_cache.layers[i].keys[:, :, pos, :]
        w.layers[i].values[:, :, pos, :] = src_cache.layers[i].values[:, :, pos, :]
    lg = step(model, w, last, dpos).logits[0, -1].float()
    return float(lg[toi["safe"]] - lg[toi["unsafe"]])


def run_dir(model, tok, scn, oid, base_is_new):
    """base_is_new=True: live field=NEW(SAFE), inject OLD notes(UNSAFE)."""
    m = META[scn]
    toi = {"safe": ftok(tok, TOK_WORDS[m["safe"]]), "unsafe": ftok(tok, TOK_WORDS[m["unsafe"]])}
    t_old = build(tok, scn, oid, m["vold"], False, True)
    t_new = build(tok, scn, oid, m["vnew"], False, True)
    al = align_pair(tok, t_old, t_new)
    oid_ids, nid_ids = al["old_ids"], al["new_ids"]
    a, b = al["field_span"]; L = oid_ids.shape[1]; dpos = L - 1
    co = prefill(model, oid_ids); cn = prefill(model, nid_ids)
    last = int(nid_ids[0, dpos])      # identical suffix token both sides
    base_cache, src_cache = (cn, co) if base_is_new else (co, cn)
    s_base = margins(model, base_cache, last, dpos, toi)
    s_src = margins(model, src_cache, last, dpos, toi)
    denom = s_src - s_base
    if abs(denom) < 0.5:
        return None
    down = list(range(b, dpos))
    # rank downstream note positions by individual injection effect
    indiv = {p: inject_margin(model, base_cache, src_cache, [p], last, dpos, toi) for p in down}
    ranked = sorted(down, key=lambda p: abs(indiv[p] - s_base), reverse=True)
    full = inject_margin(model, base_cache, src_cache, down, last, dpos, toi)
    rec_full = (full - s_base) / denom
    # categorical: did the decision flip to the injected conclusion?
    base_safe = s_base > 0
    inj_safe = full > 0
    flipped = (base_safe != inj_safe)
    follows_injected = (inj_safe == (s_src > 0))
    # dose-response: top-k note injection
    dose = {}
    for k in [1, 2, 4, 8, 16, 32]:
        if k <= len(down):
            mk = inject_margin(model, base_cache, src_cache, ranked[:k], last, dpos, toi)
            dose[k] = {"recovery": round((mk - s_base) / denom, 3),
                       "follows_injected": bool((mk > 0) == (s_src > 0))}
    return {"scn": scn, "oid": oid, "dir": ("new<-old" if base_is_new else "old<-new"),
            "s_base": round(s_base, 2), "s_src": round(s_src, 2),
            "full_recovery": round(rec_full, 3), "flipped": bool(flipped),
            "follows_injected": bool(follows_injected),
            "dose": dose, "n_down": len(down)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--max_instances", type=int, default=9)
    args = ap.parse_args()
    tok, model = load(args.model)
    recs = []
    for scn, oid in make_instances()[:args.max_instances]:
        for bin_ in (True, False):
            r = run_dir(model, tok, scn, oid, bin_)
            if r is None:
                continue
            recs.append(r)
            print(f"  [{scn}/{oid} {r['dir']}] full_recovery={r['full_recovery']:+.3f} "
                  f"flipped={r['flipped']} follows_injected={r['follows_injected']} "
                  f"dose@4={r['dose'].get(4,{}).get('recovery')} dose@16={r['dose'].get(16,{}).get('recovery')}",
                  flush=True)
    rf = [r["full_recovery"] for r in recs]
    flip_rate = np.mean([r["flipped"] for r in recs])
    follow_rate = np.mean([r["follows_injected"] for r in recs])
    dose_agg = {}
    for k in [1, 2, 4, 8, 16, 32]:
        vs = [r["dose"][k]["recovery"] for r in recs if k in r["dose"]]
        fs = [r["dose"][k]["follows_injected"] for r in recs if k in r["dose"]]
        if vs:
            dose_agg[k] = {"recovery_mean": round(np.mean(vs), 3), "ci": boot_ci(vs),
                           "follow_rate": round(float(np.mean(fs)), 3)}
    agg = {"n": len(recs),
           "full_recovery": {"mean": round(np.mean(rf), 3), "ci": boot_ci(rf)},
           "flip_rate": round(float(flip_rate), 3),
           "follows_injected_rate": round(float(follow_rate), 3),
           "dose": dose_agg}
    out = {"model": args.model, "agg": agg, "instances": recs}
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"mechd_inject_{args.tag}.json"), "w"), indent=2)
    print("\n==== EXP5 COUNTERFACTUAL NOTE INJECTION ====")
    print(f"n={agg['n']} (both directions)")
    print(f"  full-note injection recovery toward FALSE conclusion: {agg['full_recovery']['mean']} "
          f"CI{agg['full_recovery']['ci']}")
    print(f"  decision flip rate:           {agg['flip_rate']}")
    print(f"  follows-injected-note rate:   {agg['follows_injected_rate']}  "
          f"(decision adopts the written note, against the live field)")
    print("  DOSE (inject top-k note positions):")
    for k, d in dose_agg.items():
        print(f"     k={k:>3}: recovery={d['recovery_mean']:+.3f} CI{d['ci']}  follow_rate={d['follow_rate']}")
    print("MECHD_INJECT_DONE", flush=True)


if __name__ == "__main__":
    main()
