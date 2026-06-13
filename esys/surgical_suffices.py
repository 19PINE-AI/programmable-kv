"""WHERE the surgical in_place edit ALONE suffices — no erratum (reasoning vs non-reasoning).

The erratum is the robust fallback (and works by construction). The *strong* result is the cheap
one: when can you just surgically overwrite the field's KV (~0.1% recompute), leave the entire
downstream stale, and STILL get the correct (oracle) decision — with NO erratum? We measure, per
model, P(in_place decision == correct) in non-reasoning vs reasoning mode, against the oracle
(full reprefill) ceiling and the stale floor. Hypothesis (from the §7 mechanism): the surgical
edit alone suffices for REASONING models, because the chain-of-thought re-reads the refreshed
field and re-derives the conclusion — whereas non-reasoning models revert to the stale downstream.

in_place here = fieldonly_cache (refresh ONLY the field-token KV, all downstream KV stale).
Run: MECH_ATTN=sdpa python esys/surgical_suffices.py --model Qwen/Qwen3-8B --K 4
"""
import argparse, os, sys, json
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2"))
from mech_suite import (load, clone, prefill, ftok, wilson, META, TOK_WORDS, build, step,
                        fieldonly_cache, decide, gen_cot_sample)
from align import align_pair


def boot_ci(xs, B=10000, seed=0):
    """Proper bootstrap 95% CI: B resamples WITH REPLACEMENT (fixed seed -> reproducible)."""
    import random
    n = len(xs)
    if n == 0:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(B))
    return [round(means[int(0.025 * B)], 3), round(means[int(0.975 * B)], 3)]


@torch.no_grad()
def nonreasoning(model, tok, scn, oid):
    m = META[scn]
    toi = {"safe": ftok(tok, TOK_WORDS[m["safe"]]), "unsafe": ftok(tok, TOK_WORDS[m["unsafe"]]),
           "lookup": ftok(tok, "lookup")}
    t_old = build(tok, scn, oid, m["vold"], False, True)
    t_new = build(tok, scn, oid, m["vnew"], False, True)
    oid_ids = torch.tensor([tok(t_old, add_special_tokens=False)["input_ids"]])
    nid_ids = torch.tensor([tok(t_new, add_special_tokens=False)["input_ids"]])
    al = align_pair(tok, t_old, t_new); a, b = al["field_span"]; L = oid_ids.shape[1]; dpos = L - 1
    fc, cn = fieldonly_cache(model, oid_ids, nid_ids, a, b, L)        # in_place: field new, downstream stale
    co = prefill(model, oid_ids)
    ip = decide(step(model, clone(fc, dpos), int(nid_ids[0, dpos]), dpos).logits[0, -1].float(), toi)
    orc = decide(step(model, clone(cn, dpos), int(nid_ids[0, dpos]), dpos).logits[0, -1].float(), toi)
    stl = decide(step(model, clone(co, dpos), int(oid_ids[0, dpos]), dpos).logits[0, -1].float(), toi)
    return {"in_place": ip, "oracle": orc, "stale": stl}


@torch.no_grad()
def reasoning(model, tok, scn, oid, K, seed0=1000, max_new=420):
    m = META[scn]
    toi = {"safe": ftok(tok, TOK_WORDS[m["safe"]]), "unsafe": ftok(tok, TOK_WORDS[m["unsafe"]]),
           "lookup": ftok(tok, "lookup")}
    t_old = build(tok, scn, oid, m["vold"], True, False)
    t_new = build(tok, scn, oid, m["vnew"], True, False)
    oid_ids = torch.tensor([tok(t_old, add_special_tokens=False)["input_ids"]])
    nid_ids = torch.tensor([tok(t_new, add_special_tokens=False)["input_ids"]])
    al = align_pair(tok, t_old, t_new); a, b = al["field_span"]; L = oid_ids.shape[1]
    fc, cn = fieldonly_cache(model, oid_ids, nid_ids, a, b, L)
    out = {"in_place": [], "oracle": []}
    for s in range(K):
        dq, dpos, cache, _ = gen_cot_sample(model, tok, fc, nid_ids, L, seed0 + s, max_new=max_new)   # in_place + CoT
        out["in_place"].append(decide(step(model, clone(cache, dpos), dq, dpos).logits[0, -1].float(), toi))
        dq2, dpos2, cache2, _ = gen_cot_sample(model, tok, cn, nid_ids, L, seed0 + s, max_new=max_new)  # oracle + CoT
        out["oracle"].append(decide(step(model, clone(cache2, dpos2), dq2, dpos2).logits[0, -1].float(), toi))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--max_new", type=int, default=420)
    ap.add_argument("--oids", default="A4471,B8820")
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok, model = load(args.model)
    scns = list(META.keys()); oids = args.oids.split(",")

    # NON-REASONING (deterministic) — collect per-trial 0/1
    nr = {"in_place": [], "oracle": [], "stale": []}
    for scn in scns:
        for oid in oids:
            r = nonreasoning(model, tok, scn, oid)
            for k in ("in_place", "oracle", "stale"):
                nr[k].append(1 if r[k] == "safe" else 0)
    # REASONING (K stochastic CoT samples)
    rr = {"in_place": [], "oracle": []}
    for scn in scns:
        for oid in oids:
            r = reasoning(model, tok, scn, oid, args.K, max_new=args.max_new)
            rr["in_place"] += [1 if x == "safe" else 0 for x in r["in_place"]]
            rr["oracle"] += [1 if x == "safe" else 0 for x in r["oracle"]]

    def mean(xs):
        return round(sum(xs) / len(xs), 3)
    out = {"model": args.model,
           "non_reasoning": {"n": len(nr["in_place"]),
               "in_place_correct": mean(nr["in_place"]), "in_place_boot_ci": boot_ci(nr["in_place"]),
               "oracle_correct": mean(nr["oracle"]), "stale_correct": mean(nr["stale"])},
           "reasoning": {"n": len(rr["in_place"]),
               "in_place_correct": mean(rr["in_place"]), "in_place_boot_ci": boot_ci(rr["in_place"]),
               "oracle_correct": mean(rr["oracle"])}}
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"surgical_suffices_{tag}.json"), "w"), indent=2)
    nrr, rrr = out["non_reasoning"], out["reasoning"]
    print(f"\n==== SURGICAL in_place SUFFICES? — {args.model} (bootstrap CI, B=10000) ====")
    print(f"  NON-REASONING (n={nrr['n']}):  in_place {nrr['in_place_correct']} CI{nrr['in_place_boot_ci']}  "
          f"| oracle {nrr['oracle_correct']} stale {nrr['stale_correct']}")
    print(f"  REASONING     (n={rrr['n']}):  in_place {rrr['in_place_correct']} CI{rrr['in_place_boot_ci']}  "
          f"| oracle {rrr['oracle_correct']}")
    print("SURGICAL_SUFFICES_DONE")


if __name__ == "__main__":
    main()
