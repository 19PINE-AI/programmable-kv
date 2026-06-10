"""E2 — transplant faithfulness / equivalence.

For each persona x placement x seam: precompiled+RoPE-repositioned memory vs full recompute.
Endpoints (no generation, discriminating regardless of task competence):
  * top1_agree : argmax(full_logits) == argmax(transplant_logits)  (full vocab)
  * cos        : cosine(full_logits, transplant_logits)
  * dec_agree  : yes/no decision agreement (governance proxy)
Also a NO-ROTATION naive control (keys keep position-0 RoPE) to show the reposition matters.

Writes one JSONL record per (persona, placement, method) to results/e2_<tag>.jsonl.
Stats are computed later by analyze.py (TOST equivalence, cluster bootstrap).
"""
import os, sys, json, argparse, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from data import make_dataset, filler_trajectory
from memkv import (build_prompt, run_full, run_transplant, decide, EARLY, LATE,
                   precompute_chunk, repositioned_chunk_cache, cache_concat, cache_slice,
                   forward_suffix, _decision_logits_from_cache, prefill)
from composable_kv import load_lm
from transformers import AutoTokenizer
import torch.nn.functional as F

SYS = "You are a careful account-management assistant. Follow the user settings exactly."


@torch.no_grad()
def naive_no_rotation(model, tok, ids, mem_lo, mem_hi):
    """Splice isolation memory KV WITHOUT re-rotation (control)."""
    from transformers.cache_utils import DynamicCache
    L = ids.shape[1]; nb = mem_hi - mem_lo
    alone = precompute_chunk(model, ids[:, mem_lo:mem_hi])
    pre = prefill(model, ids[:, :mem_lo])
    naive = DynamicCache()
    for i, l in enumerate(alone.layers):
        naive.update(l.keys.clone(), l.values.clone(), i)
    cache = cache_concat(pre, naive)
    if mem_hi < L - 1:
        cache = forward_suffix(model, cache, ids[:, mem_hi:L - 1], mem_hi).past_key_values
    return _decision_logits_from_cache(model, cache, int(ids[0, L - 1]), L - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--mtotal", type=int, default=24)
    ap.add_argument("--nfacts", type=int, default=2)
    ap.add_argument("--seams", default="0,1,2,4,8")
    ap.add_argument("--traj_turns", type=int, default=4)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lm(args.model, attn="sdpa")
    seams = [int(x) for x in args.seams.split(",")]
    ds = make_dataset(args.n, args.mtotal, args.nfacts, seed0=5000)
    path = os.path.join(os.path.dirname(__file__), "results", f"e2_{tag}.jsonl")
    f = open(path, "w")
    t0 = time.time()
    for k, p in enumerate(ds):
        traj = filler_trajectory(args.traj_turns, p.pid)
        mem = p.memory_markdown(); q = p.decision_query(False)
        gold = "yes" if p.gold_yes else "no"
        for placement in (EARLY, LATE):
            ids, mlo, mhi, qlo = build_prompt(tok, SYS, mem, traj, q, placement)
            Lmem = mhi - mlo
            fl = run_full(model, tok, ids)
            f_arg = int(fl.argmax()); f_dec = decide(fl, tok)
            # transplant at each seam (mem precomputed once, reused)
            mem_alone = None
            for s in seams:
                tl, mem_alone = run_transplant(model, tok, ids, mlo, mhi, seam=s, mem_alone=mem_alone)
                rec = dict(model=args.model, persona=p.pid, placement=placement, method=f"seam{s}",
                           n_facts=args.nfacts, L_total=int(ids.shape[1]), L_mem=int(Lmem),
                           gold=gold, top1_agree=int(int(tl.argmax()) == f_arg),
                           cos=float(F.cosine_similarity(fl, tl, 0)),
                           dec_agree=int(decide(tl, tok) == f_dec), dec=decide(tl, tok), dec_full=f_dec)
                f.write(json.dumps(rec) + "\n")
            # naive control
            nl = naive_no_rotation(model, tok, ids, mlo, mhi)
            f.write(json.dumps(dict(model=args.model, persona=p.pid, placement=placement, method="naive",
                     n_facts=args.nfacts, L_total=int(ids.shape[1]), L_mem=int(Lmem), gold=gold,
                     top1_agree=int(int(nl.argmax()) == f_arg), cos=float(F.cosine_similarity(fl, nl, 0)),
                     dec_agree=int(decide(nl, tok) == f_dec), dec=decide(nl, tok), dec_full=f_dec)) + "\n")
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(ds)} ({time.time()-t0:.0f}s)", flush=True)
    f.close()
    print(f"E2_DONE {args.model} -> {path}")


if __name__ == "__main__":
    main()
