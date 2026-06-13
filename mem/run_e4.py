"""E4 — edit granularity / sub-chunking.

Split the memory chunk into S independently-precompiled blocks, RoPE-reposition each to its
slot, splice. Tests the cost/fidelity trade-off: a localized edit recomputes only one block
(cost ~ L_mem/S), but splitting cuts inter-block attention (fidelity may drop with S).
Endpoints vs full recompute (S=1 reference == full transplant): cos, top1_agree, dec_agree;
plus localized-edit cost in tokens. Writes results/e4_<tag>.jsonl.
"""
import os, sys, json, argparse, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from data import make_dataset, filler_trajectory
from memkv import build_prompt, run_full, decide, EARLY, LATE, _decision_logits_from_cache
from composable_kv import (load_lm, prefill, precompute_chunk, repositioned_chunk_cache,
                           cache_concat, cache_slice, forward_suffix)
from transformers import AutoTokenizer

SYS = "You are a careful account-management assistant. Follow the user settings exactly."


@torch.no_grad()
def subchunk_transplant(model, tok, ids, mem_lo, mem_hi, S):
    """Split [mem_lo,mem_hi) into S blocks, each precomputed in isolation and repositioned."""
    L = ids.shape[1]; nb = mem_hi - mem_lo
    bounds = [mem_lo + round(j * nb / S) for j in range(S + 1)]
    pre = prefill(model, ids[:, :mem_lo])
    cache = pre
    for j in range(S):
        lo, hi = bounds[j], bounds[j + 1]
        if hi <= lo:
            continue
        blk = precompute_chunk(model, ids[:, lo:hi])           # isolation (positions 0..)
        rep = repositioned_chunk_cache(model, blk, hi - lo, lo)
        cache = cache_concat(cache, rep)
    if mem_hi < L - 1:
        cache = forward_suffix(model, cache, ids[:, mem_hi:L - 1], mem_hi).past_key_values
    logits = _decision_logits_from_cache(model, cache, int(ids[0, L - 1]), L - 1)
    edit_cost = (bounds[1] - bounds[0]) if S > 0 else nb       # recompute one (first) block
    return logits, edit_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--nfacts", type=int, default=4)
    ap.add_argument("--mtotal", type=int, default=60)
    ap.add_argument("--S", default="1,2,4,8,16")
    ap.add_argument("--traj_turns", type=int, default=4)
    ap.add_argument("--layout", default="spread")   # spread | contiguous : relevant-fact placement
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lm(args.model, attn="sdpa")
    Ss = [int(x) for x in args.S.split(",")]
    contig = {"spread": False, "contiguous": True}.get(args.layout, None)
    ds = make_dataset(args.n, args.mtotal, args.nfacts, seed0=9000, contiguous=contig)
    path = os.path.join(os.path.dirname(__file__), "results", f"e4_{tag}.jsonl")
    f = open(path, "w"); t0 = time.time()
    for k, p in enumerate(ds):
        traj = filler_trajectory(args.traj_turns, p.pid)
        ids, mlo, mhi, qlo = build_prompt(tok, SYS, p.memory_markdown(), traj,
                                          p.decision_query(False), LATE)
        Lmem = mhi - mlo
        fl = run_full(model, tok, ids); f_arg = int(fl.argmax()); f_dec = decide(fl, tok)
        for S in Ss:
            sl, edit_cost = subchunk_transplant(model, tok, ids, mlo, mhi, S)
            f.write(json.dumps(dict(model=args.model, persona=p.pid, S=S, n_facts=args.nfacts,
                     mtotal=args.mtotal, layout=args.layout, L_mem=int(Lmem), top1_agree=int(int(sl.argmax()) == f_arg),
                     cos=float(F.cosine_similarity(fl, sl, 0)), dec_agree=int(decide(sl, tok) == f_dec),
                     edit_cost_tok=int(edit_cost), full_cost_tok=int(Lmem))) + "\n")
        f.flush()
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(ds)} ({time.time()-t0:.0f}s)", flush=True)
    f.close()
    print(f"E4_DONE {args.model} -> {path}")


if __name__ == "__main__":
    main()
