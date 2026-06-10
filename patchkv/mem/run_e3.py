"""E3 — editing memory mid-session.

A relevant setting is toggled mid-session (gold decision flips). Each edit method rebuilds the
late-layout decision; we compare to the full-recompute oracle (new memory) under CoT.
Endpoints per (persona, method): pred (CoT), correct (vs new gold), agree_oracle, recompute_tok.
Methods: stale, in_place, erratum, recompile_chunk, selective@4, selective@16, full_recompute.

Run across model sizes for the stickiness/scale law. Writes results/e3_<tag>.jsonl.
"""
import os, sys, json, argparse, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from data import make_dataset, filler_trajectory
from memkv import run_edit_late, generate_from_cache, parse_final, decide, EARLY, LATE
from composable_kv import load_lm
from transformers import AutoTokenizer

SYS = "You are a careful account-management assistant. Follow the user settings exactly."
METHODS = ["full_recompute", "stale", "in_place", "erratum", "recompile_chunk", "selective@4", "selective@16"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--nfacts", type=int, default=2)
    ap.add_argument("--mtotal", type=int, default=24)
    ap.add_argument("--traj_turns", type=int, default=4)
    ap.add_argument("--regime", default="cot")   # cot or direct
    ap.add_argument("--max_new", type=int, default=420)
    ap.add_argument("--methods", default=None)    # comma list to override METHODS (e.g. K-sweep)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    global METHODS
    if args.methods:
        METHODS = args.methods.split(",")
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lm(args.model, attn="sdpa")
    cot = args.regime == "cot"
    ds = make_dataset(args.n, args.mtotal, args.nfacts, seed0=7000)
    path = os.path.join(os.path.dirname(__file__), "results", f"e3_{tag}.jsonl")
    f = open(path, "w"); t0 = time.time()
    for k, p in enumerate(ds):
        traj = filler_trajectory(args.traj_turns, p.pid)
        q = p.decision_query(cot)
        # direction: even -> start enabled (gold yes), flip to disabled (gold no); odd -> reverse
        if k % 2 == 0:
            start_enabled, new_enabled = True, False
        else:
            start_enabled, new_enabled = False, True
        base = p.with_toggle(p.flip_idx, start_enabled)
        new = p.with_toggle(p.flip_idx, new_enabled)
        mem_old = base.memory_markdown(); mem_new = new.memory_markdown()
        new_gold = "yes" if new.gold_yes else "no"
        flip_attr = p.settings[p.flip_idx]["attr"]
        new_val = "enabled" if new_enabled else "disabled"
        preds = {}
        for method in METHODS:
            K = int(method.split("@")[1]) if "@" in method else 0
            mname = "selective" if method.startswith("selective") else method
            res = run_edit_late(model, tok, SYS, mem_old, mem_new, traj, q, mname,
                                erratum_label=flip_attr, erratum_value=new_val, K=K, return_cache=True)
            logits, recompute, cache, last_id, pos = res
            if cot:
                txt = generate_from_cache(model, tok, cache, last_id, pos, args.max_new)
                pred = parse_final(txt)
            else:
                pred = decide(logits, tok)
            preds[method] = pred
            f.write(json.dumps(dict(model=args.model, persona=p.pid, method=method, n_facts=args.nfacts,
                     mtotal=args.mtotal, flip=f"{start_enabled}->{new_enabled}", new_gold=new_gold,
                     pred=pred, correct=int(pred == new_gold), recompute_tok=int(recompute))) + "\n")
        # annotate agreement with oracle (full_recompute) in a second pass at analysis; store oracle here
        f.flush()
        if (k + 1) % 16 == 0:
            print(f"  {k+1}/{len(ds)} ({time.time()-t0:.0f}s) last preds={preds}", flush=True)
    f.close()
    print(f"E3_DONE {args.model} -> {path}")


if __name__ == "__main__":
    main()
