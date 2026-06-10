"""Negative control (pre-registered false-positive check): editing an IRRELEVANT memory fact
must NOT change the decision. We toggle a non-gating setting via each edit method and measure
decision stability vs the pre-edit oracle. A faithful method should be ~1.0 stable; a method
that spuriously rewrites the conclusion would drop. Writes results/negctrl_<tag>.jsonl."""
import os, sys, json, argparse, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from data import make_dataset, filler_trajectory
from memkv import run_edit_late, generate_from_cache, parse_final, decide, run_full, build_prompt, LATE
from composable_kv import load_lm
from transformers import AutoTokenizer

SYS = "You are a careful account-management assistant. Follow the user settings exactly."
METHODS = ["in_place", "erratum", "recompile_chunk", "selective@4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--nfacts", type=int, default=2)
    ap.add_argument("--mtotal", type=int, default=24)
    ap.add_argument("--regime", default="cot")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lm(args.model, attn="sdpa")
    cot = args.regime == "cot"
    ds = make_dataset(args.n, args.mtotal, args.nfacts, seed0=11000)
    path = os.path.join(os.path.dirname(__file__), "results", f"negctrl_{tag}.jsonl")
    f = open(path, "w"); t0 = time.time()
    for k, p in enumerate(ds):
        traj = filler_trajectory(4, p.pid)
        q = p.decision_query(cot)
        mem_old = p.memory_markdown()
        # toggle an IRRELEVANT setting -> gold unchanged
        irr = p.irrelevant_idx
        new_enabled = not p.settings[irr]["enabled"]
        p_new = p.with_toggle(irr, new_enabled)
        mem_new = p_new.memory_markdown()
        assert p_new.gold_yes == p.gold_yes  # irrelevant edit must not change gold
        # pre-edit oracle decision
        ids, _, _, _ = build_prompt(tok, SYS, mem_old, traj, q, LATE)
        if cot:
            from memkv import cot_decision_full
            pre, _ = cot_decision_full(model, tok, ids)
        else:
            pre = decide(run_full(model, tok, ids), tok)
        irr_attr = p.settings[irr]["attr"]; val = "enabled" if new_enabled else "disabled"
        for method in METHODS:
            K = int(method.split("@")[1]) if "@" in method else 0
            mname = "selective" if method.startswith("selective") else method
            res = run_edit_late(model, tok, SYS, mem_old, mem_new, traj, q, mname,
                                erratum_label=irr_attr, erratum_value=val, K=K, return_cache=True)
            logits, recompute, cache, last_id, pos = res
            post = parse_final(generate_from_cache(model, tok, cache, last_id, pos)) if cot else decide(logits, tok)
            f.write(json.dumps(dict(model=args.model, persona=p.pid, method=method,
                     pre=pre, post=post, stable=int(pre == post), n_facts=args.nfacts)) + "\n")
        f.flush()
        if (k + 1) % 16 == 0:
            print(f"  {k+1}/{len(ds)} ({time.time()-t0:.0f}s)", flush=True)
    f.close()
    print(f"NEGCTRL_DONE {args.model} -> {path}")


if __name__ == "__main__":
    main()
