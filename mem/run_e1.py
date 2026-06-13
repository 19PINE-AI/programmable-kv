"""E1 — placement x pre-digestion (the load-bearing experiment).

Pure placement effect under FULL recompute: early [sys][MEM][traj][q] vs late [sys][traj][MEM][q],
across reasoning (direct/cot) x n_facts x L_mem. Tests whether reading memory directly (late)
costs quality vs pre-digested (early), and whether CoT removes any gap.

Endpoints per (persona, placement, reasoning):
  * direct: pred via fast yes/no logits; correct (vs gold); margin_gold (continuous)
  * cot:    pred via generated reasoning + FINAL parse; correct (vs gold)
Records one JSONL row per decision -> results/e1_<tag>.jsonl.
"""
import os, sys, json, argparse, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from data import make_dataset, filler_trajectory
from memkv import build_prompt, run_full, decide, gold_margin, cot_decision_full, EARLY, LATE
from composable_kv import load_lm
from transformers import AutoTokenizer

SYS = "You are a careful account-management assistant. Follow the user settings exactly."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--facts", default="1,2,4")
    ap.add_argument("--mtotals", default="24,120")     # ~400 tok, ~2k tok
    ap.add_argument("--regimes", default="direct,cot")
    ap.add_argument("--traj_turns", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=420)
    ap.add_argument("--attn", default="sdpa")   # use flash_attention_2 for long (>=8k) memory
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lm(args.model, attn=args.attn)
    facts = [int(x) for x in args.facts.split(",")]
    mtotals = [int(x) for x in args.mtotals.split(",")]
    regimes = args.regimes.split(",")
    path = os.path.join(os.path.dirname(__file__), "results", f"e1_{tag}.jsonl")
    f = open(path, "w"); t0 = time.time(); nd = 0
    for mt in mtotals:
        for nf in facts:
            ds = make_dataset(args.n, mt, nf, seed0=2000 + 100 * mt + nf)
            for p in ds:
                traj = filler_trajectory(args.traj_turns, p.pid)
                mem = p.memory_markdown(); gold = "yes" if p.gold_yes else "no"
                for placement in (EARLY, LATE):
                    for reg in regimes:
                        q = p.decision_query(reg == "cot")
                        ids, mlo, mhi, qlo = build_prompt(tok, SYS, mem, traj, q, placement)
                        Lmem = mhi - mlo
                        if reg == "direct":
                            fl = run_full(model, tok, ids)
                            pred = decide(fl, tok); mg = gold_margin(fl, tok, gold)
                        else:
                            pred, _ = cot_decision_full(model, tok, ids, max_new=args.max_new); mg = None
                        rec = dict(model=args.model, persona=p.pid, placement=placement, reasoning=reg,
                                   n_facts=nf, mtotal=mt, L_total=int(ids.shape[1]), L_mem=int(Lmem),
                                   gold=gold, pred=pred, correct=int(pred == gold))
                        if mg is not None:
                            rec["margin_gold"] = mg
                        f.write(json.dumps(rec) + "\n"); nd += 1
                f.flush()
            print(f"  mt={mt} nf={nf} done ({time.time()-t0:.0f}s, {nd} decisions)", flush=True)
    f.close()
    print(f"E1_DONE {args.model} -> {path}")


if __name__ == "__main__":
    main()
