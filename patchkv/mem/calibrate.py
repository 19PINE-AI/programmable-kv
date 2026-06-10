"""Calibration: oracle (full-recompute, late) decision accuracy per model x n_facts, to
(a) validate the gated task is solvable (oracle >= 0.80 inclusion gate) and
(b) pick competent models for the confirmatory runs. Also prints the yes/no logit margin."""
import os, sys, argparse, json
import torch
sys.path.insert(0, os.path.dirname(__file__))
from data import make_dataset, filler_trajectory
from memkv import build_prompt, run_full, decide, LATE, EARLY
from composable_kv import load_lm
from transformers import AutoTokenizer

SYS = "You are a careful account-management assistant. Follow the user's settings exactly."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--facts", default="1,2,4,8")
    ap.add_argument("--mtotal", type=int, default=24)
    ap.add_argument("--traj_turns", type=int, default=4)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lm(args.model, attn="sdpa")
    out = {}
    for nf in [int(x) for x in args.facts.split(",")]:
        ds = make_dataset(args.n, args.mtotal, nf, seed0=1000)
        correct = 0; margins = []
        for p in ds:
            traj = filler_trajectory(args.traj_turns, p.pid)
            ids, mlo, mhi, qlo = build_prompt(tok, SYS, p.memory_markdown(), traj,
                                              p.decision_query(False), LATE)
            fl = run_full(model, tok, ids)
            d = decide(fl, tok)
            gold = "yes" if p.gold_yes else "no"
            correct += (d == gold)
            ty = tok("yes", add_special_tokens=False)["input_ids"][0]
            tn = tok("no", add_special_tokens=False)["input_ids"][0]
            margins.append(abs((fl[ty] - fl[tn]).item()))
        acc = correct / len(ds)
        out[nf] = dict(acc=round(acc, 3), n=len(ds), median_margin=round(sorted(margins)[len(margins)//2], 2))
        print(f"  n_facts={nf}: oracle_acc={acc:.3f} (n={len(ds)}) median|yes-no|={out[nf]['median_margin']}", flush=True)
    tag = args.model.split("/")[-1].replace(".", "_")
    json.dump({"model": args.model, "mtotal": args.mtotal, "calib": out},
              open(os.path.join(os.path.dirname(__file__), "results", f"calib_{tag}.json"), "w"), indent=2)
    print("CALIB_DONE", args.model)

if __name__ == "__main__":
    main()
