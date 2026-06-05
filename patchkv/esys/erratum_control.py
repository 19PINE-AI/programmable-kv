"""Negative control: does the erratum work via its CONTENT, or is it an artifact of
appending any text? Across the 8 diverse tasks (non-reasoning), compare:
  oracle        : full new prefill (ceiling)
  stale         : old context (floor)
  err_correct   : old + '[STATE UPDATE] field -> NEW; overrides...'   (our method)
  err_wrong     : old + '[STATE UPDATE] field -> OLD; overrides...'    (states the OLD value)
  err_irrelevant: old + an irrelevant length-matched notice
If err_wrong / err_irrelevant also recover -> the effect is an artifact (any append resets).
Prediction: only err_correct -> correct; err_wrong/irrelevant stay stale.
"""
import argparse, os, sys, json
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from mech_suite import load, ftok, prefill, clone, step, wilson
import diverse_tasks as DT
from collections import Counter

NOTICE = ("[NOTICE] Routine system health check completed successfully at the scheduled "
          "time; all subsystems nominal and no operator action is required.\n\n")


def chat(tok, content):
    try:
        return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                       add_generation_prompt=True)


@torch.no_grad()
def decide(model, tok, ids, toi):
    L = ids.shape[1]
    out = model(input_ids=ids[:, :L - 1].to("cuda"), use_cache=True)
    o = model(input_ids=ids[:, L - 1:L].to("cuda"), past_key_values=out.past_key_values,
              cache_position=torch.tensor([L - 1], device="cuda"), use_cache=True)
    lg = o.logits[0, -1].float()
    return "correct" if lg[toi["correct"]] >= lg[toi["stale"]] else "stale"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    args = ap.parse_args()
    tok, model = load(args.model)
    agg = {c: [] for c in ["oracle", "stale", "err_correct", "err_wrong", "err_irrelevant"]}
    for key in DT.TASKS:
        t = DT.TASKS[key]
        toi = {"correct": ftok(tok, t["correct"]), "stale": ftok(tok, t["stale"])}
        variants = {
            "oracle": chat(tok, DT.build(key, t["vnew"])),
            "stale": chat(tok, DT.build(key, t["vold"])),
            "err_correct": chat(tok, DT.build(key, t["vold"], erratum_value=t["vnew"])),
            "err_wrong": chat(tok, DT.build(key, t["vold"], erratum_value=t["vold"])),
            "err_irrelevant": chat(tok, DT.build(key, t["vold"]).replace("TASK\n", NOTICE + "TASK\n", 1)),
        }
        row = {}
        for c, txt in variants.items():
            ids = torch.tensor([tok(txt, add_special_tokens=False)["input_ids"]])
            d = decide(model, tok, ids, toi); agg[c].append(d); row[c] = d
        print(f"{key:14s} " + " ".join(f"{c}={row[c][:1]}" for c in agg), flush=True)
    res = {"model": args.model}
    for c in agg:
        n = len(agg[c]); kc = sum(x == "correct" for x in agg[c])
        res[c] = {"P_correct": round(kc / n, 2), "ci": wilson(kc, n), "dist": dict(Counter(agg[c]))}
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"erratum_control_{args.tag}.json"), "w"), indent=2)
    print("\n=== erratum content control (n=8 tasks) ===")
    for c in agg:
        print(f"  {c:14s} P_correct={res[c]['P_correct']} {res[c]['ci']}")
    print("ERRATUM_CONTROL_DONE", flush=True)


if __name__ == "__main__":
    main()
