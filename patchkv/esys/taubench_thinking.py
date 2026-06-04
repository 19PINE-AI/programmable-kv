"""Validate the field-only + thinking recipe on REAL tau-bench policy.

order_status flips pending<->delivered (gated by the real retail wiki). With a
thinking model, does refreshing only the field token (leave all else stale) recover
the correct flipped decision? Compares stale_full / field_only / oracle_new.
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2"))
import capture  # noqa
from align import align_pair
from run_e2 import load_model, prefill, clone_cache
import taubench_ctx as TBC
import mechanism as M
from thinking_test import decode_think


def chat(tok, content, think):
    return tok.apply_chat_template([{"role": "user", "content": content}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=think)


def run(model, tok, scn, think, max_new):
    s = TBC.SCEN[scn]
    t_old = chat(tok, TBC.build(scn, s["v_old"]), think)
    t_new = chat(tok, TBC.build(scn, s["v_new"]), think)
    al = align_pair(tok, t_old, t_new); a, b = al["field_span"]; T = al["seq_len"]; upto = T - 1
    co = prefill(model, al["old_ids"]); cn = prefill(model, al["new_ids"])

    def dec(cache, last):
        return decode_think(model, tok, clone_cache(cache, upto), last, upto, max_new=max_new)

    oracle = dec(cn, al["new_ids"][0, upto])
    old = dec(co, al["old_ids"][0, upto])
    fcache, nf = M.patchkv_cache(model, al["new_ids"], co, (a, b), 0, upto)
    field_only = dec(fcache, al["new_ids"][0, upto])
    return {"scenario": scn, "cls": s["cls"], "field": s["field"], "seq_len": T,
            "field_span": [a, b], "exact_reuse_frac": a / T,
            "field_recompute_frac": nf / T,
            "oracle_new": oracle["tool"], "oracle_old": old["tool"],
            "field_only": field_only["tool"],
            "decision_changed": old["tool"] != oracle["tool"],
            "field_only_recovers": field_only["tool"] == oracle["tool"],
            "think_tokens": {"oracle": oracle["think_tokens"], "field_only": field_only["think_tokens"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--max_new", type=int, default=1536)
    ap.add_argument("--no_think", action="store_true")
    args = ap.parse_args()
    tok, model = load_model(args.model)
    recs = []
    for scn in TBC.SCEN:
        r = run(model, tok, scn, not args.no_think, args.max_new)
        recs.append(r)
        print(f"{scn:20s} [{r['cls']}] changed={int(r['decision_changed'])} "
              f"field_only_recovers={int(r['field_only_recovers'])} "
              f"(field refresh {r['field_recompute_frac']*100:.1f}%, exact-reuse {r['exact_reuse_frac']*100:.0f}%)")
        print(f"    oracle_new={r['oracle_new']}  oracle_old={r['oracle_old']}  field_only={r['field_only']}")
    json.dump(recs, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"taubench_thinking_{args.tag}.json"), "w"), indent=2)
    print("wrote results")


if __name__ == "__main__":
    main()
