"""E2b: decision-flip faithfulness on DECISION-RELEVANT scenarios (the teeth).

Reuses the cache machinery from run_e2 but with engineered scenarios where
flipping the field should flip the correct action. Reports, per scenario:
  decision_changed   = oracle_old != oracle_new   (field is decision-relevant)
  patched_tracks     = patched == oracle_new       (leave-stale gets it right)
  stale_tracks       = stale_full == oracle_new     (floor: should be FALSE when changed)
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.dirname(__file__))
import capture  # noqa
from align import align_pair
import scenarios as S
from run_e2 import (load_model, prefill, clone_cache, patched_cache,
                    greedy_decode, first_line)

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def build_text(scn_key, value, n_neutral, tok, use_chat):
    ctx = S.build(scn_key, value, n_neutral)
    if use_chat:
        return tok.apply_chat_template([{"role": "user", "content": ctx}],
                                       tokenize=False, add_generation_prompt=True)
    return ctx


def run_one(tok, model, scn_key, n_neutral, use_chat, max_new):
    s = S.SCENARIOS[scn_key]
    old_text = build_text(scn_key, s["v_old"], n_neutral, tok, use_chat)
    new_text = build_text(scn_key, s["v_new"], n_neutral, tok, use_chat)
    al = align_pair(tok, old_text, new_text)
    a, b = al["field_span"]; T = al["seq_len"]
    co = prefill(model, al["old_ids"]); cn = prefill(model, al["new_ids"])
    eos = {tok.eos_token_id}
    last = al["new_ids"][0, T - 1]
    g_on = greedy_decode(model, clone_cache(cn, T - 1), last, T - 1, max_new, eos)
    g_oo = greedy_decode(model, clone_cache(co, T - 1), al["old_ids"][0, T - 1], T - 1, max_new, eos)
    g_pa = greedy_decode(model, patched_cache(co, cn, (a, b), T - 1), last, T - 1, max_new, eos)
    g_st = greedy_decode(model, clone_cache(co, T - 1), last, T - 1, max_new, eos)
    L = {k: first_line(tok, g) for k, g in
         dict(oracle_new=g_on, oracle_old=g_oo, patched=g_pa, stale_full=g_st).items()}
    changed = L["oracle_old"] != L["oracle_new"]
    return {
        "scenario": scn_key, "cls": s["cls"], "field_span": [a, b], "seq_len": T,
        "v_old": s["v_old"], "v_new": s["v_new"],
        "exp_old": s["exp_old"], "exp_new": s["exp_new"], "lines": L,
        "decision_changed": changed,
        "patched_tracks": L["patched"] == L["oracle_new"],
        "stale_tracks": L["stale_full"] == L["oracle_new"],
        "patched_eq_stale": L["patched"] == L["stale_full"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_neutral", type=int, default=40)
    ap.add_argument("--max_new", type=int, default=40)
    ap.add_argument("--chat", action="store_true")
    args = ap.parse_args()
    tok, model = load_model(args.model)
    recs = []
    for k in S.SCENARIOS:
        r = run_one(tok, model, k, args.n_neutral, args.chat, args.max_new)
        r["model"] = args.model; r["tag"] = args.tag
        recs.append(r)
        print(f"{k:18s} cls={r['cls']:6s} changed={int(r['decision_changed'])} "
              f"patched_tracks={int(r['patched_tracks'])} stale_tracks={int(r['stale_tracks'])}")
        for cond in ["oracle_old", "oracle_new", "patched", "stale_full"]:
            print(f"    {cond:11s}: {r['lines'][cond][:78]}")
    json.dump(recs, open(os.path.join(RES, f"e2b_{args.tag}.json"), "w"), indent=2)
    print("wrote", os.path.join(RES, f"e2b_{args.tag}.json"))


if __name__ == "__main__":
    main()
