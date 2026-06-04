"""Phase B: E1 blast radius + E2 decision-flip on tau-bench-grounded contexts."""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.dirname(__file__))
import capture
from align import align_pair
from deviation import attention_output_deviation
from run_e2 import load_model, prefill, clone_cache, greedy_decode, first_line
from run_selection import capture_fwd
from run_e2c import refresh_spans
import taubench_ctx as TBC

RES = os.path.join(os.path.dirname(__file__), "..", "results")
K_GRID = [0, 8, 16, 32, 64, 128, 256, 512]


def build_text(scn_key, value, tok, use_chat):
    ctx = TBC.build(scn_key, value)
    if use_chat:
        return tok.apply_chat_template([{"role": "user", "content": ctx}],
                                       tokenize=False, add_generation_prompt=True)
    return ctx


def run_one(tok, model, scn_key, use_chat, max_new):
    s = TBC.SCEN[scn_key]
    ot = build_text(scn_key, s["v_old"], tok, use_chat)
    nt = build_text(scn_key, s["v_new"], tok, use_chat)
    al = align_pair(tok, ot, nt); a, b = al["field_span"]; T = al["seq_len"]
    # E1 blast radius
    capo = capture_fwd(model, al["old_ids"]); capn = capture_fwd(model, al["new_ids"])
    dev = np.zeros(T)
    for li in range(len(capn)):
        ad = attention_output_deviation(capn[li]["q"][0], capn[li]["k"][0], capn[li]["v"][0],
                                        capo[li]["k"][0], capo[li]["v"][0], (a, b),
                                        capn[li]["scaling"], device="cuda").numpy()
        dev = np.maximum(dev, ad)
    pos = np.arange(T)
    exact_max = float(dev[pos < a].max()) if (pos < a).any() else 0.0
    down = pos >= b
    br = {f"{t:g}": float((dev[down] > t).mean()) for t in [0.05, 0.1, 0.2]}
    # E2 decision + recency recovery
    co = prefill(model, al["old_ids"]); cn = prefill(model, al["new_ids"])
    eos = {tok.eos_token_id}; last = al["new_ids"][0, T - 1]
    oracle_new = first_line(tok, greedy_decode(model, clone_cache(cn, T - 1), last, T - 1, max_new, eos))
    oracle_old = first_line(tok, greedy_decode(model, clone_cache(co, T - 1), al["old_ids"][0, T - 1], T - 1, max_new, eos))
    ndown = (T - 1) - b
    sweep = []; min_rec = None
    for K in K_GRID:
        st = max(b, T - 1 - K) if K > 0 else None
        spans = [(a, b)] + ([(st, T - 1)] if K > 0 else [])
        line = first_line(tok, greedy_decode(model, refresh_spans(co, cn, spans, T - 1), last, T - 1, max_new, eos))
        tr = (line == oracle_new)
        sweep.append({"K": K, "frac": min(K, ndown) / max(1, ndown), "tracks": tr})
        if tr and min_rec is None:
            min_rec = min(K, ndown) / max(1, ndown)
    return {"scenario": scn_key, "cls": s["cls"], "field": s["field"],
            "seq_len": T, "field_span": [a, b], "downstream": ndown,
            "exact_region_dev_max": exact_max, "blast_radius": br,
            "oracle_old": oracle_old, "oracle_new": oracle_new,
            "decision_changed": oracle_old != oracle_new,
            "min_recover_frac": min_rec, "recovery_sweep": sweep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max_new", type=int, default=40)
    ap.add_argument("--chat", action="store_true")
    args = ap.parse_args()
    tok, model = load_model(args.model)
    recs = []
    for k in TBC.SCEN:
        r = run_one(tok, model, k, args.chat, args.max_new)
        r["model"] = args.model
        recs.append(r)
        mr = r["min_recover_frac"]
        print(f"{k:20s} cls={r['cls']:5s} field={r['field']:13s} T={r['seq_len']} "
              f"exact_dev={r['exact_region_dev_max']:.1e} BR@0.1={r['blast_radius']['0.1']*100:.1f}% "
              f"changed={int(r['decision_changed'])} "
              f"min_recover={'%.1f%%'%(mr*100) if mr is not None else 'never'}")
        print(f"    oracle_old: {r['oracle_old'][:75]}")
        print(f"    oracle_new: {r['oracle_new'][:75]}")
    json.dump(recs, open(os.path.join(RES, f"taubench_{args.tag}.json"), "w"), indent=2)
    print("wrote", os.path.join(RES, f"taubench_{args.tag}.json"))


if __name__ == "__main__":
    main()
