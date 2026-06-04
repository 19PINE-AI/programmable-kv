"""Phase D: cost/quality/latency frontier for the in-place field edit.

Scenario: an OLD context is already cached; a field flips to NEW. We compare how
each method produces a decode-ready cache, on three axes:
  * decision agreement vs the full-reprefill oracle (correctness)
  * recompute fraction = recomputed tokens / T (hardware-independent cost)
  * wall-clock latency of the update (warmup + median of trials)

Methods:
  full_reprefill   recompute everything            (ceiling: correct, expensive)
  stale_reuse      reuse old cache, no refresh      (floor: free, wrong on change)
  hoist_to_end     field moved to suffix; recompute only field+tail (real baseline)
  patchkv_kK       faithful: exact field refresh + recompute last-K tokens (ours)
"""
import argparse, json, os, sys, statistics
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2"))
import capture  # noqa
from align import align_pair
import scenarios as S
from run_e2 import load_model, prefill, clone_cache, greedy_decode, first_line
import mechanism as M

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def timed(fn, trials=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


import re
def _tool(line):
    m = re.search(r"([A-Za-z_]\w*)\s*\(", line)
    return m.group(1) if m else (line.strip().split() or [""])[0]


def decide(model, tok, cache, last_tok, upto, max_new, eos):
    return first_line(tok, greedy_decode(model, clone_cache(cache, upto), last_tok, upto, max_new, eos))


def run_scenario(tok, model, scn_key, n_neutral, max_new, recents):
    s = S.SCENARIOS[scn_key]
    # natural placement old/new
    ot = tok.apply_chat_template([{"role": "user", "content": S.build(scn_key, s["v_old"], n_neutral)}],
                                 tokenize=False, add_generation_prompt=True)
    nt = tok.apply_chat_template([{"role": "user", "content": S.build(scn_key, s["v_new"], n_neutral)}],
                                 tokenize=False, add_generation_prompt=True)
    al = align_pair(tok, ot, nt); a, b = al["field_span"]; T = al["seq_len"]; upto = T - 1
    new_ids = al["new_ids"]
    co = prefill(model, al["old_ids"]); cn_full = prefill(model, al["new_ids"])
    eos = {tok.eos_token_id}; last = new_ids[0, upto]

    # oracle decision (full reprefill, natural placement)
    oracle = decide(model, tok, cn_full, last, upto, max_new, eos)
    old_dec = decide(model, tok, co, al["old_ids"][0, upto], upto, max_new, eos)

    results = {}

    def record(name, build_cache_fn, recompute_tokens, decode_last, decode_ids_upto):
        c, _ = build_cache_fn()
        dec = decide(model, tok, c, decode_last, decode_ids_upto, max_new, eos)
        lat = timed(lambda: build_cache_fn())
        results[name] = {"decision": dec, "agree_oracle": _tool(dec) == _tool(oracle),
                         "recompute_tokens": int(recompute_tokens),
                         "recompute_frac": recompute_tokens / T,
                         "latency_ms": round(lat, 3)}

    # full reprefill
    record("full_reprefill", lambda: M.full_reprefill_cache(model, new_ids, upto), upto, last, upto)
    # stale reuse
    record("stale_reuse", lambda: M.stale_cache(co, upto), 0, last, upto)
    # patchkv faithful, several recency windows
    for K in recents:
        record(f"patchkv_k{K}", lambda K=K: M.patchkv_cache(model, new_ids, co, (a, b), K, upto),
               # recompute tokens counted from the call below
               _patchkv_cost(M, model, new_ids, co, (a, b), K, upto), last, upto)

    # hoist-to-end baseline: field at suffix. static prefix cached; recompute field+tail.
    h_ot = tok.apply_chat_template([{"role": "user", "content": S.build(scn_key, s["v_old"], n_neutral, hoist=True)}],
                                   tokenize=False, add_generation_prompt=True)
    h_nt = tok.apply_chat_template([{"role": "user", "content": S.build(scn_key, s["v_new"], n_neutral, hoist=True)}],
                                   tokenize=False, add_generation_prompt=True)
    hal = align_pair(tok, h_ot, h_nt); hs, he = hal["field_span"]; hT = hal["seq_len"]; hupto = hT - 1
    h_new = hal["new_ids"]; h_last = h_new[0, hupto]
    h_co = prefill(model, hal["old_ids"])
    # hoist recompute = everything from the (late) field start to end
    hoist_recompute = hupto - hs
    h_oracle_full = prefill(model, hal["new_ids"])
    def hoist_build():
        return M.recompute_suffix(model, h_new, clone_cache(h_co, hupto), hs, hupto)
    c, _ = hoist_build()
    hdec = decide(model, tok, c, h_last, hupto, max_new, eos)
    hlat = timed(hoist_build)
    results["hoist_to_end"] = {"decision": hdec, "agree_oracle": _tool(hdec) == _tool(oracle),
                               "recompute_tokens": int(hoist_recompute),
                               "recompute_frac": hoist_recompute / hT,
                               "latency_ms": round(hlat, 3),
                               "note": "field physically moved to suffix"}

    return {"scenario": scn_key, "cls": s["cls"], "seq_len": T, "field_span": [a, b],
            "decision_changed": old_dec != oracle, "oracle": oracle, "old_decision": old_dec,
            "methods": results}


def _patchkv_cost(M, model, new_ids, co, span, K, upto):
    s, e = span
    win_start = max(e, upto - K) if K > 0 else upto
    return (e - s) + (upto - win_start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_neutral", type=int, default=40)
    ap.add_argument("--max_new", type=int, default=40)
    ap.add_argument("--recents", default="0,32,128,256")
    args = ap.parse_args()
    tok, model = load_model(args.model)
    recents = [int(x) for x in args.recents.split(",")]
    recs = []
    for k in S.SCENARIOS:
        r = run_scenario(tok, model, k, args.n_neutral, args.max_new, recents)
        recs.append(r)
        print(f"\n=== {k} [{r['cls']}] T={r['seq_len']} changed={int(r['decision_changed'])} oracle={r['oracle'][:40]}")
        for name, m in r["methods"].items():
            print(f"  {name:16s} agree={int(m['agree_oracle'])} recompute={m['recompute_frac']*100:5.1f}% "
                  f"lat={m['latency_ms']:7.2f}ms  '{m['decision'][:34]}'")
    json.dump(recs, open(os.path.join(RES, f"frontier_{args.tag}.json"), "w"), indent=2)
    print("\nwrote", os.path.join(RES, f"frontier_{args.tag}.json"))


if __name__ == "__main__":
    main()
