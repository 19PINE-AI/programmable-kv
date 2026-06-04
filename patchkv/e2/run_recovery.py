"""E2 recovery-contract sweep (reproducible artifact).

For each decision-relevant scenario, decode under:
  oracle_new / oracle_old, plus leave-stale with a refreshed FIELD + a recent
  window of the last-K downstream tokens, for K in a grid. Records the minimal
  recent-window fraction that recovers the oracle_new decision = the per-field
  refresh contract. Low-conditioning controls included (should track at K=0).
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.dirname(__file__))
import capture  # noqa
from align import align_pair
import scenarios as S
from run_e2 import load_model, prefill, clone_cache, greedy_decode, first_line
from run_e2c import refresh_spans, build_text

RES = os.path.join(os.path.dirname(__file__), "..", "results")
K_GRID = [0, 8, 16, 32, 64, 128, 192, 256, 384, 512]


def run_one(tok, model, scn_key, n_neutral, max_new):
    s = S.SCENARIOS[scn_key]
    ot = build_text(scn_key, s["v_old"], n_neutral, tok, True)
    nt = build_text(scn_key, s["v_new"], n_neutral, tok, True)
    al = align_pair(tok, ot, nt); a, b = al["field_span"]; T = al["seq_len"]
    down = (T - 1) - b
    co = prefill(model, al["old_ids"]); cn = prefill(model, al["new_ids"])
    eos = {tok.eos_token_id}; last = al["new_ids"][0, T - 1]
    oracle_new = first_line(tok, greedy_decode(model, clone_cache(cn, T - 1), last, T - 1, max_new, eos))
    oracle_old = first_line(tok, greedy_decode(model, clone_cache(co, T - 1), al["old_ids"][0, T - 1], T - 1, max_new, eos))
    sweep = []
    min_recover = None
    for K in K_GRID:
        st = max(b, T - 1 - K) if K > 0 else None
        spans = [(a, b)] + ([(st, T - 1)] if K > 0 else [])
        line = first_line(tok, greedy_decode(model, refresh_spans(co, cn, spans, T - 1), last, T - 1, max_new, eos))
        tracks = (line == oracle_new)
        sweep.append({"K": K, "frac_down": min(K, down) / max(1, down),
                      "tracks": tracks, "line": line})
        if tracks and min_recover is None:
            min_recover = min(K, down) / max(1, down)
    return {"scenario": scn_key, "cls": s["cls"], "seq_len": T,
            "downstream": down, "oracle_old": oracle_old, "oracle_new": oracle_new,
            "decision_changed": oracle_old != oracle_new,
            "min_recover_frac": min_recover, "sweep": sweep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_neutral", type=int, default=40)
    ap.add_argument("--max_new", type=int, default=40)
    args = ap.parse_args()
    tok, model = load_model(args.model)
    recs = [run_one(tok, model, k, args.n_neutral, args.max_new) for k in S.SCENARIOS]
    for r in recs:
        mr = r["min_recover_frac"]
        print(f"{r['scenario']:18s} cls={r['cls']:6s} changed={int(r['decision_changed'])} "
              f"min_recover={'%.1f%%'%(mr*100) if mr is not None else 'never'} "
              f"(down={r['downstream']})")
    json.dump(recs, open(os.path.join(RES, f"recovery_{args.tag}.json"), "w"), indent=2)
    print("wrote", os.path.join(RES, f"recovery_{args.tag}.json"))


if __name__ == "__main__":
    main()
