"""Phase A: residual SELECTION policy comparison (E1 deviation -> E2 decision).

For each decision-relevant scenario we rank downstream tokens by their E1
attention-output deviation under a FIELD-ONLY leave-stale patch (the deviation a
token's attention output suffers because its KV is stale). We then refresh the
field plus the top-k% of downstream tokens under three policies and ask: what is
the minimal refresh fraction that recovers the oracle_new decision?

  deviation : refresh field + top-k% highest-deviation downstream tokens   (ours)
  recency   : refresh field + last-k% downstream tokens (contiguous suffix)
  random    : refresh field + a random k% of downstream tokens (control)

If deviation-ranked recovers with the smallest fraction, selection is validated.
"""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.dirname(__file__))
import capture
from align import align_pair
from deviation import attention_output_deviation
import scenarios as S
from run_e2 import load_model, prefill, clone_cache, greedy_decode, first_line
from run_e2c import build_text

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FRACS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]


def capture_fwd(model, ids):
    capture.enable_capture(to_cpu=True)
    with torch.no_grad():
        model(input_ids=ids.to("cuda"), use_cache=True)
    capture.disable_capture()
    return capture.get_capture()


def downstream_deviation(cap_old, cap_new, field_span, T):
    """Per-token max-over-layers attention-output deviation (field-only patch)."""
    s, e = field_span
    nl = len(cap_new)
    dev = np.zeros(T)
    for li in range(nl):
        co, cn = cap_old[li], cap_new[li]
        ad = attention_output_deviation(cn["q"][0], cn["k"][0], cn["v"][0],
                                        co["k"][0], co["v"][0], (s, e),
                                        cn["scaling"], device="cuda").numpy()
        dev = np.maximum(dev, ad)
    return dev


def refresh_index_set(cache_old, cache_new, idx, upto):
    """Patched cache: old truncated to upto, positions in idx overwritten by new."""
    c = clone_cache(cache_old, upto)
    if len(idx) == 0:
        return c
    t = torch.tensor(sorted(idx), device=c.layers[0].keys.device, dtype=torch.long)
    for i, layer in enumerate(cache_new.layers):
        c.layers[i].keys[:, :, t, :] = layer.keys[:, :, t, :]
        c.layers[i].values[:, :, t, :] = layer.values[:, :, t, :]
    return c


def run_one(tok, model, scn_key, n_neutral, max_new, seed_offset):
    s = S.SCENARIOS[scn_key]
    ot = build_text(scn_key, s["v_old"], n_neutral, tok, True)
    nt = build_text(scn_key, s["v_new"], n_neutral, tok, True)
    al = align_pair(tok, ot, nt); a, b = al["field_span"]; T = al["seq_len"]
    co = prefill(model, al["old_ids"]); cn = prefill(model, al["new_ids"])
    # captures for deviation ranking
    capo = capture_fwd(model, al["old_ids"]); capn = capture_fwd(model, al["new_ids"])
    dev = downstream_deviation(capo, capn, (a, b), T)

    down_idx = list(range(b, T - 1))           # downstream, excluding decode token
    ndown = len(down_idx)
    dev_down = dev[b:T - 1]
    order_dev = [down_idx[i] for i in np.argsort(-dev_down)]   # high->low deviation
    eos = {tok.eos_token_id}; last = al["new_ids"][0, T - 1]

    oracle_new = first_line(tok, greedy_decode(model, clone_cache(cn, T - 1), last, T - 1, max_new, eos))
    oracle_old = first_line(tok, greedy_decode(model, clone_cache(co, T - 1), al["old_ids"][0, T - 1], T - 1, max_new, eos))

    # deterministic pseudo-random order (no Math.random); rotate by seed_offset
    rng_order = list(down_idx)
    off = (seed_offset * 2654435761) % max(1, ndown)
    rng_order = rng_order[off:] + rng_order[:off]

    policies = {"deviation": order_dev,
                "recency": list(reversed(down_idx)),     # last tokens first
                "random": rng_order}
    out = {p: [] for p in policies}
    min_rec = {}
    for p, order in policies.items():
        for f in FRACS:
            k = int(round(f * ndown))
            sel = set(order[:k]) | set(range(a, b))      # always include field
            line = first_line(tok, greedy_decode(
                model, refresh_index_set(co, cn, sel, T - 1), last, T - 1, max_new, eos))
            tracks = (line == oracle_new)
            out[p].append({"frac": f, "k": k, "tracks": tracks})
            if tracks and p not in min_rec:
                min_rec[p] = f
    return {"scenario": scn_key, "cls": s["cls"], "seq_len": T, "downstream": ndown,
            "field_span": [a, b], "oracle_old": oracle_old, "oracle_new": oracle_new,
            "decision_changed": oracle_old != oracle_new,
            "min_recover": {p: min_rec.get(p) for p in policies},
            "sweep": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_neutral", type=int, default=40)
    ap.add_argument("--max_new", type=int, default=40)
    args = ap.parse_args()
    tok, model = load_model(args.model)
    recs = []
    for j, k in enumerate(S.SCENARIOS):
        r = run_one(tok, model, k, args.n_neutral, args.max_new, j + 1)
        r["model"] = args.model
        recs.append(r)
        mr = r["min_recover"]
        def fmt(x): return f"{x*100:.0f}%" if x is not None else "never"
        print(f"{k:18s} cls={r['cls']:6s} changed={int(r['decision_changed'])} "
              f"min_recover: deviation={fmt(mr['deviation'])} recency={fmt(mr['recency'])} "
              f"random={fmt(mr['random'])}")
    json.dump(recs, open(os.path.join(RES, f"selection_{args.tag}.json"), "w"), indent=2)
    print("wrote", os.path.join(RES, f"selection_{args.tag}.json"))


if __name__ == "__main__":
    main()
