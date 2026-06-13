"""E2c: residual-refresh recovery (bridges E1 -> mechanism).

For the decision-relevant scenarios where field-only leave-stale FAILS, test
whether refreshing a SPARSE residual -- the field span PLUS the gating-rule span
(the high-deviation region E1 flags) -- recovers the correct (oracle_new)
decision, while everything else stays stale. Reports the residual size as a
fraction of downstream tokens (should be small => sparse).

Conditions:
  oracle_new           full new prefill (ground truth)
  field_only           refresh field span only            (= E2b patched; fails for high)
  field_plus_gate      refresh field span + gating rule    (sparse residual)
  field_plus_window    refresh field + a +/-K token window around gate (robustness)
  stale_full           refresh nothing                     (floor)
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.dirname(__file__))
import capture  # noqa
from align import align_pair
import scenarios as S
from run_e2 import load_model, prefill, clone_cache, greedy_decode, first_line
from transformers.cache_utils import DynamicCache

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def refresh_spans(cache_old, cache_new, spans, upto):
    c = clone_cache(cache_old, upto)
    for i, layer in enumerate(cache_new.layers):
        for (s, e) in spans:
            c.layers[i].keys[:, :, s:e, :] = layer.keys[:, :, s:e, :]
            c.layers[i].values[:, :, s:e, :] = layer.values[:, :, s:e, :]
    return c


def find_token_span(tok, full_ids, needle_text):
    """Locate the token span of needle_text inside full_ids (best-effort)."""
    needle = tok(needle_text, add_special_tokens=False)["input_ids"]
    fi = full_ids.tolist()
    n = len(needle)
    for i in range(len(fi) - n + 1):
        if fi[i:i + n] == needle:
            return (i, i + n)
    # fallback: match on a distinctive prefix of the needle
    for L in (n - 1, n - 2, 12, 8):
        if L <= 0:
            continue
        sub = needle[:L]
        for i in range(len(fi) - L + 1):
            if fi[i:i + L] == sub:
                return (i, i + L)
    return None


def build_text(scn_key, value, n_neutral, tok, use_chat):
    ctx = S.build(scn_key, value, n_neutral)
    if use_chat:
        return tok.apply_chat_template([{"role": "user", "content": ctx}],
                                       tokenize=False, add_generation_prompt=True)
    return ctx


def run_one(tok, model, scn_key, n_neutral, use_chat, max_new, window):
    s = S.SCENARIOS[scn_key]
    old_text = build_text(scn_key, s["v_old"], n_neutral, tok, use_chat)
    new_text = build_text(scn_key, s["v_new"], n_neutral, tok, use_chat)
    al = align_pair(tok, old_text, new_text)
    a, b = al["field_span"]; T = al["seq_len"]
    new_ids = al["new_ids"][0]
    gate_span = find_token_span(tok, new_ids, s["gate"][:120])
    co = prefill(model, al["old_ids"]); cn = prefill(model, al["new_ids"])
    eos = {tok.eos_token_id}
    last = new_ids[T - 1]

    def decode(spans):
        c = refresh_spans(co, cn, spans, T - 1)
        return first_line(tok, greedy_decode(model, c, last, T - 1, max_new, eos))

    field = (a, b)
    gate = gate_span if gate_span else (a, b)
    win = (max(0, gate[0] - window), min(T - 1, gate[1] + window))
    L = {
        "oracle_new": first_line(tok, greedy_decode(model, clone_cache(cn, T - 1), last, T - 1, max_new, eos)),
        "oracle_old": first_line(tok, greedy_decode(model, clone_cache(co, T - 1), al["old_ids"][0, T - 1], T - 1, max_new, eos)),
        "field_only": decode([field]),
        "field_plus_gate": decode([field, gate]),
        "field_plus_window": decode([field, win]),
        "stale_full": decode([]),  # refresh nothing
    }
    down = (T - 1) - b  # downstream token count (after field, before decode token)
    resid_gate = (gate[1] - gate[0])
    resid_win = (win[1] - win[0])
    return {
        "scenario": scn_key, "cls": s["cls"], "seq_len": T,
        "field_span": [a, b], "gate_span": list(gate), "gate_found": gate_span is not None,
        "lines": L,
        "decision_changed": L["oracle_old"] != L["oracle_new"],
        "field_only_tracks": L["field_only"] == L["oracle_new"],
        "field_plus_gate_tracks": L["field_plus_gate"] == L["oracle_new"],
        "field_plus_window_tracks": L["field_plus_window"] == L["oracle_new"],
        "residual_gate_frac_downstream": resid_gate / max(1, down),
        "residual_window_frac_downstream": resid_win / max(1, down),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_neutral", type=int, default=40)
    ap.add_argument("--max_new", type=int, default=40)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--chat", action="store_true")
    args = ap.parse_args()
    tok, model = load_model(args.model)
    recs = []
    for k in S.SCENARIOS:
        r = run_one(tok, model, k, args.n_neutral, args.chat, args.max_new, args.window)
        r["model"] = args.model
        recs.append(r)
        print(f"{k:18s} cls={r['cls']:6s} changed={int(r['decision_changed'])} "
              f"field_only={int(r['field_only_tracks'])} "
              f"+gate={int(r['field_plus_gate_tracks'])} "
              f"+win={int(r['field_plus_window_tracks'])} "
              f"resid_gate={r['residual_gate_frac_downstream']*100:.1f}% "
              f"gate_found={int(r['gate_found'])}")
        for c in ["oracle_old", "oracle_new", "field_only", "field_plus_gate", "stale_full"]:
            print(f"    {c:18s}: {r['lines'][c][:70]}")
    json.dump(recs, open(os.path.join(RES, f"e2c_{args.tag}.json"), "w"), indent=2)
    print("wrote", os.path.join(RES, f"e2c_{args.tag}.json"))


if __name__ == "__main__":
    main()
