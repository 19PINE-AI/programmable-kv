"""Sparse-refresh probe: is the necessary residual 'everything from the rule
onward', or just field + gate + recency (skipping the neutral middle)?

Faithfully recomputes arbitrary spans IN POSITION ORDER against the current
working cache (so each refreshed span attends to whatever has already been
refreshed). Answers a correction to FINDINGS_EXTENSIONS.md: the sufficient refresh
is governed by the field's conditioning BREADTH (~E1 blast radius), not placement
alone. Moderate fields (safety_mode) recover from a ~6% sparse set; broad fields
(account_role) need ~everything after the field.
"""
import os, sys, re
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2"))
import capture
from align import align_pair
from run_e2 import load_model, prefill, clone_cache, greedy_decode, first_line
from run_e2c import build_text as e2_build, find_token_span
import scenarios as S


def tool(line):
    m = re.search(r"([A-Za-z_]\w*)\s*\(", line)
    return m.group(1) if m else (line.split() or [""])[0]


def decide(model, tok, cache, last, upto, eos):
    return tool(first_line(tok, greedy_decode(model, clone_cache(cache, upto), last, upto, 40, eos)))


@torch.no_grad()
def recompute_span_inplace(model, work, new_ids, s, e):
    """Recompute KV for [s,e) against the CURRENT working cache[0:s]; write in place."""
    if e <= s:
        return 0
    pref = clone_cache(work, s)
    capture.enable_capture(to_cpu=False)
    model(input_ids=new_ids[:, s:e].to("cuda"), past_key_values=pref,
          cache_position=torch.arange(s, e, device="cuda"), use_cache=True)
    capture.disable_capture()
    cap = capture.get_capture()
    for i in range(len(work.layers)):
        k, v = cap[i]["k"], cap[i]["v"]
        if k.dim() == 3:
            k, v = k[None], v[None]
        work.layers[i].keys[:, :, s:e, :] = k[:, :, s:e, :]
        work.layers[i].values[:, :, s:e, :] = v[:, :, s:e, :]
    return e - s


def probe(model, tok, scn, eos):
    s = S.SCENARIOS[scn]
    al = align_pair(tok, e2_build(scn, s["v_old"], 40, tok, True),
                    e2_build(scn, s["v_new"], 40, tok, True))
    a, b = al["field_span"]; T = al["seq_len"]; upto = T - 1; nid = al["new_ids"]; last = nid[0, upto]
    co = prefill(model, al["old_ids"]); cn = prefill(model, al["new_ids"])
    g0, g1 = find_token_span(tok, nid[0], s["gate"][:120])
    oracle = decide(model, tok, cn, last, upto, eos)
    print(f"\n=== {scn} [{s['cls']}]  oracle={oracle}  field[{a},{b}] gate[{g0},{g1}] T={T}")
    rows = []

    def run(label, spans):
        w = clone_cache(co, upto); n = 0
        for (s0, s1) in spans:
            n += recompute_span_inplace(model, w, nid, s0, s1)
        d = decide(model, tok, w, last, upto, eos)
        print(f"  {label:34s} {n:4d} tok ({n/T*100:4.1f}%) -> {d:30s} {'RECOVER' if d==oracle else ''}")
        rows.append((label, n, n / T, d == oracle))

    for K in [16, 32, 64, 128]:
        run(f"sparse field+gate+last{K}", [(a, b), (g0, g1), (max(g1, upto - K), upto)])
    run("contiguous field+[gate..end]", [(a, b), (g0, upto)])
    run("all-after-field field+[b..end]", [(a, b), (b, upto)])
    return rows


def main():
    tok, model = load_model("Qwen/Qwen2.5-7B-Instruct")
    eos = {tok.eos_token_id}
    for scn in ["safety_mode", "subscription_tier", "account_role"]:
        probe(model, tok, scn, eos)


if __name__ == "__main__":
    main()
