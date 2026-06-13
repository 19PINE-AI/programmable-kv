"""Phase D verification: faithful recompute, and the field-placement hypothesis.

(1) field-refresh exactness: recompute_field_inplace must equal the full-new-prefill
    KV on the field span (H2 free lunch).
(2) early-gated field (synthetic safety_mode): faithful recency-only recompute FAILS;
    faithful refresh of field + [gate_start .. end] RECOVERS (but costs more, because
    the gate is early). Contrast with oracle-copy recency (recovers cheaply but is not
    realizable).
(3) late-placed field (tau-bench order status): the gating rules precede the field
    (causally exact), so faithful recompute of just the small post-field tail recovers
    cheaply -- the regime where PatchKV genuinely wins.
"""
import os, sys, re
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2"))
import capture  # noqa
from align import align_pair
from run_e2 import load_model, prefill, clone_cache, greedy_decode, first_line
from run_e2c import refresh_spans, find_token_span, build_text as e2_build
import scenarios as S
import taubench_ctx as TBC
import mechanism as M


def tool(line):
    m = re.search(r"([A-Za-z_]\w*)\s*\(", line)
    return m.group(1) if m else (line.strip().split() or [""])[0]


def decode_line(model, tok, cache, last, upto, eos, mx=40):
    return first_line(tok, greedy_decode(model, clone_cache(cache, upto), last, upto, mx, eos))


def main():
    tok, model = load_model("Qwen/Qwen2.5-7B-Instruct")
    eos = {tok.eos_token_id}

    # ---------- (1) field-refresh exactness ----------
    print("== (1) faithful field-refresh exactness (vs full-new-prefill KV) ==")
    s = S.SCENARIOS["safety_mode"]
    ot = e2_build("safety_mode", s["v_old"], 40, tok, True)
    nt = e2_build("safety_mode", s["v_new"], 40, tok, True)
    al = align_pair(tok, ot, nt); a, b = al["field_span"]; T = al["seq_len"]; upto = T - 1
    co = prefill(model, al["old_ids"]); cn = prefill(model, al["new_ids"])
    work, _ = M.recompute_field_inplace(model, al["new_ids"], co, (a, b))
    maxerr = 0.0
    for i in range(len(work.layers)):
        e1 = (work.layers[i].keys[:, :, a:b, :].float() - cn.layers[i].keys[:, :, a:b, :].float()).abs().max().item()
        maxerr = max(maxerr, e1)
    print(f"   max|K_faithful - K_oracle| on field span = {maxerr:.3e}  (should be ~0)")

    # ---------- (2) early-gated field: recency vs field+gate, faithful vs oracle-copy ----------
    print("\n== (2) safety_mode (EARLY gate): faithful recency vs field+gate; vs oracle-copy ==")
    last = al["new_ids"][0, upto]
    oracle = tool(decode_line(model, tok, cn, last, upto, eos))
    gate = find_token_span(tok, al["new_ids"][0], s["gate"][:120])
    print(f"   oracle decision tool = {oracle};  field@[{a},{b}] gate@{gate}")
    # faithful recency k=128
    c, n = M.patchkv_cache(model, al["new_ids"], co, (a, b), 128, upto)
    print(f"   faithful recency k=128 ({n}tok): {tool(decode_line(model,tok,c,last,upto,eos))}  recover={tool(decode_line(model,tok,c,last,upto,eos))==oracle}")
    # faithful field + recompute [gate_start .. upto]  (refresh the early gate and all after)
    cg, _ = M.recompute_field_inplace(model, al["new_ids"], co, (a, b))
    cg, ntail = M.recompute_suffix(model, al["new_ids"], cg, gate[0], upto)
    print(f"   faithful field+[gate..end] ({(b-a)+ntail}tok, {((b-a)+ntail)/T*100:.0f}%): "
          f"{tool(decode_line(model,tok,cg,last,upto,eos))}  recover={tool(decode_line(model,tok,cg,last,upto,eos))==oracle}")
    # oracle-copy recency k=128 (not realizable, upper bound)
    cc = refresh_spans(co, cn, [(a, b), (max(b, upto - 128), upto)], upto)
    print(f"   oracle-COPY recency k=128 (unrealizable): {tool(decode_line(model,tok,cc,last,upto,eos))}  "
          f"recover={tool(decode_line(model,tok,cc,last,upto,eos))==oracle}")

    # ---------- (3) late-placed field (tau-bench): faithful tail recompute ----------
    print("\n== (3) tau-bench order_status (LATE field, gates precede it): faithful tail recompute ==")
    h_ot = tok.apply_chat_template([{"role": "user", "content": TBC.build("order_status_cancel", "pending")}],
                                   tokenize=False, add_generation_prompt=True)
    h_nt = tok.apply_chat_template([{"role": "user", "content": TBC.build("order_status_cancel", "delivered")}],
                                   tokenize=False, add_generation_prompt=True)
    tal = align_pair(tok, h_ot, h_nt); ta, tb = tal["field_span"]; tT = tal["seq_len"]; tupto = tT - 1
    tco = prefill(model, tal["old_ids"]); tcn = prefill(model, tal["new_ids"])
    tlast = tal["new_ids"][0, tupto]
    toracle = tool(decode_line(model, tok, tcn, tlast, tupto, eos))
    tstale = tool(decode_line(model, tok, tco, tal["old_ids"][0, tupto], tupto, eos))
    print(f"   field@[{ta},{tb}] of T={tT}; oracle(delivered)={toracle}  old(pending)={tstale}")
    # faithful PatchKV: exact field refresh + recompute the post-field tail [b..upto]
    cpt, nf = M.recompute_field_inplace(model, tal["new_ids"], tco, (ta, tb))
    cpt, ntail = M.recompute_suffix(model, tal["new_ids"], cpt, tb, tupto)
    cost = nf + ntail
    dec = tool(decode_line(model, tok, cpt, tlast, tupto, eos))
    print(f"   faithful field+tail ({cost}tok, {cost/tT*100:.1f}% recompute): {dec}  recover={dec==toracle}")
    print(f"   exact-reusable prefix (before field) = {ta}/{tT} = {ta/tT*100:.1f}% reused for FREE")


if __name__ == "__main__":
    main()
