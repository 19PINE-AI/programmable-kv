"""Making editable KV work cheaply: the ERRATUM mechanism (+ optional thinking).

Instead of recomputing the stale downstream, we leave it stale and inject a short,
salient correction at the suffix: '[STATE UPDATE] <field> has changed to <new>'.
Only those ~tens of tokens are recomputed (contiguous suffix). The live decode
(and thinking, if enabled) reads the erratum with full recency and an explicit
override instruction -- an attention magnet the refreshed-but-not-salient KV lacks.

Methods (all decode the decision; --think adds CoT):
  oracle_new   full new prefill                         (ceiling)
  stale_full   old cache, nothing changed                (floor)
  field_only   old cache + field KV refreshed (exact)    (the cheap patch that failed w/o CoT)
  erratum      old cache (stale field) + suffix erratum   (recompute only the suffix)
  field+erratum field refreshed + suffix erratum
  hoist_to_end field moved to suffix                      (baseline to beat)
Reports tool-name agreement with oracle_new and recompute fraction.
"""
import argparse, json, os, sys, re
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2"))
import capture  # noqa
from align import align_pair, _common_prefix_len
from run_e2 import load_model, prefill, clone_cache, greedy_decode, first_line
import scenarios as S
import mechanism as M
from thinking_test import decode_think

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def chat(tok, content, think):
    return tok.apply_chat_template([{"role": "user", "content": content}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=think)


def tool_of(line):
    m = re.search(r"tool_call:\s*([A-Za-z_]\w*)\s*\(", line) or re.search(r"([A-Za-z_]\w*)\s*\(", line)
    return m.group(1) if m else ""


def decision(model, tok, cache, last, upto, think, max_new):
    c = clone_cache(cache, upto)
    if think:
        r = decode_think(model, tok, c, last, upto, max_new=max_new)
        return r["tool"], r.get("think_tokens")
    line = first_line(tok, greedy_decode(model, c, last, upto, max_new, {tok.eos_token_id}))
    return tool_of(line), None


def ids_of(tok, text):
    return torch.tensor([tok(text, add_special_tokens=False)["input_ids"]])


def run_one(tok, model, scn_key, n_neutral, think, max_new):
    s = S.SCENARIOS[scn_key]
    # texts (chat-wrapped)
    t_old = chat(tok, S.build(scn_key, s["v_old"], n_neutral), think)
    t_new = chat(tok, S.build(scn_key, s["v_new"], n_neutral), think)
    t_err = chat(tok, S.build(scn_key, s["v_old"], n_neutral, erratum_value=s["v_new"]), think)
    t_hoist = chat(tok, S.build(scn_key, s["v_new"], n_neutral, hoist=True), think)

    old_ids = ids_of(tok, t_old); new_ids = ids_of(tok, t_new); err_ids = ids_of(tok, t_err)
    # field span (length-preserving old vs new)
    al = align_pair(tok, t_old, t_new); a, b = al["field_span"]
    T = old_ids.shape[1]; upto = T - 1; last = old_ids[0, upto]

    co = prefill(model, old_ids); cn = prefill(model, new_ids)
    Tg = float(T)  # denominator for recompute fraction

    res = {}

    def rec(name, cache, declast, decupto, recompute_tokens):
        tool, think_tok = decision(model, tok, cache, declast, decupto, think, max_new)
        res[name] = {"tool": tool, "recompute_frac": recompute_tokens / Tg,
                     "think_tok": think_tok}

    # oracle_new / stale / field_only
    rec("oracle_new", cn, new_ids[0, upto], upto, T)
    rec("stale_full", co, last, upto, 0)
    fcache, _ = M.patchkv_cache(model, new_ids, co, (a, b), 0, upto)  # exact field refresh
    rec("field_only", fcache, last, upto, (b - a))

    # erratum: old cache truncated at insertion point p, recompute [p..upto_err]
    p = _common_prefix_len(old_ids[0].tolist(), err_ids[0].tolist())
    upto_e = err_ids.shape[1] - 1; last_e = err_ids[0, upto_e]
    work, _ = M.recompute_suffix(model, err_ids, co, p, upto_e)  # stale[0:p] + recomputed suffix
    rec("erratum", work, last_e, upto_e, (upto_e - p))

    # field+erratum: refresh field KV (exact, copy from cn) inside the kept prefix, then erratum suffix
    base = clone_cache(co, p)
    for i in range(len(base.layers)):
        base.layers[i].keys[:, :, a:b, :] = cn.layers[i].keys[:, :, a:b, :]
        base.layers[i].values[:, :, a:b, :] = cn.layers[i].values[:, :, a:b, :]
    work2, _ = M.recompute_suffix(model, err_ids, base, p, upto_e)
    rec("field_erratum", work2, last_e, upto_e, (b - a) + (upto_e - p))

    # hoist
    hoist_ids = ids_of(tok, t_hoist)
    hal_old = chat(tok, S.build(scn_key, s["v_old"], n_neutral, hoist=True), think)
    hal = align_pair(tok, hal_old, t_hoist); hs, he = hal["field_span"]
    hT = hoist_ids.shape[1]; hupto = hT - 1
    hco = prefill(model, ids_of(tok, hal_old))
    hwork, _ = M.recompute_suffix(model, hoist_ids, clone_cache(hco, hupto), hs, hupto)
    rec("hoist_to_end", hwork, hoist_ids[0, hupto], hupto, (hupto - hs))

    oracle = res["oracle_new"]["tool"]
    for k in res:
        res[k]["recovers"] = res[k]["tool"] == oracle
    return {"scenario": scn_key, "cls": s["cls"], "think": think, "seq_len": T,
            "field_span": [a, b], "erratum_insert_p": p, "oracle": oracle, "methods": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--n_neutral", type=int, default=40)
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--scenarios", default="account_role,safety_mode,subscription_tier,timestamp,request_id")
    args = ap.parse_args()
    tok, model = load_model(args.model)
    recs = []
    for k in args.scenarios.split(","):
        r = run_one(tok, model, k, args.n_neutral, args.think, args.max_new)
        recs.append(r)
        print(f"\n=== {k} [{r['cls']}] think={r['think']} oracle={r['oracle']}")
        for name, m in r["methods"].items():
            print(f"  {name:14s} recompute={m['recompute_frac']*100:5.1f}%  recovers={int(m['recovers'])}  tool={m['tool']}")
    json.dump(recs, open(os.path.join(RES, f"erratum_{args.tag}.json"), "w"), indent=2)
    print("\nwrote", os.path.join(RES, f"erratum_{args.tag}.json"))


if __name__ == "__main__":
    main()
