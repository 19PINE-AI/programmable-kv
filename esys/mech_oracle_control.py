"""Oracle-controlled reasoning study: isolate the cache-EDIT penalty from competence,
and test erratum-at-scale.

Per model, per instance, K stochastic CoT samples for THREE conditions (reasoning):
  oracle     = full new prefill                 -> model COMPETENCE ceiling
  field_only = field-token KV swapped, rest stale
  erratum    = stale + appended suffix override
Edit penalty = oracle_safe - field_only_safe (isolates the edit, not the model).
Erratum recovery = does erratum match/approach oracle where field_only fails?
Reports safe / unsafe / hedge rates with Wilson CIs.
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2"))
from mech_suite import (load, clone, prefill, step, decide, ftok, wilson, META, TOK_WORDS,
                        fieldonly_cache, gen_cot_sample, build, ORDER_IDS)
from align import align_pair, _common_prefix_len
import scenarios as S
from collections import Counter


def erratum_prompt(tok, scn, oid, vold, vnew):
    body = S.build(scn, vold, 30, erratum_value=vnew).replace("A4471", oid)
    return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)


def decision_of(model, tok, cache, ids, L, seed, toi):
    dq, dpos, c, _ = gen_cot_sample(model, tok, cache, ids, L, seed)
    return decide(step(model, clone(c, dpos), dq, dpos).logits[0, -1].float(), toi)


def run(model, tok, scn, oid, K, seed0):
    m = META[scn]
    toi = {"safe": ftok(tok, TOK_WORDS[m["safe"]]), "unsafe": ftok(tok, TOK_WORDS[m["unsafe"]]),
           "lookup": ftok(tok, "lookup")}
    t_old = build(tok, scn, oid, m["vold"], True, False)
    t_new = build(tok, scn, oid, m["vnew"], True, False)
    oid_ids = torch.tensor([tok(t_old, add_special_tokens=False)["input_ids"]])
    nid_ids = torch.tensor([tok(t_new, add_special_tokens=False)["input_ids"]])
    al = align_pair(tok, t_old, t_new); a, b = al["field_span"]; L = oid_ids.shape[1]
    fc, cn = fieldonly_cache(model, oid_ids, nid_ids, a, b, L)
    # erratum cache
    t_err = erratum_prompt(tok, scn, oid, m["vold"], m["vnew"])
    eid = torch.tensor([tok(t_err, add_special_tokens=False)["input_ids"]])
    co = prefill(model, oid_ids)
    p = _common_prefix_len(oid_ids[0].tolist(), eid[0].tolist()); Le = eid.shape[1]
    ew = clone(co, p)
    if Le - 1 > p:
        model(input_ids=eid[:, p:Le - 1].to("cuda"), past_key_values=ew,
              cache_position=torch.arange(p, Le - 1, device="cuda"), use_cache=True)
    # ew now length Le-1; gen_cot_sample expects cache of length L-1 then feeds ids[L-1]
    # build a length-Le cache by also adding token Le-1? simpler: re-prefill full erratum then it'll clone to Le-1
    ew_full = prefill(model, eid)
    out = {"oracle": [], "field_only": [], "erratum": []}
    for s in range(K):
        out["oracle"].append(decision_of(model, tok, cn, nid_ids, L, seed0 + s, toi))
        out["field_only"].append(decision_of(model, tok, fc, nid_ids, L, seed0 + s, toi))
        out["erratum"].append(decision_of(model, tok, ew_full, eid, Le, seed0 + s, toi))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--instances", type=int, default=4)
    ap.add_argument("--K", type=int, default=4)
    args = ap.parse_args()
    tok, model = load(args.model)
    insts = [(scn, oid) for scn in META for oid in ORDER_IDS][:args.instances]
    agg = {c: [] for c in ["oracle", "field_only", "erratum"]}
    for j, (scn, oid) in enumerate(insts):
        r = run(model, tok, scn, oid, args.K, 6000 + 50 * j)
        for c in agg:
            agg[c].extend(r[c])
        print(f"[{scn}/{oid}] " + " ".join(
            f"{c}:safe{sum(x=='safe' for x in r[c])}/uns{sum(x=='unsafe' for x in r[c])}" for c in agg), flush=True)
    res = {"model": args.model, "n": len(agg["oracle"])}
    for c in agg:
        n = len(agg[c]); ks = sum(x == "safe" for x in agg[c]); ku = sum(x == "unsafe" for x in agg[c])
        res[c] = {"P_safe": round(ks / n, 2), "ci_safe": wilson(ks, n),
                  "P_unsafe": round(ku / n, 2), "dist": dict(Counter(agg[c]))}
    res["edit_penalty_safe"] = round(res["oracle"]["P_safe"] - res["field_only"]["P_safe"], 2)
    res["erratum_recovers"] = round(res["erratum"]["P_safe"] - res["field_only"]["P_safe"], 2)
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"mech_oracle_{args.tag}.json"), "w"), indent=2)
    print(f"\n=== {args.model} (n={res['n']}) ===")
    print(f"oracle  P_safe={res['oracle']['P_safe']} {res['oracle']['ci_safe']} unsafe={res['oracle']['P_unsafe']}")
    print(f"field_only P_safe={res['field_only']['P_safe']} {res['field_only']['ci_safe']} unsafe={res['field_only']['P_unsafe']}")
    print(f"erratum P_safe={res['erratum']['P_safe']} {res['erratum']['ci_safe']} unsafe={res['erratum']['P_unsafe']}")
    print(f"EDIT PENALTY (oracle-field_only) = {res['edit_penalty_safe']}; ERRATUM RECOVERS = {res['erratum_recovers']}")
    print("ORACLE_CONTROL_DONE", flush=True)


if __name__ == "__main__":
    main()
