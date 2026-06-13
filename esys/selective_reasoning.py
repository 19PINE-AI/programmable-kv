"""Selective recompute (decision-attention) under REASONING vs the golden erratum — thinking models.

For a thinking model (Qwen3), on the e2 gating scenarios, K stochastic CoT samples: build the
selective-recompute cache (patch the top-k decision-attention downstream positions from the NEW
prefill into the stale OLD cache), generate a CoT, and decide. Compare P(correct=safe) to the golden
ERRATUM (append-at-end + CoT) and to full reprefill + CoT. Tests whether selective recompute also
works once the CoT re-reads the refreshed tokens. Run: MECH_ATTN=eager python esys/selective_reasoning.py
"""
import argparse, os, sys, json
import torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2"))
from mech_suite import (load, clone, prefill, ftok, wilson, META, TOK_WORDS, build, step, decide, gen_cot_sample)
from align import align_pair
import scenarios as S

KS = [32]


@torch.no_grad()
def dattn_rank(model, co, last, dpos, a):
    out = step(model, clone(co, dpos), last, dpos)             # output_attentions on (keys=None)
    att = torch.stack([x[0] for x in out.attentions])[:, :, -1, :].mean(1).mean(0)
    return sorted(range(a, dpos), key=lambda i: float(att[i]), reverse=True)


def patched_cache(co, cn, positions, L):
    w = clone(co, L); p = torch.tensor(positions, device="cuda")
    for i in range(len(w.layers)):
        w.layers[i].keys[:, :, p, :] = cn.layers[i].keys[:, :, p, :]
        w.layers[i].values[:, :, p, :] = cn.layers[i].values[:, :, p, :]
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--scns", default="account_role,safety_mode")
    ap.add_argument("--oids", default="A4471,B8820")
    ap.add_argument("--max_new", type=int, default=420)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok, model = load(args.model)
    methods = ["full", "erratum", "in_place(field)"] + [f"selective@{k}(no field)" for k in KS] + [f"field+selective@{k}" for k in KS]
    corr = {m: 0 for m in methods}; ntot = 0
    for scn in args.scns.split(","):
        m = META[scn]
        toi = {"safe": ftok(tok, TOK_WORDS[m["safe"]]), "unsafe": ftok(tok, TOK_WORDS[m["unsafe"]]),
               "lookup": ftok(tok, "lookup")}
        for oid in args.oids.split(","):
            t_old = build(tok, scn, oid, m["vold"], True, False)         # thinking ON
            t_new = build(tok, scn, oid, m["vnew"], True, False)
            t_err = build(tok, scn, oid, m["vold"], True, False)         # erratum text (insert before TASK)
            oid_ids = torch.tensor([tok(t_old, add_special_tokens=False)["input_ids"]])
            nid_ids = torch.tensor([tok(t_new, add_special_tokens=False)["input_ids"]])
            al = align_pair(tok, t_old, t_new); a, b = al["field_span"]; L = oid_ids.shape[1]
            co = prefill(model, oid_ids); cn = prefill(model, nid_ids)
            order = dattn_rank(model, co, int(oid_ids[0, L - 1]), L - 1, a)
            # erratum cache: old text with the [STATE UPDATE] inserted (S.build erratum_value), prefill
            err_text = build(tok, scn, oid, m["vold"], True, False)
            err_body = S.build(scn, m["vold"], 30, erratum_value=m["vnew"]).replace("A4471", oid)
            err_full = tok.apply_chat_template([{"role": "user", "content": err_body}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=True)
            eid = torch.tensor([tok(err_full, add_special_tokens=False)["input_ids"]]); Le = eid.shape[1]
            ce = prefill(model, eid)
            for s in range(args.K):
                # full + CoT (oracle)
                dq, dpos, cache, _ = gen_cot_sample(model, tok, cn, nid_ids, L, 100 + s, max_new=args.max_new)
                corr["full"] += (decide(step(model, clone(cache, dpos), dq, dpos).logits[0, -1].float(), toi) == "safe")
                # erratum + CoT (golden)
                dq, dpos, cache, _ = gen_cot_sample(model, tok, ce, eid, Le, 200 + s, max_new=args.max_new)
                corr["erratum"] += (decide(step(model, clone(cache, dpos), dq, dpos).logits[0, -1].float(), toi) == "safe")
                # in_place (FIELD only) + CoT  -- the field is what the CoT re-reads
                pc = patched_cache(co, cn, list(range(a, b)), L)
                dq, dpos, cache, _ = gen_cot_sample(model, tok, pc, nid_ids, L, 250 + s, max_new=args.max_new)
                corr["in_place(field)"] += (decide(step(model, clone(cache, dpos), dq, dpos).logits[0, -1].float(), toi) == "safe")
                for k in KS:
                    # selective@k WITHOUT the field (the original, broken-for-CoT test)
                    pc = patched_cache(co, cn, order[:k], L)
                    dq, dpos, cache, _ = gen_cot_sample(model, tok, pc, nid_ids, L, 300 + s + k, max_new=args.max_new)
                    corr[f"selective@{k}(no field)"] += (decide(step(model, clone(cache, dpos), dq, dpos).logits[0, -1].float(), toi) == "safe")
                    # FIELD + selective@k (the corrected test: include the field the CoT re-reads)
                    pc = patched_cache(co, cn, list(range(a, b)) + order[:k], L)
                    dq, dpos, cache, _ = gen_cot_sample(model, tok, pc, nid_ids, L, 350 + s + k, max_new=args.max_new)
                    corr[f"field+selective@{k}"] += (decide(step(model, clone(cache, dpos), dq, dpos).logits[0, -1].float(), toi) == "safe")
                ntot += 1
            print(f"  {scn}/{oid} done ({ntot} samples)", flush=True)
    out = {"model": args.model, "n_samples": ntot, "methods": {}}
    print(f"\n==== SELECTIVE vs ERRATUM under REASONING ({args.model}, n={ntot}) ====")
    for mm in methods:
        pc = corr[mm] / ntot if ntot else 0
        out["methods"][mm] = {"P_correct": round(pc, 3), "ci": wilson(corr[mm], ntot)}
        print(f"  {mm:14s} P_correct={pc:.2f} CI{wilson(corr[mm], ntot)}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"selective_reasoning_{tag}.json"), "w"), indent=2)
    print("SELECTIVE_REASONING_DONE")


if __name__ == "__main__":
    main()
