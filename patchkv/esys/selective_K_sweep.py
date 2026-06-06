"""K-sweep: minimal K for which "field + selective@K" recovers the REASONING decision at golden level.

Practical surgical-editing question: under reasoning, always refresh the field token + the top-K
downstream tokens (ranked by decision-attention). For K in {0,4,8,16,32,64} measure P(safe/correct)
after a CoT, and find the smallest K that matches the golden append-at-end ERRATUM. Swept over e2
gating domains x surface-variant prompts (order ids) x K stochastic CoT samples, per model.
Run: MECH_ATTN=eager python esys/selective_K_sweep.py --model Qwen/Qwen3-8B
"""
import argparse, os, sys, json
import torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2"))
from mech_suite import (load, clone, prefill, ftok, wilson, META, TOK_WORDS, build, step, decide, gen_cot_sample)
from align import align_pair
import scenarios as S

KLIST = [0, 4, 8, 16, 32, 64]


@torch.no_grad()
def dattn_rank(model, co, last, dpos, a):
    att = torch.stack([x[0, :, -1, :] for x in step(model, clone(co, dpos), last, dpos).attentions]).mean(1).mean(0)
    return sorted(range(a, dpos), key=lambda i: float(att[i]), reverse=True)


def patched(co, cn, positions, L):
    w = clone(co, L)
    if positions:
        p = torch.tensor(positions, device="cuda")
        for i in range(len(w.layers)):
            w.layers[i].keys[:, :, p, :] = cn.layers[i].keys[:, :, p, :]
            w.layers[i].values[:, :, p, :] = cn.layers[i].values[:, :, p, :]
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B"); ap.add_argument("--tag", default=None)
    ap.add_argument("--K", type=int, default=3); ap.add_argument("--max_new", type=int, default=340)
    ap.add_argument("--scns", default="account_role,safety_mode,subscription_tier")
    ap.add_argument("--oids", default="A4471,B8820")
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok, model = load(args.model)
    METH = [f"field+sel@{k}" for k in KLIST] + ["erratum", "full"]
    safe = {m: 0 for m in METH}; n = 0
    for scn in args.scns.split(","):
        m = META[scn]
        toi = {"safe": ftok(tok, TOK_WORDS[m["safe"]]), "unsafe": ftok(tok, TOK_WORDS[m["unsafe"]]), "lookup": ftok(tok, "lookup")}
        for oid in args.oids.split(","):
            t_old = build(tok, scn, oid, m["vold"], True, False); t_new = build(tok, scn, oid, m["vnew"], True, False)
            oid_ids = torch.tensor([tok(t_old, add_special_tokens=False)["input_ids"]])
            nid_ids = torch.tensor([tok(t_new, add_special_tokens=False)["input_ids"]])
            al = align_pair(tok, t_old, t_new); a, b = al["field_span"]; L = oid_ids.shape[1]
            co = prefill(model, oid_ids); cn = prefill(model, nid_ids)
            order = dattn_rank(model, co, int(oid_ids[0, L - 1]), L - 1, a)
            fld = list(range(a, b)); extra = [p for p in order if p not in range(a, b)]
            err_body = S.build(scn, m["vold"], 30, erratum_value=m["vnew"]).replace("A4471", oid)
            err_full = tok.apply_chat_template([{"role": "user", "content": err_body}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=True)
            eid = torch.tensor([tok(err_full, add_special_tokens=False)["input_ids"]]); Le = eid.shape[1]
            ce = prefill(model, eid)
            for s in range(args.K):
                for k in KLIST:
                    pc = patched(co, cn, fld + extra[:k], L)
                    dq, dp, cache, _ = gen_cot_sample(model, tok, pc, nid_ids, L, 17 + s * 7 + k, max_new=args.max_new)
                    safe[f"field+sel@{k}"] += (decide(step(model, clone(cache, dp), dq, dp).logits[0, -1].float(), toi) == "safe")
                dq, dp, cache, _ = gen_cot_sample(model, tok, ce, eid, Le, 900 + s, max_new=args.max_new)
                safe["erratum"] += (decide(step(model, clone(cache, dp), dq, dp).logits[0, -1].float(), toi) == "safe")
                dq, dp, cache, _ = gen_cot_sample(model, tok, cn, nid_ids, L, 800 + s, max_new=args.max_new)
                safe["full"] += (decide(step(model, clone(cache, dp), dq, dp).logits[0, -1].float(), toi) == "safe")
                n += 1
            print(f"  {scn}/{oid} done ({n})", flush=True)
    out = {"model": args.model, "n": n, "K_safe": {}}
    er = safe["erratum"] / n if n else 0
    kstar = None
    for k in KLIST:
        p = safe[f"field+sel@{k}"] / n if n else 0
        out["K_safe"][k] = {"P_safe": round(p, 3), "ci": wilson(safe[f"field+sel@{k}"], n)}
        if kstar is None and p >= er - 1e-9:
            kstar = k
    out["erratum_P_safe"] = round(er, 3); out["full_P_safe"] = round(safe["full"] / n, 3); out["K_star"] = kstar
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"selective_Ksweep_{tag}.json"), "w"), indent=2)
    print(f"\n==== K-SWEEP (field+selective under REASONING) — {args.model} (n={n}) ====")
    for k in KLIST:
        print(f"  field+sel@{k:<3d} P_safe={out['K_safe'][k]['P_safe']:.2f} CI{out['K_safe'][k]['ci']}")
    print(f"  erratum(golden)={er:.2f}  full={out['full_P_safe']:.2f}  => K* (min K matching golden) = {kstar}")
    print("KSWEEP_DONE")


if __name__ == "__main__":
    main()
