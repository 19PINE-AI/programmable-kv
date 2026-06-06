"""Corrected selective recompute: ALWAYS refresh the field token + the top-k most-affected downstream
tokens (by ATTENTION-DIFFERENCE), tested in BOTH reasoning and non-reasoning modes.

Fixes the earlier test (which omitted the field and so failed under reasoning, where the CoT re-reads
the stale field). The selection criterion is the user's original one: rank downstream tokens by how
much their attention distribution CHANGES under the field edit (profiled by comparing old vs new full
prefill). We always include the field span. Methods: stale / full / erratum (golden) / field-only
(in_place) / field+attn_diff@k / field+dec_attn@k. Non-reasoning = decision logit; reasoning = CoT.
Run: MECH_ATTN=eager python esys/selective_field.py --model Qwen/Qwen3-8B
"""
import argparse, os, sys, json
import torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2"))
from mech_suite import (load, clone, prefill, ftok, wilson, META, TOK_WORDS, build, step, decide, gen_cot_sample)
from align import align_pair
import scenarios as S


@torch.no_grad()
def full_attn(model, ids):
    out = model(input_ids=ids.to("cuda"), use_cache=True, output_attentions=True)
    return out.past_key_values, out.attentions      # attentions: tuple(layers)[1,heads,L,L]


@torch.no_grad()
def attn_diff_rank(att_o, att_n, a, dpos):
    """Per downstream position i: total change in its attention distribution (to 0..i) old->new."""
    chg = torch.zeros(dpos, device=att_o[0].device)
    for lo, ln in zip(att_o, att_n):
        d = (ln[0, :, :dpos, :dpos] - lo[0, :, :dpos, :dpos]).abs().sum(dim=(0, 2))   # [dpos]
        chg[:dpos] += d
    return sorted(range(a, dpos), key=lambda i: float(chg[i]), reverse=True)


@torch.no_grad()
def dec_attn_rank(att_o, last_row, a, dpos):
    att = torch.stack([x[0, :, -1, :] for x in att_o]).mean(1).mean(0)   # decision row mean heads/layers
    return sorted(range(a, dpos), key=lambda i: float(att[i]), reverse=True)


def patched(co, cn, positions, L):
    w = clone(co, L); p = torch.tensor(positions, device="cuda")
    for i in range(len(w.layers)):
        w.layers[i].keys[:, :, p, :] = cn.layers[i].keys[:, :, p, :]
        w.layers[i].values[:, :, p, :] = cn.layers[i].values[:, :, p, :]
    return w


@torch.no_grad()
def dec_nonreason(model, cache, last, dpos, toi):
    return decide(step(model, clone(cache, dpos), last, dpos).logits[0, -1].float(), toi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default=None); ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--k", type=int, default=32); ap.add_argument("--max_new", type=int, default=400)
    ap.add_argument("--scns", default="account_role,safety_mode"); ap.add_argument("--oids", default="A4471,B8820")
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok, model = load(args.model)
    K = args.k
    METHODS = ["stale", "full", "erratum", "field_only", f"field+attndiff@{K}", f"field+decattn@{K}"]
    res = {mode: {m: 0 for m in METHODS} for mode in ["non_reasoning", "reasoning"]}
    nr_n = 0; rr_n = 0
    for scn in args.scns.split(","):
        m = META[scn]
        toi = {"safe": ftok(tok, TOK_WORDS[m["safe"]]), "unsafe": ftok(tok, TOK_WORDS[m["unsafe"]]), "lookup": ftok(tok, "lookup")}
        for oid in args.oids.split(","):
            # ----- build (non-reasoning uses force_suffix; reasoning uses thinking) -----
            for mode in ["non_reasoning", "reasoning"]:
                think = (mode == "reasoning")
                t_old = build(tok, scn, oid, m["vold"], think, not think)
                t_new = build(tok, scn, oid, m["vnew"], think, not think)
                oid_ids = torch.tensor([tok(t_old, add_special_tokens=False)["input_ids"]])
                nid_ids = torch.tensor([tok(t_new, add_special_tokens=False)["input_ids"]])
                al = align_pair(tok, t_old, t_new); a, b = al["field_span"]; L = oid_ids.shape[1]; dpos = L - 1
                co, att_o = full_attn(model, oid_ids); cn, att_n = full_attn(model, nid_ids)
                fld = list(range(a, b))
                ad = attn_diff_rank(att_o, att_n, a, dpos)
                da = dec_attn_rank(att_o, None, a, dpos)
                sets = {"stale": [], "full": list(range(a, dpos)), "field_only": fld,
                        f"field+attndiff@{K}": fld + [p for p in ad if p not in range(a, b)][:K],
                        f"field+decattn@{K}": fld + [p for p in da if p not in range(a, b)][:K]}
                # erratum cache (append-at-end), built separately
                err_body = S.build(scn, m["vold"], 30, erratum_value=m["vnew"]).replace("A4471", oid)
                err_full = (tok.apply_chat_template([{"role": "user", "content": err_body}], tokenize=False,
                            add_generation_prompt=True, enable_thinking=think) + ("" if think else "tool_call:"))
                eid = torch.tensor([tok(err_full, add_special_tokens=False)["input_ids"]]); Le = eid.shape[1]
                ce = prefill(model, eid)
                if mode == "non_reasoning":
                    last = int(nid_ids[0, dpos])
                    for name, S_ in sets.items():
                        c = co if name == "stale" else patched(co, cn, S_, L)
                        lt = int(oid_ids[0, dpos]) if name == "stale" else last
                        res[mode][name] += (dec_nonreason(model, c, lt, dpos, toi) == "safe")
                    res[mode]["erratum"] += (decide(step(model, clone(ce, Le - 1), int(eid[0, Le - 1]), Le - 1).logits[0, -1].float(), toi) == "safe")
                    nr_n += 1
                else:
                    for s in range(args.K):
                        for name in ["full", "field_only", f"field+attndiff@{K}", f"field+decattn@{K}"]:
                            c = patched(co, cn, sets[name], L)
                            dq, dp, cache, _ = gen_cot_sample(model, tok, c, nid_ids, L, 100 + s + hash(name) % 50, max_new=args.max_new)
                            res[mode][name] += (decide(step(model, clone(cache, dp), dq, dp).logits[0, -1].float(), toi) == "safe")
                        dq, dp, cache, _ = gen_cot_sample(model, tok, ce, eid, Le, 500 + s, max_new=args.max_new)
                        res[mode]["erratum"] += (decide(step(model, clone(cache, dp), dq, dp).logits[0, -1].float(), toi) == "safe")
                        rr_n += 1
            print(f"  {scn}/{oid} done", flush=True)

    out = {"model": args.model, "k": K, "non_reasoning_n": nr_n, "reasoning_n": rr_n, "results": {}}
    print(f"\n==== FIELD + SELECTIVE (k={K}) — BOTH MODES — {args.model} ====")
    for mode, nn in [("non_reasoning", nr_n), ("reasoning", rr_n)]:
        print(f"  [{mode}] (n={nn})")
        out["results"][mode] = {}
        for mm in METHODS:
            if mode == "reasoning" and mm == "stale":
                continue
            pc = res[mode][mm] / nn if nn else 0
            out["results"][mode][mm] = {"P_correct": round(pc, 3), "ci": wilson(res[mode][mm], nn)}
            print(f"     {mm:22s} P_correct={pc:.2f} CI{wilson(res[mode][mm], nn)}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"selective_field_{tag}.json"), "w"), indent=2)
    print("SELECTIVE_FIELD_DONE")


if __name__ == "__main__":
    main()
