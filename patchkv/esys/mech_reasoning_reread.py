"""D4 — The reasoning-axis mechanism: does the CoT *re-read* the field?

The headline behavioral finding is that reasoning models tolerate the cheap in_place edit
where non-reasoning models revert to stale. HYPOTHESIS: the chain-of-thought re-reads and
re-derives the field AFTER it, so an in_place-refreshed field token gets re-consumed by the
freshly-generated CoT tokens — which the decision then reads. If true, blocking the CoT's
attention to the field (during generation) should make the in_place benefit collapse, while
blocking the *decision's* direct read of the field should NOT (the CoT already carried it).

Setup: reasoning ON, in_place cache (field=NEW value, all downstream KV stale=OLD). Generate
a CoT, then the decision. Conditions (P(correct = NEW action), instances x K samples):
  inplace_base        : CoT + decision, no masking                      -> expect CORRECT
  block_cot_field (A) : mask every CoT-gen query's attn to field span   -> expect REVERT(old)
  block_dec_field (B) : CoT normal; mask only the DECISION->field        -> expect CORRECT
  block_cot_gate (ctl): mask CoT-gen attn to a same-width gate band      -> expect CORRECT
Plus references: oracle (full NEW) and stale (full OLD).
Also measures mean CoT->field attention mass vs CoT->gate band (eager attentions).

A>collapse while B,ctl hold => the CoT re-read is the causal carrier of reasoning robustness.
Run: MECH_ATTN=eager python esys/mech_reasoning_reread.py [--K 4 --oids A4471,B8820]
"""
import argparse, json, os, sys, math
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2"))
from mech_suite import (load, clone, prefill, ftok, wilson, META, TOK_WORDS, build, step,
                        fieldonly_cache, decide, _KO)
from align import align_pair
import scenarios as S


def gate_band(tok, text, a, b, scn):
    """A same-width control band located at the gating rule (a stale downstream region)."""
    w = b - a
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    c = text.find(S.SCENARIOS[scn]["gate"][:24])
    g = b
    if c >= 0:
        for ti, (s, e) in enumerate(offs):
            if s <= c < e or s >= c:
                g = ti; break
    return list(range(g, g + w))


@torch.no_grad()
def gen_cot(model, tok, fc, nid, L, seed, mask_keys=None, measure_field=None,
            measure_gate=None, temp=0.7, max_new=1200):
    """Generate a CoT from the in_place cache. If mask_keys given, every generated query's
    attention to those key positions is knocked out. Optionally accumulate mean attention
    mass to field/gate key sets (requires eager + attentions, i.e. mask_keys is None)."""
    g = torch.Generator(device="cuda"); g.manual_seed(seed)
    cache = clone(fc, L - 1); cur = int(nid[0, L - 1]); pos = L - 1; gen = []
    eos = tok.eos_token_id
    fmass = []; gmass = []
    for _ in range(max_new):
        out = step(model, cache, cur, pos, keys=mask_keys)   # keys=None -> attentions returned
        if mask_keys is None and measure_field is not None and out.attentions is not None:
            att = torch.stack([x[0] for x in out.attentions])[:, :, -1, :].mean(1)  # [Lyr, pos+1]
            a_mean = att.mean(0)                                                      # over layers
            fmass.append(float(a_mean[measure_field].sum()))
            gmass.append(float(a_mean[measure_gate].sum()))
        pos += 1
        p = torch.softmax(out.logits[0, -1].float() / temp, -1)
        nx = int(torch.multinomial(p, 1, generator=g)); gen.append(nx); cur = nx
        if "</think>" in tok.decode(gen) or nx == eos:
            break
    cot_end = pos
    scaffold = tok("\ntool_call:", add_special_tokens=False)["input_ids"]
    for t in [cur] + scaffold[:-1]:
        step(model, cache, t, pos, keys=mask_keys); pos += 1
    return scaffold[-1], pos, cache, cot_end, fmass, gmass


@torch.no_grad()
def run_instance(model, tok, scn, oid, K, seed0):
    m = META[scn]
    toi = {"safe": ftok(tok, TOK_WORDS[m["safe"]]), "unsafe": ftok(tok, TOK_WORDS[m["unsafe"]]),
           "lookup": ftok(tok, "lookup")}
    t_old = build(tok, scn, oid, m["vold"], True, False)   # thinking ON
    t_new = build(tok, scn, oid, m["vnew"], True, False)
    oid_ids = torch.tensor([tok(t_old, add_special_tokens=False)["input_ids"]])
    nid_ids = torch.tensor([tok(t_new, add_special_tokens=False)["input_ids"]])
    al = align_pair(tok, t_old, t_new); a, b = al["field_span"]; L = oid_ids.shape[1]
    fld = list(range(a, b)); gate = gate_band(tok, t_old, a, b, scn)
    fc, cn = fieldonly_cache(model, oid_ids, nid_ids, a, b, L)        # field NEW, downstream stale

    out = {"inplace_base": [], "block_cot_field": [], "block_dec_field": [],
           "block_cot_gate": [], "fmass": [], "gmass": []}
    for s in range(K):
        sd = seed0 + s
        # inplace_base (+ attention measurement)
        dq, dpos, cache, ce, fmass, gmass = gen_cot(model, tok, fc, nid_ids, L, sd,
                                                    measure_field=fld, measure_gate=gate)
        out["inplace_base"].append(decide(step(model, clone(cache, dpos), dq, dpos).logits[0, -1].float(), toi))
        if fmass:
            out["fmass"].append(sum(fmass) / len(fmass)); out["gmass"].append(sum(gmass) / len(gmass))
        # block_dec_field (B): same CoT, mask only decision -> field
        out["block_dec_field"].append(decide(step(model, clone(cache, dpos), dq, dpos, keys=fld).logits[0, -1].float(), toi))
        # block_cot_field (A): regenerate CoT with field masked for every gen query
        dqA, dposA, cacheA, _, _, _ = gen_cot(model, tok, fc, nid_ids, L, sd, mask_keys=fld)
        out["block_cot_field"].append(decide(step(model, clone(cacheA, dposA), dqA, dposA).logits[0, -1].float(), toi))
        # block_cot_gate (control): regenerate with gate band masked
        dqG, dposG, cacheG, _, _, _ = gen_cot(model, tok, fc, nid_ids, L, sd, mask_keys=gate)
        out["block_cot_gate"].append(decide(step(model, clone(cacheG, dposG), dqG, dposG).logits[0, -1].float(), toi))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--scns", default="account_role,safety_mode,subscription_tier")
    ap.add_argument("--oids", default="A4471,B8820")
    ap.add_argument("--K", type=int, default=4)
    args = ap.parse_args()
    os.environ.setdefault("MECH_ATTN", "eager")     # knockout + attentions need eager
    tok, model = load(args.model)
    conds = ["inplace_base", "block_cot_field", "block_dec_field", "block_cot_gate"]
    tally = {c: [] for c in conds}; fmass = []; gmass = []
    for scn in args.scns.split(","):
        for oid in args.oids.split(","):
            r = run_instance(model, tok, scn, oid, args.K, seed0=1234)
            for c in conds:
                tally[c].extend([1 if x == "safe" else 0 for x in r[c]])
            fmass.extend(r["fmass"]); gmass.extend(r["gmass"])
            line = " ".join(f"{c}={sum(1 for x in r[c] if x=='safe')}/{len(r[c])}" for c in conds)
            print(f"  [{scn}/{oid}] {line} | CoT->field={sum(r['fmass'])/max(1,len(r['fmass'])):.3f} "
                  f"CoT->gate={sum(r['gmass'])/max(1,len(r['gmass'])):.3f}", flush=True)

    agg = {"model": args.model, "n_samples": len(tally["inplace_base"])}
    for c in conds:
        k = sum(tally[c]); n = len(tally[c])
        agg[c] = {"P_correct": round(k / n, 3) if n else None, "ci": wilson(k, n), "n": n}
    agg["CoT_to_field_attn"] = round(sum(fmass) / len(fmass), 4) if fmass else None
    agg["CoT_to_gate_attn"] = round(sum(gmass) / len(gmass), 4) if gmass else None
    json.dump(agg, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"mech_reasoning_reread_{args.tag}.json"), "w"), indent=2)
    print("\n==== D4 REASONING RE-READ SUMMARY ====")
    for c in conds:
        print(f"  {c:18s} P_correct={agg[c]['P_correct']} CI{agg[c]['ci']} (n={agg[c]['n']})")
    print(f"  CoT->field attn mass={agg['CoT_to_field_attn']}  CoT->gate band={agg['CoT_to_gate_attn']}")
    print("  HYPOTHESIS: block_cot_field collapses while block_dec_field & block_cot_gate hold")
    print("D4_REASONING_REREAD_DONE")


if __name__ == "__main__":
    main()
