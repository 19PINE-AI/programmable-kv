"""Exp4 - Causal-scrubbing faithfulness of the write -> note -> read circuit.

Hypothesis H: the decision is a function of the field-conditioned CONCLUSION carried by the
aggregator note; everything else downstream is interchangeable across inputs that H labels as
having the same conclusion. Causal scrubbing tests H by resampling each activation from a
different input that AGREES with H's labelling and checking the behaviour is preserved.

We instantiate H at the KV level. Donors share the SAME scenario and differ only in the order
id (identical token length -> every position aligns), so we can transplant KV positions exactly.
For a base instance with conclusion p:
  donor_same = same scenario, same conclusion p, different order id
  donor_opp  = same scenario, OPPOSITE conclusion, same order id (only the rule trigger flips)
Let Agg = the top-k aggregator positions (the note); Rest = the other post-trigger downstream.

Faithfulness (H-allowed resampling must PRESERVE the decision; drift = |s'-s_base|/denom):
  scrub_rest_same : overwrite Rest with donor_same        -> drift ~ 0   (Rest is interchangeable)
  scrub_note_same : overwrite Agg  with donor_same        -> drift ~ 0   (same conclusion note)
  scrub_all_same  : overwrite Agg+Rest with donor_same    -> drift ~ 0
Necessity / interchange (resample the NOTE from the OPPOSITE conclusion -> behaviour must FLIP):
  swap_note_opp   : overwrite Agg with donor_opp          -> recovery toward donor_opp ~ 1
  swap_rest_opp   : overwrite Rest with donor_opp         -> recovery ~ 0  (Rest carries no conclusion)
We report drift, interchange recovery, and decision-agreement, with bootstrap CIs.
Run: python esys/circ_scrub.py --model unsloth/Meta-Llama-3.1-8B-Instruct --tag llama31_8b
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
import circuit_common as cc
from mechd_common import POL, build_pol


def boot_ci(xs, B=2000):
    n = len(xs)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (round(float(xs[0]), 3), round(float(xs[0]), 3))
    means = sorted(sum(xs[(bsi * 2654435761 + j * 40503) % n] for j in range(n)) / n for bsi in range(B))
    return (round(means[int(0.025 * B)], 3), round(means[int(0.975 * B)], 3))


def prompt_ids(tok, scn, oid, field, trig):
    t = build_pol(tok, scn, oid, field, trig, False, True)
    return torch.tensor([tok(t, add_special_tokens=False)["input_ids"]])


@torch.no_grad()
def overwrite(model, base_cache, donor_cache, positions, last, dpos, toi):
    w = cc.clone_cache(base_cache, dpos)
    pos = torch.tensor(positions, device=w.layers[0].keys.device)
    for i in range(len(w.layers)):
        w.layers[i].keys[:, :, pos, :] = donor_cache.layers[i].keys[:, :, pos, :]
        w.layers[i].values[:, :, pos, :] = donor_cache.layers[i].values[:, :, pos, :]
    lg = cc.decision_logits(model, w, last, dpos)
    return cc.conc_score(lg, toi), lg


@torch.no_grad()
def run_scenario(model, tok, scn, k_note=8):
    s = POL[scn]; vA, vB = s["values"]
    toi = {"safe": cc.ftok(tok, cc.ACT_TOK[s["safe"]]), "unsafe": cc.ftok(tok, cc.ACT_TOK[s["unsafe"]])}
    recs = []
    # Work entirely in build_pair's ALIGNED coordinate so every prompt in a scenario has the
    # same length and aligned positions (same scenario -> same trigger values -> same padding).
    # conclusion p in {SAFE (trig==field), UNSAFE (trig!=field)}.
    for p_safe in (True, False):
        for bi, base_oid in enumerate(cc.OIDS):
            P = cc.build_pair(tok, scn, base_oid)
            base_ids = P["safe_ids"] if p_safe else P["unsafe_ids"]
            do_ids = P["unsafe_ids"] if p_safe else P["safe_ids"]   # opposite conclusion, aligned
            dpos = P["dpos"]; last = P["last"]
            base_cache = cc.prefill(model, base_ids).past_key_values
            do_cache = cc.prefill(model, do_ids).past_key_values
            s_base = cc.conc_score(cc.decision_logits(model, cc.clone_cache(base_cache, dpos), last, dpos), toi)
            lg_base = cc.decision_logits(model, cc.clone_cache(base_cache, dpos), last, dpos)
            s_opp = cc.conc_score(cc.decision_logits(model, cc.clone_cache(do_cache, dpos), last, dpos), toi)

            # aggregator set (top-k) and the rest of the post-trigger downstream
            agg_list, rec, s_un, s_sa, denom = cc.find_aggregators(model, P, topn=k_note)
            if abs(denom) < 0.8:
                continue
            b0 = P["trig_span"][1]
            agg = sorted(set(int(x) for x in agg_list))
            rest = [pp for pp in range(b0, dpos) if pp not in set(agg)]

            # donor_same: same conclusion, different oid (same scenario -> aligned, same length)
            donor_oid = cc.OIDS[(bi + 1) % len(cc.OIDS)]
            Pd = cc.build_pair(tok, scn, donor_oid)
            ds_ids = Pd["safe_ids"] if p_safe else Pd["unsafe_ids"]
            if ds_ids.shape[1] != base_ids.shape[1]:
                continue                                       # require aligned length for transplant
            ds_cache = cc.prefill(model, ds_ids).past_key_values

            def drift(positions, donor):
                s2, _ = overwrite(model, base_cache, donor, positions, last, dpos, toi)
                return abs(s2 - s_base) / abs(denom)

            def recov(positions, donor, target):
                s2, _ = overwrite(model, base_cache, donor, positions, last, dpos, toi)
                return (s2 - s_base) / (target - s_base) if abs(target - s_base) > 1e-6 else 0.0

            def cos(positions, donor):
                s2, lg2 = overwrite(model, base_cache, donor, positions, last, dpos, toi)
                return float(torch.nn.functional.cosine_similarity(lg_base, lg2, dim=0))

            recs.append({
                "scn": scn, "oid": base_oid, "p_safe": p_safe, "denom": round(denom, 3),
                "drift_rest_same": drift(rest, ds_cache),
                "drift_note_same": drift(agg, ds_cache),
                "drift_all_same": drift(agg + rest, ds_cache),
                "cos_all_same": cos(agg + rest, ds_cache),
                "rec_note_opp": recov(agg, do_cache, s_opp),
                "rec_rest_opp": recov(rest, do_cache, s_opp),
            })
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--tag", default="llama31_8b")
    ap.add_argument("--k_note", type=int, default=8)
    args = ap.parse_args()
    tok, model = cc.load_eager(args.model)
    recs = []
    for scn in cc.SCNS:
        rs = run_scenario(model, tok, scn, args.k_note)
        recs.extend(rs)
        print(f"  [{scn}] n={len(rs)} "
              f"drift_all_same={sum(r['drift_all_same'] for r in rs)/max(1,len(rs)):.3f} "
              f"rec_note_opp={sum(r['rec_note_opp'] for r in rs)/max(1,len(rs)):.3f}", flush=True)

    def agg(key):
        v = [r[key] for r in recs]
        return {"mean": round(sum(v) / len(v), 3), "ci": boot_ci(v), "n": len(v)}
    summary = {"model": args.model, "n": len(recs), "k_note": args.k_note,
               "faithfulness_drift": {k: agg(k) for k in ["drift_rest_same", "drift_note_same", "drift_all_same"]},
               "cos_all_same": agg("cos_all_same"),
               "interchange_recovery": {k: agg(k) for k in ["rec_note_opp", "rec_rest_opp"]}}
    out = {"summary": summary, "instances": recs}
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"circ_scrub_{args.tag}.json"), "w"), indent=2)
    print("\n==== Exp4 CAUSAL SCRUBBING (%s, n=%d) ====" % (args.tag, len(recs)))
    print("Faithfulness (H-allowed resampling, drift toward base = |s'-s_base|/denom, want ~0):")
    for k in ["drift_rest_same", "drift_note_same", "drift_all_same"]:
        d = summary["faithfulness_drift"][k]
        print(f"   {k:>18}: {d['mean']:.3f} CI{d['ci']}")
    print(f"   decision-logit cosine (scrub_all_same vs base): {summary['cos_all_same']['mean']:.3f} CI{summary['cos_all_same']['ci']}")
    print("Necessity / interchange (resample the NOTE from OPPOSITE conclusion, recovery toward it, want note~1 rest~0):")
    for k in ["rec_note_opp", "rec_rest_opp"]:
        d = summary["interchange_recovery"][k]
        print(f"   {k:>18}: {d['mean']:.3f} CI{d['ci']}")
    print("CIRC_SCRUB_DONE", flush=True)


if __name__ == "__main__":
    main()
