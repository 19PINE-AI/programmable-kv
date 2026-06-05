"""D1 — Causal KV patching: a *memoization map* (the deep-mechanism flagship).

ROME/causal-tracing logic applied to the KV cache itself — the very object editkv
edits. Two token-aligned prefills (length-preserving field flip):
  OLD value -> decision = UNSAFE action (e.g. issue_refund)
  NEW value -> decision = SAFE   action (e.g. escalate)
We take the OLD cache and *patch in* the NEW cache's (K,V) at chosen (layer, position)
sites, then re-read the decision logits. The metric is the RECOVERY fraction

    score(c)      = logit[safe] - logit[unsafe]   at the decision position
    recovery(P)   = (score_patched - score_old) / (score_new - score_old)

i.e. how far patching site-set P moves the decision from OLD(unsafe) toward NEW(safe).
recovery≈0 => that site carries none of the field's decision-relevant content;
recovery≈1 => patching it alone reproduces a full re-prefill there.

Outputs (per scenario instance, aggregated with bootstrap CIs over instances):
  - FIELD-ONLY recovery: patch only the field token span (= the IN_PLACE edit). The
    central claim: this is SMALL -> that's *why* in_place fails.
  - POSITION-resolved recovery (patch each downstream position across all layers):
    where does the new value have to be written? (field vs gate vs reasoning vs decision)
  - LAYER-resolved recovery (patch all downstream positions within one layer): which
    layers hold the memoized inference.
  - LOCALITY curve: rank positions by individual recovery, patch cumulative top-k,
    report recovery vs k -> "diffuse" becomes a number (k needed for 90% recovery).
  - A coarse layer x region heatmap for one representative instance.
Run: MECH_ATTN=sdpa python esys/mech_causal_patch.py [--model ...] [--quick]
"""
import argparse, json, os, sys, math
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2"))
from mech_suite import (load, clone, prefill, ftok, wilson, META, TOK_WORDS, build, step)
from align import align_pair
import scenarios as S


def boot_ci(xs, B=2000, z=1.96):
    """Bootstrap 95% CI of the mean over instances (deterministic: index hashing, no RNG)."""
    n = len(xs)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (round(xs[0], 3), round(xs[0], 3))
    means = []
    for bsi in range(B):
        s = 0.0
        for j in range(n):
            # deterministic pseudo-resample (avoids banned Math.random/torch rng nondeterminism)
            idx = (bsi * 2654435761 + j * 40503) % n
            s += xs[idx]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * B)]; hi = means[int(0.975 * B)]
    return (round(lo, 3), round(hi, 3))


@torch.no_grad()
def score(model, cache_full, last, dpos, toi):
    """logit[safe]-logit[unsafe] at the decision position, reading a length-dpos cache."""
    out = step(model, clone(cache_full, dpos), last, dpos)
    lg = out.logits[0, -1].float()
    return float(lg[toi["safe"]] - lg[toi["unsafe"]])


@torch.no_grad()
def patched_score(model, co, cn, positions, layers, last, dpos, toi):
    """Clone OLD cache; overwrite (K,V) at `positions` for `layers` from NEW cache; score."""
    w = clone(co, dpos)
    nl = len(w.layers)
    lay = range(nl) if layers is None else layers
    pos = torch.tensor(positions, device=w.layers[0].keys.device)
    for i in lay:
        w.layers[i].keys[:, :, pos, :] = cn.layers[i].keys[:, :, pos, :]
        w.layers[i].values[:, :, pos, :] = cn.layers[i].values[:, :, pos, :]
    out = step(model, w, last, dpos)
    lg = out.logits[0, -1].float()
    return float(lg[toi["safe"]] - lg[toi["unsafe"]])


def regions(tok, text, a, b, scn):
    """Best-effort semantic segmentation of token positions (for interpretation)."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    def tokidx(sub, start=0):
        c = text.find(sub, start)
        if c < 0:
            return None
        for ti, (s, e) in enumerate(offs):
            if s <= c < e or s >= c:
                return ti
        return None
    gate_anchor = S.SCENARIOS[scn]["gate"][:24]
    gi = tokidx(gate_anchor)
    ci = tokidx("Let me check")
    di = tokidx("TASK")
    return {"field": (a, b), "gate_start": gi, "convo_start": ci, "decision_start": di}


def run_instance(model, tok, scn, oid, quick=False):
    m = META[scn]
    toi = {"safe": ftok(tok, TOK_WORDS[m["safe"]]), "unsafe": ftok(tok, TOK_WORDS[m["unsafe"]])}
    t_old = build(tok, scn, oid, m["vold"], False, True)
    t_new = build(tok, scn, oid, m["vnew"], False, True)
    al = align_pair(tok, t_old, t_new)
    old_ids = al["old_ids"]; new_ids = al["new_ids"]   # already [1, L] tensors
    a, b = al["field_span"]; L = old_ids.shape[1]; dpos = L - 1
    last = int(new_ids[0, dpos])               # identical suffix on both sides
    co = prefill(model, old_ids); cn = prefill(model, new_ids)
    nl = len(co.layers)

    s_old = score(model, co, last, dpos, toi)
    s_new = score(model, cn, last, dpos, toi)
    denom = (s_new - s_old)
    if abs(denom) < 1e-4:
        return None                            # non-flipping instance, skip
    def rec(positions, layers=None):
        return (patched_score(model, co, cn, positions, layers, last, dpos, toi) - s_old) / denom

    # --- field-only (== IN_PLACE) ---
    field_only = rec(list(range(a, b)))
    # --- full downstream (sanity: should ~1.0) ---
    downstream = list(range(a, dpos))
    full_down = rec(downstream)
    # --- position-resolved (patch each downstream position, all layers) ---
    step_pos = 2 if quick else 1
    posrec = {}
    for i in range(a, dpos, step_pos):
        posrec[i] = rec([i])
    # --- layer-resolved (patch ALL downstream positions within one layer) ---
    layrec = {}
    lstep = 2 if quick else 1
    for li in range(0, nl, lstep):
        layrec[li] = rec(downstream, layers=[li])
    # --- locality: cumulative top-k positions (by individual recovery) ---
    ranked = sorted(posrec, key=lambda i: posrec[i], reverse=True)
    ks = [1, 2, 4, 8, 16, 32, 64, 128]
    loc = {}
    for k in ks:
        if k <= len(ranked):
            loc[k] = rec(ranked[:k])
    top_pos = ranked[0]
    # --- cumulative SUFFIX recovery (nested, additive): patch the last k downstream
    #     positions [dpos-k, dpos). This is the system-relevant "how deep a suffix
    #     recompute recovers the decision" curve (relates to partial-reprefill cost). ---
    ndown = dpos - a
    frac_grid = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    cum_suffix = {}
    for fr in frac_grid:
        k = max(1, int(round(fr * ndown)))
        cum_suffix[round(fr, 2)] = rec(list(range(dpos - k, dpos)))
    # --- cumulative PREFIX recovery: patch the first k downstream positions [a, a+k). ---
    cum_prefix = {}
    for fr in frac_grid:
        k = max(1, int(round(fr * ndown)))
        cum_prefix[round(fr, 2)] = rec(list(range(a, a + k)))
    reg = regions(tok, t_old, a, b, scn)
    return {"scn": scn, "oid": oid, "L": L, "field_span": [a, b], "s_old": round(s_old, 3),
            "s_new": round(s_new, 3), "field_only_recovery": round(field_only, 3),
            "full_downstream_recovery": round(full_down, 3),
            "posrec": {int(k): round(v, 4) for k, v in posrec.items()},
            "layrec": {int(k): round(v, 4) for k, v in layrec.items()},
            "locality_topk": {int(k): round(v, 3) for k, v in loc.items()},
            "cum_suffix": {float(k): round(v, 3) for k, v in cum_suffix.items()},
            "cum_prefix": {float(k): round(v, 3) for k, v in cum_prefix.items()},
            "top_pos": int(top_pos), "top_pos_recovery": round(posrec[top_pos], 3),
            "regions": reg, "nlayers": nl}


def _region_bins(rec):
    reg = rec["regions"]; a, b = rec["field_span"]; dpos = rec["L"] - 1
    gs = reg["gate_start"] or b; cs = reg["convo_start"] or dpos; ds = reg["decision_start"] or dpos
    return {"field": (a, b), "field->gate": (b, gs), "gate->convo": (gs, cs),
            "convo->decision": (cs, ds), "decision": (ds, dpos)}


def agg_region_recovery(rec):
    """MEAN single-position recovery within each semantic region (NOT additive across
    positions — single-position patches are individually weak/noisy because the memoized
    inference is synergistic; this only indicates *where* the strongest single sites sit)."""
    bins = _region_bins(rec)
    out = {}
    for name, (s, e) in bins.items():
        vals = [v for k, v in rec["posrec"].items() if s <= k < e]
        out[name] = round(sum(vals) / len(vals), 4) if vals else None
    return out


def region_of(rec, pos):
    for name, (s, e) in _region_bins(rec).items():
        if s <= pos < e:
            return name
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--scns", default="account_role,safety_mode,subscription_tier")
    ap.add_argument("--oids", default="A4471,B8820,C1093,D5567")
    args = ap.parse_args()
    tok, model = load(args.model)
    scns = args.scns.split(","); oids = (args.oids.split(",")[:2] if args.quick else args.oids.split(","))
    recs = []
    for scn in scns:
        for oid in oids:
            r = run_instance(model, tok, scn, oid, quick=args.quick)
            if r is None:
                print(f"  [{scn}/{oid}] non-flipping, skipped", flush=True); continue
            r["region_recovery"] = agg_region_recovery(r)
            r["top_pos_region"] = region_of(r, r["top_pos"])
            recs.append(r)
            print(f"  [{scn}/{oid}] L={r['L']} field_only={r['field_only_recovery']:.3f} "
                  f"full_down={r['full_downstream_recovery']:.2f} "
                  f"suffix@0.1={r['cum_suffix'].get(0.1)} suffix@0.3={r['cum_suffix'].get(0.3)} "
                  f"top_pos_region={r['top_pos_region']}", flush=True)

    # ---- aggregate ----
    fo = [r["field_only_recovery"] for r in recs]
    fd = [r["full_downstream_recovery"] for r in recs]
    agg = {"model": args.model, "n_instances": len(recs),
           "field_only_recovery": {"mean": round(sum(fo) / len(fo), 3), "ci": boot_ci(fo)},
           "full_downstream_recovery": {"mean": round(sum(fd) / len(fd), 3), "ci": boot_ci(fd)}}
    # locality aggregate (top-k by individual recovery; non-additive => weak)
    agg["locality_topk_mean"] = {}
    for k in [1, 2, 4, 8, 16, 32, 64, 128]:
        vs = [r["locality_topk"][k] for r in recs if k in r["locality_topk"]]
        if vs:
            agg["locality_topk_mean"][k] = {"mean": round(sum(vs) / len(vs), 3), "ci": boot_ci(vs), "n": len(vs)}
    # cumulative suffix / prefix (nested, additive) — the system-relevant locality
    for which in ["cum_suffix", "cum_prefix"]:
        agg[which + "_mean"] = {}
        fracs = sorted({k for r in recs for k in r[which]})
        for fr in fracs:
            vs = [r[which][fr] for r in recs if fr in r[which]]
            if vs:
                agg[which + "_mean"][fr] = {"mean": round(sum(vs) / len(vs), 3), "ci": boot_ci(vs)}
    # where does the single strongest patch site live?
    from collections import Counter
    agg["top_pos_region_counts"] = dict(Counter(r["top_pos_region"] for r in recs))
    # region aggregate
    agg["region_recovery_mean"] = {}
    for name in ["field", "field->gate", "gate->convo", "convo->decision", "decision"]:
        vs = [r["region_recovery"][name] for r in recs if name in r["region_recovery"]]
        agg["region_recovery_mean"][name] = round(sum(vs) / len(vs), 3) if vs else None
    # layer profile (mean recovery vs relative layer depth, binned into thirds)
    thirds = {"early": [], "mid": [], "late": []}
    for r in recs:
        nl = r["nlayers"]
        for li, v in r["layrec"].items():
            band = "early" if li < nl / 3 else ("mid" if li < 2 * nl / 3 else "late")
            thirds[band].append(v)
    agg["layer_band_recovery"] = {b: (round(sum(v) / len(v), 3) if v else None) for b, v in thirds.items()}

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    out = {"agg": agg, "instances": recs}
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"mech_causal_patch_{args.tag}.json"), "w"), indent=2)
    print("\n==== D1 CAUSAL PATCHING SUMMARY ====")
    print(f"n={len(recs)} instances")
    print(f"FIELD-ONLY recovery (= in_place): {agg['field_only_recovery']['mean']} CI{agg['field_only_recovery']['ci']}  "
          f"(small => why in_place fails)")
    print(f"FULL-DOWNSTREAM recovery (sanity): {agg['full_downstream_recovery']['mean']} CI{agg['full_downstream_recovery']['ci']}")
    print(f"REGION mean single-pos recovery: {agg['region_recovery_mean']}")
    print(f"strongest single-site region counts: {agg['top_pos_region_counts']}")
    print(f"LAYER-band recovery: {agg['layer_band_recovery']}")
    print("CUMULATIVE SUFFIX recovery (patch last fraction of downstream -> recovery):")
    for fr, v in agg["cum_suffix_mean"].items():
        print(f"   suffix {fr:>4}: {v['mean']:.2f} CI{v['ci']}")
    print("CUMULATIVE PREFIX recovery (patch first fraction of downstream -> recovery):")
    for fr, v in agg["cum_prefix_mean"].items():
        print(f"   prefix {fr:>4}: {v['mean']:.2f} CI{v['ci']}")
    print("LOCALITY top-k (non-additive ranking; weak by construction):")
    for k, v in agg["locality_topk_mean"].items():
        print(f"   top-{k:3d} pos -> {v['mean']:.2f} CI{v['ci']}")
    print("D1_CAUSAL_PATCH_DONE")


if __name__ == "__main__":
    main()
