"""Exp2 - A causal 1-D conclusion direction on the aggregator residual (DAS-style).

The linear probe in mechd_* shows the conclusion is *decodable* downstream. Decodability is
correlational. Here we test a 1-D direction d for CAUSAL sufficiency and necessity, using the
2x2 (field {vA,vB} x trigger {vA,vB}; conclusion = SAFE iff field==trigger) so that averaging
over field values makes d a *conclusion* direction, not a field-content direction.

Direction (per layer L, at the aggregator residual): difference-of-means
  d = mean(resid | conclusion=SAFE) - mean(resid | conclusion=UNSAFE),   field-balanced.
We also fit a logistic-probe direction (conclusion label, field-balanced) and compare.
The aggregator is anchored by its offset-from-end (from the pair's top KV-patch position), so
it is consistent across the field values whose token lengths differ.

Causal tests (inject a delta at the aggregator residual of layer L during a CORRUPT=UNSAFE
prefill, propagate, decode; recovery toward SAFE):
  full   : delta = resid_clean - resid_corrupt                      (upper bound ~ KV patch)
  along  : delta projected onto d-hat                               (1-D sufficiency)
  orth   : delta minus its d-component                              (residual not on d)
  random : random unit dir, matched norm to the along-component     (control)
And NECESSITY: from a CLEAN=SAFE prefill, subtract the d-component of the clean residual at the
aggregator -> decision should fall toward UNSAFE.
Direction is fit LEAVE-ONE-SCENARIO-OUT (fit on 2 scenarios, test on the 3rd) to avoid circularity.
Run: python esys/circ_direction.py --model unsloth/Meta-Llama-3.1-8B-Instruct --tag llama31_8b
"""
import argparse, json, os, sys
import torch
import numpy as np
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


class ResidInject:
    """Add per-position vectors to the residual at `positions` on the OUTPUT of decoder `layer`."""
    def __init__(self, model, layer, positions, vecs):
        self.layer = cc.decoder_layers(model)[layer]
        self.positions, self.vecs, self.h = positions, vecs, None

    def __enter__(self):
        def hook(mod, args, out):
            hs = out[0] if isinstance(out, tuple) else out
            if hs.shape[1] > max(self.positions):
                for p, v in zip(self.positions, self.vecs):
                    hs[0, p] = hs[0, p] + v.to(hs.dtype)
            return (hs,) + out[1:] if isinstance(out, tuple) else hs
        self.h = self.layer.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.h.remove()


@torch.no_grad()
def prefill_hs(model, ids, layer, positions, vecs=None):
    """Prefill (optionally injecting vecs at layer/positions); return (cache, resid[positions])."""
    if vecs is not None:
        with ResidInject(model, layer, positions, vecs):
            out = model(input_ids=ids.to("cuda"), use_cache=True, output_hidden_states=True)
    else:
        out = model(input_ids=ids.to("cuda"), use_cache=True, output_hidden_states=True)
    resid = out.hidden_states[layer + 1][0, positions].float().clone()  # [len(pos), hidden]
    return out.past_key_values, resid


@torch.no_grad()
def score_cache(model, cache, last, dpos, toi):
    return cc.conc_score(cc.decision_logits(model, cc.clone_cache(cache, dpos), last, dpos), toi)


def make_prompt(tok, scn, oid, field, trig):
    t = build_pol(tok, scn, oid, field, trig, False, True)
    ids = torch.tensor([tok(t, add_special_tokens=False)["input_ids"]])
    return ids


@torch.no_grad()
def instance_meta(model, tok, scn, oid, topn_agg=8):
    """Anchor aggregator offsets-from-end via the pair (top-N), + readout tokens + denom."""
    P = cc.build_pair(tok, scn, oid)
    agg_list, rec, s_un, s_sa, denom = cc.find_aggregators(model, P, topn=topn_agg)
    offs = sorted(P["dpos"] - p for p in agg_list)
    return dict(scn=scn, oid=oid, offs=offs, toi=P["toi"], denom=denom)


@torch.no_grad()
def collect_resid(model, tok, scn, oid, offs, layers):
    """Residuals at the aggregator SET (anchored by offset-from-end) for the 4 conditions.
    Each (condition, position) contributes one labeled sample to the direction pool."""
    s = POL[scn]; vA, vB = s["values"]
    conds = [(vA, vA, 1), (vA, vB, 0), (vB, vB, 1), (vB, vA, 0)]   # (field,trig,is_safe)
    res = []
    for field, trig, safe in conds:
        ids = make_prompt(tok, scn, oid, field, trig)
        dpos = ids.shape[1] - 1
        positions = [dpos - o for o in offs]
        out = model(input_ids=ids.to("cuda"), use_cache=False, output_hidden_states=True)
        for pi, pos in enumerate(positions):
            hs = {L: out.hidden_states[L + 1][0, pos].float().cpu().numpy() for L in layers}
            res.append((safe, hs))
    return res


def fit_dirs(resid_pool, layers, exclude_scn):
    """Per-layer difference-of-means + logistic-probe directions, fit on scenarios != exclude."""
    dirs = {}
    for L in layers:
        X, y = [], []
        for (scn, safe, hs) in resid_pool:
            if scn == exclude_scn:
                continue
            X.append(hs[L]); y.append(safe)
        X = np.array(X); y = np.array(y)
        mu_s = X[y == 1].mean(0); mu_u = X[y == 0].mean(0)
        d_dm = mu_s - mu_u
        d_dm = d_dm / (np.linalg.norm(d_dm) + 1e-8)
        # logistic probe (standardized), conclusion label
        from sklearn.linear_model import LogisticRegression
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xs, y)
        d_pr = clf.coef_[0] / (X.std(0) + 1e-6)
        d_pr = d_pr / (np.linalg.norm(d_pr) + 1e-8)
        dirs[L] = (torch.tensor(d_dm, dtype=torch.float32, device="cuda"),
                   torch.tensor(d_pr, dtype=torch.float32, device="cuda"))
    return dirs


@torch.no_grad()
def causal_test(model, tok, meta, layer, dhat):
    """Directional patching at one layer across the aggregator SET. Returns recovery dict.
    Intervention: in the CORRUPT prefill, at each aggregator position, add the projection of
    (resid_clean - resid_corrupt) onto d-hat (along), the orthogonal remainder (orth), the full
    delta (full = single-site ceiling), or a matched-norm random direction (random)."""
    scn, oid, offs, toi = meta["scn"], meta["oid"], meta["offs"], meta["toi"]
    s = POL[scn]; fld = s["values"][0]
    clean_ids = make_prompt(tok, scn, oid, fld, fld)            # SAFE
    corr_ids = make_prompt(tok, scn, oid, fld, s["values"][1])  # UNSAFE
    dpos_c = clean_ids.shape[1] - 1; pos_c = [dpos_c - o for o in offs]
    dpos_x = corr_ids.shape[1] - 1; pos_x = [dpos_x - o for o in offs]
    last_c = int(clean_ids[0, dpos_c]); last_x = int(corr_ids[0, dpos_x])

    cache_c, resid_c = prefill_hs(model, clean_ids, layer, pos_c)   # [K,hidden]
    cache_x, resid_x = prefill_hs(model, corr_ids, layer, pos_x)
    s_clean = score_cache(model, cache_c, last_c, dpos_c, toi)
    s_corr = score_cache(model, cache_x, last_x, dpos_x, toi)
    denom2 = s_clean - s_corr
    if abs(denom2) < 0.5:
        return None
    delta = resid_c - resid_x                                  # [K,hidden]
    comp = (delta @ dhat).unsqueeze(1) * dhat.unsqueeze(0)     # along d, per position
    orth = delta - comp
    g = torch.Generator(device="cuda").manual_seed(7)
    rdir = torch.randn(dhat.shape, generator=g, device="cuda"); rdir = rdir / rdir.norm()
    rnd = (delta @ rdir).unsqueeze(1) * rdir.unsqueeze(0)      # matched: project onto random dir

    def rec(vecs):
        cache, _ = prefill_hs(model, corr_ids, layer, pos_x, vecs=vecs)
        return (score_cache(model, cache, last_x, dpos_x, toi) - s_corr) / denom2

    out = {"full": rec(delta), "along": rec(comp), "orth": rec(orth), "random": rec(rnd)}
    # necessity: remove d-component from CLEAN run at all agg positions -> drop toward UNSAFE
    proj_clean = (resid_c @ dhat).unsqueeze(1) * dhat.unsqueeze(0)
    cache_n, _ = prefill_hs(model, clean_ids, layer, pos_c, vecs=-proj_clean)
    s_removed = score_cache(model, cache_n, last_c, dpos_c, toi)
    out["necessity_drop"] = (s_clean - s_removed) / denom2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--tag", default="llama31_8b")
    ap.add_argument("--layers", default="8,12,16,20,24,28")
    ap.add_argument("--max_oids", type=int, default=3)
    args = ap.parse_args()
    tok, model = cc.load_eager(args.model)
    nh, hd, hidden, nl = cc.cfg_dims(model)
    layers = [int(x) for x in args.layers.split(",")]
    oids = cc.OIDS[:args.max_oids]

    # 1) collect residual pool + per-instance meta
    print("collecting residuals...", flush=True)
    resid_pool = []        # (scn, safe, {L: vec})
    metas = []
    for scn in cc.SCNS:
        for oid in oids:
            m = instance_meta(model, tok, scn, oid)
            metas.append(m)
            for (safe, hs) in collect_resid(model, tok, scn, oid, m["offs"], layers):
                resid_pool.append((scn, safe, hs))
    print(f"  pool={len(resid_pool)} condition-residuals, {len(metas)} instances", flush=True)

    # 2) leave-one-scenario-out directions, 3) causal tests per layer
    res = {L: {"dm": {k: [] for k in ["full", "along", "orth", "random", "necessity_drop"]},
               "pr": {k: [] for k in ["full", "along", "orth", "random", "necessity_drop"]}}
           for L in layers}
    for held in cc.SCNS:
        dirs = fit_dirs(resid_pool, layers, exclude_scn=held)
        for m in metas:
            if m["scn"] != held:
                continue
            for L in layers:
                d_dm, d_pr = dirs[L]
                for name, dhat in (("dm", d_dm), ("pr", d_pr)):
                    out = causal_test(model, tok, m, L, dhat)
                    if out is None:
                        continue
                    for k in res[L][name]:
                        res[L][name][k].append(out[k])
            print(f"  tested {m['scn']}/{m['oid']} (held={held})", flush=True)

    summary = {"model": args.model, "layers": layers, "n_instances": len(metas), "per_layer": {}}
    for L in layers:
        summary["per_layer"][L] = {}
        for name in ("dm", "pr"):
            d = res[L][name]
            summary["per_layer"][L][name] = {k: {"mean": round(float(np.mean(v)), 3),
                                                 "ci": boot_ci(v)} for k, v in d.items() if v}
    out = {"summary": summary}
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"circ_direction_{args.tag}.json"), "w"), indent=2)
    print("\n==== Exp2 CONCLUSION DIRECTION (difference-of-means, leave-scenario-out) ====")
    print(f"{'layer':>5} | {'full':>6} | {'along-d':>8} | {'along/full':>10} | {'orth':>6} | {'random':>7} | {'necessity':>9}")
    for L in layers:
        dm = summary["per_layer"][L]["dm"]
        ratio = dm['along']['mean'] / dm['full']['mean'] if dm['full']['mean'] else 0
        print(f"{L:>5} | {dm['full']['mean']:>6.2f} | {dm['along']['mean']:>8.2f} | {ratio:>10.2f} | "
              f"{dm['orth']['mean']:>6.2f} | {dm['random']['mean']:>7.2f} | {dm['necessity_drop']['mean']:>9.2f}")
    # best layer by along/full sufficiency ratio (fraction of single-site causal effect a 1-D dir carries)
    best = max(layers, key=lambda L: (summary["per_layer"][L]["dm"]["along"]["mean"] /
                                      (summary["per_layer"][L]["dm"]["full"]["mean"] + 1e-9)))
    b = summary["per_layer"][best]["dm"]
    ratio = b['along']['mean'] / (b['full']['mean'] + 1e-9)
    print(f"\nBEST layer L{best}: along-d={b['along']['mean']} CI{b['along']['ci']} of full={b['full']['mean']} "
          f"=> 1-D carries {ratio:.0%} of the single-site causal effect "
          f"(orth={b['orth']['mean']}, random={b['random']['mean']}, necessity drop={b['necessity_drop']['mean']})")
    pr = summary["per_layer"][best]["pr"]
    print(f"  probe-direction along-d at L{best}: {pr['along']['mean']} CI{pr['along']['ci']}")
    print("CIRC_DIRECTION_DONE", flush=True)


if __name__ == "__main__":
    main()
