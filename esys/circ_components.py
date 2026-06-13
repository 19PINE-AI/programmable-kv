"""Exp5 - Attention-vs-MLP decomposition of the WRITE at the aggregator.

Does attention merely ROUTE the conclusion onto the aggregator, or do the MLPs COMPUTE it
there? We decompose the residual that distinguishes SAFE from UNSAFE at the aggregator into
per-layer attention-block and MLP-block contributions (this decomposition is EXACT: at the
aggregator the embedding is identical across the pair -- the trigger token is upstream -- so
  resid_clean - resid_corrupt  =  sum_L [ d_attn_out(L) + d_mlp_out(L) ]  ).

Read direction d-hat = unit vector of (resid_clean - resid_corrupt) at the aggregator's final
layer (the direction the note ultimately points). Per layer L:
  a_L = <d_attn_out(L), d-hat>,  m_L = <d_mlp_out(L), d-hat>;   sum_L (a_L + m_L) = ||delta||.
We report the attention vs MLP share of the write and the depth profile (which layers open the
SAFE-vs-UNSAFE gap), averaged over instances with bootstrap CIs.
Run: python esys/circ_components.py --model unsloth/Meta-Llama-3.1-8B-Instruct --tag llama31_8b
"""
import argparse, json, os, sys
import torch
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import circuit_common as cc


def boot_ci(xs, B=2000):
    n = len(xs)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (round(float(xs[0]), 3), round(float(xs[0]), 3))
    means = sorted(sum(xs[(bsi * 2654435761 + j * 40503) % n] for j in range(n)) / n for bsi in range(B))
    return (round(means[int(0.025 * B)], 3), round(means[int(0.975 * B)], 3))


@torch.no_grad()
def capture_components(model, ids, pos, nl):
    """Return attn_out, mlp_out and post-block residual at `pos` for every layer."""
    with cc.Capture(model) as cap:
        out = model(input_ids=ids.to("cuda"), use_cache=False, output_hidden_states=True)
        a = {L: cap.attn_out[L][0, pos].float() for L in range(nl)}
        m = {L: cap.mlp_out[L][0, pos].float() for L in range(nl)}
        resid = {L: out.hidden_states[L + 1][0, pos].float() for L in range(nl)}
    return a, m, resid


@torch.no_grad()
def run_instance(model, tok, scn, oid, nl, T):
    """Decompose the conclusion note at the aggregator, read at layer T (the CAUSAL conclusion
    layer from Exp2). d-hat = unit (resid_clean - resid_corrupt) at layer T; contributions of
    layers 0..T are projected onto it (sum_{L<=T} (a_L + m_L) = ||delta_T||, exact)."""
    P = cc.build_pair(tok, scn, oid)
    agg_list, rec, s_un, s_sa, denom = cc.find_aggregators(model, P, topn=1)
    if abs(denom) < 0.8:
        return None
    agg = agg_list[0]
    a_c, m_c, r_c = capture_components(model, P["safe_ids"], agg, nl)
    a_x, m_x, r_x = capture_components(model, P["unsafe_ids"], agg, nl)
    da = {L: (a_c[L] - a_x[L]) for L in range(nl)}     # d attn_out
    dm = {L: (m_c[L] - m_x[L]) for L in range(nl)}     # d mlp_out
    delta_T = r_c[T] - r_x[T]                            # conclusion delta at readout layer T
    dhat = delta_T / (delta_T.norm() + 1e-8)
    a_proj = np.array([float(da[L] @ dhat) for L in range(nl)])    # full profile (for display)
    m_proj = np.array([float(dm[L] @ dhat) for L in range(nl)])
    total = float(a_proj[:T + 1].sum() + m_proj[:T + 1].sum())     # = ||delta_T||
    return {"scn": scn, "oid": oid, "agg": int(agg), "denom": round(denom, 3), "T": T,
            "total": total, "a_proj": a_proj, "m_proj": m_proj,
            "attn_share": float(a_proj[:T + 1].sum() / total),
            "mlp_share": float(m_proj[:T + 1].sum() / total)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--tag", default="llama31_8b")
    ap.add_argument("--max_instances", type=int, default=12)
    ap.add_argument("--readout_layer", type=int, default=14, help="causal conclusion layer (Exp2)")
    args = ap.parse_args()
    tok, model = cc.load_eager(args.model)
    nh, hd, hidden, nl = cc.cfg_dims(model)
    T = args.readout_layer
    recs = []
    for scn in cc.SCNS:
        for oid in cc.OIDS:
            r = run_instance(model, tok, scn, oid, nl, T)
            if r is None:
                continue
            recs.append(r)
            print(f"  [{scn}/{oid}] agg={r['agg']} attn_share={r['attn_share']:+.2f} "
                  f"mlp_share={r['mlp_share']:+.2f}", flush=True)
            if len(recs) >= args.max_instances:
                break
        if len(recs) >= args.max_instances:
            break

    A = np.stack([r["a_proj"] for r in recs])[:, :T + 1]    # restrict to layers 0..T (the readout)
    M = np.stack([r["m_proj"] for r in recs])[:, :T + 1]
    tot = np.array([r["total"] for r in recs])
    An = A / tot[:, None]; Mn = M / tot[:, None]            # per-instance normalized contributions
    attn_share = [r["attn_share"] for r in recs]
    mlp_share = [r["mlp_share"] for r in recs]
    cum = np.cumsum(An + Mn, axis=1)                         # [n, T+1]
    layer_band = {}
    for name, lo, hi in [("early", 0, (T + 1) // 3), ("mid", (T + 1) // 3, 2 * (T + 1) // 3),
                         ("late", 2 * (T + 1) // 3, T + 1)]:
        band = (An[:, lo:hi].sum(1) + Mn[:, lo:hi].sum(1))
        layer_band[name] = {"mean": round(float(band.mean()), 3), "ci": boot_ci(list(band))}

    def depth_at(frac):
        ds = []
        for i in range(len(recs)):
            idx = int(np.searchsorted(cum[i], frac * cum[i, -1]))
            ds.append(idx / T)
        return round(float(np.mean(ds)), 3)
    summary = {
        "model": args.model, "n_instances": len(recs), "nlayers": nl, "readout_layer": T,
        "attn_share": {"mean": round(float(np.mean(attn_share)), 3), "ci": boot_ci(attn_share)},
        "mlp_share": {"mean": round(float(np.mean(mlp_share)), 3), "ci": boot_ci(mlp_share)},
        "layer_band_share": layer_band,
        "write_depth_p50": depth_at(0.5), "write_depth_p90": depth_at(0.9),
        "attn_per_layer": [round(float(x), 4) for x in An.mean(0)],
        "mlp_per_layer": [round(float(x), 4) for x in Mn.mean(0)],
    }
    out = {"summary": summary, "instances": [{k: v for k, v in r.items() if k not in ("a_proj", "m_proj")} for r in recs]}
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"circ_components_{args.tag}.json"), "w"), indent=2)
    print("\n==== Exp5 ATTN-vs-MLP WRITE DECOMPOSITION (%s, n=%d) ====" % (args.tag, len(recs)))
    print(f"attention share of the write: {summary['attn_share']['mean']:+.2f} CI{summary['attn_share']['ci']}")
    print(f"MLP       share of the write: {summary['mlp_share']['mean']:+.2f} CI{summary['mlp_share']['ci']}")
    print(f"layer-band share: { {k: v['mean'] for k, v in layer_band.items()} }")
    print(f"write accumulated: 50% by depth {summary['write_depth_p50']}, 90% by depth {summary['write_depth_p90']}")
    # top contributing layers (attn and mlp)
    ap_ = summary["attn_per_layer"]; mp_ = summary["mlp_per_layer"]
    top_a = sorted(range(len(ap_)), key=lambda L: ap_[L], reverse=True)[:5]
    top_m = sorted(range(len(mp_)), key=lambda L: mp_[L], reverse=True)[:5]
    print(f"top attn layers (share): { {L: ap_[L] for L in top_a} }")
    print(f"top mlp  layers (share): { {L: mp_[L] for L in top_m} }")
    print("CIRC_COMPONENTS_DONE", flush=True)


if __name__ == "__main__":
    main()
