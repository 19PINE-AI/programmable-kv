"""Exp1 - Name the WRITE heads and READ heads of the memoization circuit.

Builds on the localization (mechd_*): the conclusion is memoized on a downstream aggregator
and read by the decision token. Here we name the attention HEADS that do it.

Pair: conclusion-flip with the FIELD held byte-identical (circuit_common.build_pair); only the
rule trigger token differs. clean = SAFE-conclusion, corrupt = UNSAFE-conclusion.

WRITE heads (carry field/trigger -> aggregator note, at prefill):
  candidate ranking: per-head contribution to the aggregator residual that changes most
    between clean and corrupt: ||contrib_clean(h) - contrib_corrupt(h)|| at agg position.
  causal test (path patch): re-prefill CORRUPT with head h's o_proj-input slice at the
    aggregator replaced by the CLEAN value; decode the decision; recovery toward SAFE.
  attention: does head h attend agg -> trigger/field at prefill?

READ heads (aggregator note -> decision logit, at decode):
  candidate ranking: per-head direct-logit-attribution contrast at the decision token
    (contrib . (e_safe - e_unsafe)), clean-decode minus corrupt-decode.
  causal test: decode from the CORRUPT cache with head h's decode context replaced by the
    CLEAN-decode value; recovery toward SAFE.
  attention: does head h attend decision -> aggregator at decode?

Aggregated over instances with bootstrap CIs; random-head control for both axes.
Run: python esys/circ_heads.py --model unsloth/Meta-Llama-3.1-8B-Instruct --tag llama31_8b
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
import circuit_common as cc


def boot_ci(xs, B=2000):
    n = len(xs)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (round(xs[0], 3), round(xs[0], 3))
    means = sorted(sum(xs[(bsi * 2654435761 + j * 40503) % n] for j in range(n)) / n for bsi in range(B))
    return (round(means[int(0.025 * B)], 3), round(means[int(0.975 * B)], 3))


@torch.no_grad()
def prefill_cache(model, ids, patch=None, agg_pos=None, hd=None):
    if patch:
        with cc.HeadPatch(model, patch, agg_pos, hd):
            out = model(input_ids=ids.to("cuda"), use_cache=True)
    else:
        out = model(input_ids=ids.to("cuda"), use_cache=True)
    return out.past_key_values


@torch.no_grad()
def decode_score(model, cache, last, dpos, toi, patch=None, hd=None):
    ids = torch.tensor([[last]], device="cuda")
    cp = torch.tensor([dpos], device="cuda")
    if patch:
        with cc.HeadPatch(model, patch, 0, hd):          # tensor index 0 for the decode token
            out = model(input_ids=ids, past_key_values=cache, cache_position=cp, use_cache=True)
    else:
        out = model(input_ids=ids, past_key_values=cache, cache_position=cp, use_cache=True)
    return cc.conc_score(out.logits[0, -1].float(), toi)


@torch.no_grad()
def capture_attn_in(model, ids):
    """Run prefill, return {layer: o_proj-input [1,L,H*Dh]} (per-head context)."""
    with cc.Capture(model) as cap:
        model(input_ids=ids.to("cuda"), use_cache=False)
        return {li: cap.attn_in[li].clone() for li in cap.attn_in}


@torch.no_grad()
def capture_decode_in(model, cache, last, dpos):
    """Decode one step from `cache`, return {layer: o_proj-input [1,1,H*Dh]}."""
    ids = torch.tensor([[last]], device="cuda")
    cp = torch.tensor([dpos], device="cuda")
    with cc.Capture(model) as cap:
        model(input_ids=ids, past_key_values=cc.clone_cache(cache, dpos), cache_position=cp, use_cache=True)
        return {li: cap.attn_in[li].clone() for li in cap.attn_in}


@torch.no_grad()
def run_instance(model, tok, scn, oid, nh, hd, nl, n_cand=24, topn_agg=3):
    P = cc.build_pair(tok, scn, oid)
    last, dpos, toi = P["last"], P["dpos"], P["toi"]
    agg_list, rec, s_un, s_sa, denom = cc.find_aggregators(model, P, topn=topn_agg)
    if abs(denom) < 0.8:
        return None
    agg = agg_list[0]
    W = [model.model.layers[li].self_attn.o_proj.weight for li in range(nl)]

    # caches
    c_clean = cc.prefill(model, P["safe_ids"]).past_key_values
    c_corr = cc.prefill(model, P["unsafe_ids"]).past_key_values

    # ---- WRITE heads ----
    ain_clean = capture_attn_in(model, P["safe_ids"])
    ain_corr = capture_attn_in(model, P["unsafe_ids"])
    # candidate ranking by change in per-head contribution to agg residual
    wscore = {}
    for li in range(nl):
        cc_ = cc.head_contribs(ain_clean[li], W[li], nh, hd, agg)
        cr_ = cc.head_contribs(ain_corr[li], W[li], nh, hd, agg)
        d = (cc_ - cr_).norm(dim=1)                        # [nh]
        for h in range(nh):
            wscore[(li, h)] = float(d[h])
    wcands = sorted(wscore, key=lambda k: wscore[k], reverse=True)[:n_cand]
    # causal path patch: prefill corrupt with head (li,h) @agg := clean value
    write_rec = {}
    for (li, h) in wcands:
        vec = ain_clean[li][0, agg, h * hd:(h + 1) * hd]
        cache = prefill_cache(model, P["unsafe_ids"], patch={li: [(h, vec)]}, agg_pos=agg, hd=hd)
        sc = decode_score(model, cache, last, dpos, toi)
        write_rec[(li, h)] = (sc - s_un) / denom
    # random control (heads not in candidates)
    allk = [(li, h) for li in range(nl) for h in range(nh)]
    rng = torch.Generator().manual_seed(1234 + hash((scn, oid)) % 9999)
    ctrl = [allk[i] for i in torch.randperm(len(allk), generator=rng)[:8].tolist() if allk[i] not in wcands][:6]
    write_ctrl = {}
    for (li, h) in ctrl:
        vec = ain_clean[li][0, agg, h * hd:(h + 1) * hd]
        cache = prefill_cache(model, P["unsafe_ids"], patch={li: [(h, vec)]}, agg_pos=agg, hd=hd)
        write_ctrl[(li, h)] = (decode_score(model, cache, last, dpos, toi) - s_un) / denom

    # ---- READ heads ----
    din_clean = capture_decode_in(model, c_clean, last, dpos)
    din_corr = capture_decode_in(model, c_corr, last, dpos)
    ld = cc.logit_dir(model, toi)                          # e_safe - e_unsafe  [hidden]
    # DLA contrast per head at decision token
    rscore = {}
    for li in range(nl):
        cc_ = cc.head_contribs(din_clean[li], W[li], nh, hd, 0)     # [nh, hidden]
        cr_ = cc.head_contribs(din_corr[li], W[li], nh, hd, 0)
        proj = ((cc_ - cr_).float() @ ld)                          # [nh]
        for h in range(nh):
            rscore[(li, h)] = float(proj[h])
    rcands = sorted(rscore, key=lambda k: rscore[k], reverse=True)[:n_cand]
    # causal test: decode from CORRUPT cache, head (li,h) decode-context := clean value
    read_rec = {}
    for (li, h) in rcands:
        vec = din_clean[li][0, 0, h * hd:(h + 1) * hd]
        sc = decode_score(model, cc.clone_cache(c_corr, dpos), last, dpos, toi,
                          patch={li: [(h, vec)]}, hd=hd)
        read_rec[(li, h)] = (sc - s_un) / denom
    ctrl_r = [allk[i] for i in torch.randperm(len(allk), generator=rng)[:8].tolist() if allk[i] not in rcands][:6]
    read_ctrl = {}
    for (li, h) in ctrl_r:
        vec = din_clean[li][0, 0, h * hd:(h + 1) * hd]
        read_ctrl[(li, h)] = (decode_score(model, cc.clone_cache(c_corr, dpos), last, dpos, toi,
                              patch={li: [(h, vec)]}, hd=hd) - s_un) / denom

    # ---- CUMULATIVE recovery: patch the top-k named heads JOINTLY (the circuit set) ----
    KS = [1, 2, 3, 5, 8, 12]
    wrank = sorted(wcands, key=lambda k: write_rec[k], reverse=True)
    write_cumk = {}
    for k in KS:
        if k > len(wrank):
            continue
        patch = {}
        for (li, h) in wrank[:k]:
            patch.setdefault(li, []).append((h, ain_clean[li][0, agg, h * hd:(h + 1) * hd]))
        cache = prefill_cache(model, P["unsafe_ids"], patch=patch, agg_pos=agg, hd=hd)
        write_cumk[k] = (decode_score(model, cache, last, dpos, toi) - s_un) / denom
    rrank = sorted(rcands, key=lambda k: read_rec[k], reverse=True)
    read_cumk = {}
    for k in KS:
        if k > len(rrank):
            continue
        patch = {}
        for (li, h) in rrank[:k]:
            patch.setdefault(li, []).append((h, din_clean[li][0, 0, h * hd:(h + 1) * hd]))
        read_cumk[k] = (decode_score(model, cc.clone_cache(c_corr, dpos), last, dpos, toi,
                        patch=patch, hd=hd) - s_un) / denom

    # ---- attention patterns (one SAFE prefill + SAFE decode with output_attentions) ----
    a0, b0 = P["trig_span"]
    with torch.no_grad():
        op = model(input_ids=P["safe_ids"].to("cuda"), use_cache=True, output_attentions=True)
        attn_pref = op.attentions                          # tuple[nl] [1,nh,L,L]
        od = model(input_ids=torch.tensor([[last]], device="cuda"), past_key_values=op.past_key_values,
                   cache_position=torch.tensor([dpos], device="cuda"), use_cache=True, output_attentions=True)
        attn_dec = od.attentions                            # tuple[nl] [1,nh,1,L+1]
    # write-head attn: agg -> trigger span (and agg -> earlier rule region [b0: agg])
    def w_attn(li, h):
        row = attn_pref[li][0, h, agg].float()
        return float(row[a0:b0].sum())
    # read-head attn: decision -> aggregator set
    aggset = agg_list
    def r_attn(li, h):
        row = attn_dec[li][0, h, 0].float()
        return float(sum(row[p] for p in aggset))

    return {
        "scn": scn, "oid": oid, "agg": int(agg), "agg_set": [int(x) for x in agg_list],
        "s_un": round(s_un, 3), "s_sa": round(s_sa, 3), "denom": round(denom, 3),
        "agg_recovery_top1": round(rec[agg], 3),
        "write": {f"{li}.{h}": {"cand": round(wscore[(li, h)], 3), "rec": round(write_rec[(li, h)], 3),
                                "attn_trig": round(w_attn(li, h), 3)} for (li, h) in wcands},
        "write_ctrl_rec": [round(v, 3) for v in write_ctrl.values()],
        "read": {f"{li}.{h}": {"dla": round(rscore[(li, h)], 3), "rec": round(read_rec[(li, h)], 3),
                               "attn_agg": round(r_attn(li, h), 3)} for (li, h) in rcands},
        "read_ctrl_rec": [round(v, 3) for v in read_ctrl.values()],
        "write_cumk": {int(k): round(v, 3) for k, v in write_cumk.items()},
        "read_cumk": {int(k): round(v, 3) for k, v in read_cumk.items()},
    }


def aggregate_heads(recs, key):
    """Mean causal recovery + mean attribution + attn per head, across instances."""
    from collections import defaultdict
    rec = defaultdict(list); attr = defaultdict(list); att = defaultdict(list)
    attr_name = "cand" if key == "write" else "dla"
    att_name = "attn_trig" if key == "write" else "attn_agg"
    for r in recs:
        for hk, d in r[key].items():
            rec[hk].append(d["rec"]); attr[hk].append(d[attr_name]); att[hk].append(d[att_name])
    out = []
    for hk in rec:
        out.append({"head": hk, "n": len(rec[hk]),
                    "rec_mean": round(sum(rec[hk]) / len(rec[hk]), 3), "rec_ci": boot_ci(rec[hk]),
                    "attr_mean": round(sum(attr[hk]) / len(attr[hk]), 3),
                    "attn_mean": round(sum(att[hk]) / len(att[hk]), 3)})
    out.sort(key=lambda x: x["rec_mean"], reverse=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--tag", default="llama31_8b")
    ap.add_argument("--max_instances", type=int, default=8)
    args = ap.parse_args()
    tok, model = cc.load_eager(args.model)
    nh, hd, hidden, nl = cc.cfg_dims(model)
    insts = [(s, o) for s in cc.SCNS for o in cc.OIDS][:args.max_instances]
    recs = []
    for scn, oid in insts:
        r = run_instance(model, tok, scn, oid, nh, hd, nl)
        if r is None:
            print(f"  [{scn}/{oid}] weak denom, skipped", flush=True); continue
        recs.append(r)
        tw = sorted(r["write"].items(), key=lambda kv: kv[1]["rec"], reverse=True)[:3]
        tr = sorted(r["read"].items(), key=lambda kv: kv[1]["rec"], reverse=True)[:3]
        print(f"  [{scn}/{oid}] agg={r['agg']} denom={r['denom']} "
              f"WRITE top={[ (k, tw_i['rec']) for k,tw_i in tw]} "
              f"READ top={[ (k, tr_i['rec']) for k,tr_i in tr]}", flush=True)

    write_agg = aggregate_heads(recs, "write")
    read_agg = aggregate_heads(recs, "read")
    wc = [v for r in recs for v in r["write_ctrl_rec"]]
    rc = [v for r in recs for v in r["read_ctrl_rec"]]
    KS = [1, 2, 3, 5, 8, 12]
    write_cum = {k: [r["write_cumk"][k] for r in recs if k in r["write_cumk"]] for k in KS}
    read_cum = {k: [r["read_cumk"][k] for r in recs if k in r["read_cumk"]] for k in KS}
    summary = {
        "model": args.model, "n_instances": len(recs),
        "write_heads_ranked": write_agg[:15],
        "read_heads_ranked": read_agg[:15],
        "write_ctrl_recovery": {"mean": round(sum(wc) / len(wc), 3), "ci": boot_ci(wc), "n": len(wc)},
        "read_ctrl_recovery": {"mean": round(sum(rc) / len(rc), 3), "ci": boot_ci(rc), "n": len(rc)},
        "write_cumk": {k: {"mean": round(sum(v) / len(v), 3), "ci": boot_ci(v)} for k, v in write_cum.items() if v},
        "read_cumk": {k: {"mean": round(sum(v) / len(v), 3), "ci": boot_ci(v)} for k, v in read_cum.items() if v},
    }
    out = {"summary": summary, "instances": recs}
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"circ_heads_{args.tag}.json"), "w"), indent=2)
    print("\n==== Exp1 HEAD NAMING SUMMARY (%s, n=%d) ====" % (args.tag, len(recs)))
    print("TOP WRITE heads (causal recovery from patching head@aggregator clean->corrupt):")
    for h in write_agg[:8]:
        print(f"   L{h['head']:>6}  rec={h['rec_mean']:+.3f} CI{h['rec_ci']}  attr={h['attr_mean']:.2f}  attn(agg->trig)={h['attn_mean']:.3f}  (n={h['n']})")
    print(f"   write control heads rec={summary['write_ctrl_recovery']['mean']:+.3f} CI{summary['write_ctrl_recovery']['ci']}")
    print("TOP READ heads (causal recovery from patching head@decision clean->corrupt):")
    for h in read_agg[:8]:
        print(f"   L{h['head']:>6}  rec={h['rec_mean']:+.3f} CI{h['rec_ci']}  DLA={h['attr_mean']:+.2f}  attn(dec->agg)={h['attn_mean']:.3f}  (n={h['n']})")
    print(f"   read control heads rec={summary['read_ctrl_recovery']['mean']:+.3f} CI{summary['read_ctrl_recovery']['ci']}")
    print("CUMULATIVE recovery from the top-k named heads patched JOINTLY:")
    print("   write:", {k: f"{v['mean']:+.2f}" for k, v in summary["write_cumk"].items()})
    print("   read: ", {k: f"{v['mean']:+.2f}" for k, v in summary["read_cumk"].items()})
    print("CIRC_HEADS_DONE", flush=True)


if __name__ == "__main__":
    main()
