"""Print markdown tables for REPORT.md from the T1/T7/T1b result JSONs (run on rtx-pro)."""
import json, os, sys
import numpy as np

D = os.environ.get("FOLLOWUP_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
J = lambda f: json.load(open(os.path.join(D, f))) if os.path.exists(os.path.join(D, f)) else None
r3 = lambda x: "—" if x is None else f"{x:+.3f}"


def causal_table(tag):
    d = J(f"t1_neurons_{tag}.json")
    if not d:
        return f"(no t1_neurons_{tag}.json)\n"
    c = d["causal"]
    out = [f"n = {d['n_instances']} pairs ({d['n_train']} train / {d['n_held']} held-out order id)\n",
           "| positions | test | k=1 | 3 | 10 | 30 | 100 | 300 | all neurons |", "|---|---|---|---|---|---|---|---|---|"]
    for pos in ("agg8", "agg1"):
        for mode, name in (("suff", "sufficiency (patch SAFE→UNSAFE)"), ("nec", "necessity (patch UNSAFE→SAFE)"),
                           ("zero", "zero-ablate (drop)"), ("mean", "mean-ablate (drop)")):
            row = [pos, name]
            for k in (1, 3, 10, 30, 100, 300):
                e = c.get(f"{pos}/{mode}/k{k}")
                row.append(f"{e['top_mean']:+.3f} (rand {e['rand_mean']:+.3f})" if e else "—")
            al = c.get(f"{pos}/{mode}/ALL_NEURONS")
            row.append(f"{al['mean']:+.3f}" if al else "")
            out.append("| " + " | ".join(row) + " |")
    out.append("")
    out.append("Held-out order id only (sufficiency, agg8): " + ", ".join(
        f"k={k}: {c[f'agg8/suff/k{k}']['top_held']:+.3f}" for k in (1, 3, 10, 30, 100, 300)))
    conc = d["concentration"]
    a = conc["atp_train_mean"]
    out.append("")
    out.append(f"Concentration of |AtP| (train mean): top-1 {a['abs_frac_topk']['1']:.3f}, top-10 {a['abs_frac_topk']['10']:.3f}, "
               f"top-30 {a['abs_frac_topk']['30']:.3f}, top-100 {a['abs_frac_topk']['100']:.3f}, top-300 {a['abs_frac_topk']['300']:.3f}, "
               f"top-1000 {a['abs_frac_topk']['1000']:.3f}, top-10000 {a['abs_frac_topk']['10000']:.3f}; "
               f"neurons needed for 50/80/90% of |AtP|: {a['n_for_abs_frac']['0.5']}/{a['n_for_abs_frac']['0.8']}/{a['n_for_abs_frac']['0.9']} "
               f"(per-instance median for 50%: {conc['atp_per_instance_median_n_for_50pct']:.0f}).")
    out.append(f"Net signed AtP sum {a['net_sum']:.3f}; positive top-k as a fraction of the net: " +
               ", ".join(f"k={k}: {v:.2f}" for k, v in a["pos_topk_over_net"].items() if v is not None))
    out.append(f"In-sample (per-instance top-k) sufficiency: {d['insample_suff']}")
    lin = d["linearity"]
    out.append(f"AtP linearity: patching all neurons at agg8 = {np.mean([l['all_neurons_agg8_actual'] for l in lin]):.3f} vs sum of AtP {np.mean([l['atp_sum'] for l in lin]):.3f}.")
    ls = conc["layer_abs_share"]
    top_layers = sorted(range(len(ls)), key=lambda i: -ls[i])[:6]
    out.append("Layer share of |AtP| (top layers): " + ", ".join(f"L{i}: {ls[i]:.3f}" for i in top_layers))
    return "\n".join(out) + "\n"


def labels_table():
    L = J("labels_llama31_8b.json")
    g = J("gao_llama31_8b.json")
    if not L:
        return "(no labels)\n"
    from collections import Counter
    out = []
    for kind in ("top", "random"):
        ls = [x for x in L if x.get("kind", "top") == kind]
        if ls:
            out.append(f"{kind}: n={len(ls)} jobs={dict(Counter(x.get('jobs') for x in ls))} mean conf={np.mean([x.get('confidence') or 0 for x in ls]):.2f}")
    out.append("")
    out.append("| rank | neuron | label | negative pole | jobs | r(sim,real) general | damage avoided (general, vs mean-abl.) | r task | damage avoided (task) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for x in L:
        if x.get("kind", "top") != "top" or x["rank"] >= 30:
            continue
        j = x["rank"]
        cg = g["corr_general"].get(str(j)) if g else None
        gg = g["general"].get(str(j)) if g else None
        ts = g["task"]["single"].get(str(j)) if g else None
        out.append(f"| {j} | L{x['layer']}.{x['idx']} | {x.get('label')} | {str(x.get('negative_pole'))[:70]} | {x.get('jobs')} | "
                   f"{(cg or {}).get('all') if cg else '—'} | {r3(gg['all']['damage_avoided_vs_mean']) if gg and gg.get('all') else '—'} | "
                   f"{(ts or {}).get('r_task_acts') if ts else '—'} | {r3(ts['damage_avoided_vs_mean']) if ts else '—'} |")
    return "\n".join(out) + "\n"


def gao_summary():
    g = J("gao_llama31_8b.json")
    if not g:
        return "(no gao results)\n"
    kinds = [x.get("kind", "top") for x in sorted(J("labels_llama31_8b.json"), key=lambda x: x["rank"])]
    out = []
    for kind in ("top", "random"):
        idx = [j for j in range(len(kinds)) if kinds[j] == kind]
        if not idx:
            continue
        def col(f):
            v = [f(j) for j in idx]
            v = [x for x in v if x is not None and x == x]
            return (np.median(v), np.mean(v), len(v)) if v else (None, None, 0)
        rA = col(lambda j: g["corr_general"][str(j)]["all"])
        rR = col(lambda j: g["corr_general"][str(j)]["random_only"])
        dA = col(lambda j: (g["general"].get(str(j)) or {}).get("all", {}).get("damage_avoided_vs_mean") if g["general"].get(str(j)) else None)
        dR = col(lambda j: (g["general"].get(str(j)) or {}).get("random", {}) and g["general"][str(j)]["random"]["damage_avoided_vs_mean"])
        tR = col(lambda j: g["task"]["single"][str(j)]["r_task_acts"])
        tD = col(lambda j: g["task"]["single"][str(j)]["r_task_safe_minus_unsafe"])
        tK = col(lambda j: g["task"]["single"][str(j)]["damage_avoided_vs_mean"])
        out.append(f"| {kind} (n={len(idx)}) | {rA[0]:.2f} / {rA[1]:.2f} | {rR[0]:.2f} / {rR[1]:.2f} | {dA[0]:+.2f} / {dA[1]:+.2f} | "
                   f"{dR[0]:+.2f} / {dR[1]:+.2f} | {tR[0]:.2f} / {tR[1]:.2f} | {tD[0]:.2f} / {tD[1]:.2f} | {tK[0]:+.2f} / {tK[1]:+.2f} |")
    hdr = ["| neurons | r(sim,real) all general windows, median/mean | r random windows only | damage avoided general (all windows) | damage avoided general (random windows) | r task (agg positions) | r of SAFE−UNSAFE difference | damage avoided at decision |",
           "|---|---|---|---|---|---|---|---|"]
    out = hdr + out
    out.append("")
    out.append("Joint substitution on general text (shared random windows): " + "; ".join(
        f"{k}: damage avoided vs mean {v['damage_avoided_vs_mean']:+.3f}, vs zero {v['damage_avoided_vs_zero']:+.3f}" for k, v in g["joint_general"].items()))
    rows = g["task"]["per_instance"]
    out.append("")
    out.append("| set | real patch recovery (SAFE values into UNSAFE) | simulated patch recovery | decision KL sim (SAFE / UNSAFE prompt) | KL mean-abl. | damage avoided |")
    out.append("|---|---|---|---|---|---|")
    for sn in g["sets"]:
        rp = np.mean([r[f"{sn}/realpatch_recovery"] for r in rows]); sp = np.mean([r[f"{sn}/simpatch_recovery"] for r in rows])
        ks = [np.mean([r[f"{sn}/{w}/sim/KL"] for r in rows]) for w in ("safe", "unsafe")]
        km = [np.mean([r[f"{sn}/{w}/mean/KL"] for r in rows]) for w in ("safe", "unsafe")]
        da = 1 - (ks[0] + ks[1]) / (km[0] + km[1])
        out.append(f"| {sn} | {rp:+.3f} | {sp:+.3f} | {ks[0]:.4f} / {ks[1]:.4f} | {km[0]:.4f} / {km[1]:.4f} | {da:+.3f} |")
    return "\n".join(out) + "\n"


def t7_tables():
    d = J("t7_llama31_8b.json")
    if not d:
        return "(no t7)\n"
    out = ["| set | zero positive parts | zero negative parts | zero both | random sign-matched (pos / neg / both) | |activation| mass removed (pos / neg) |", "|---|---|---|---|---|---|"]
    for k, r in d["parts"].items():
        if k == "ALL":
            out.append(f"| ALL 458,752 neurons | {r['pos']['top']:+.3f} | {r['neg']['top']:+.3f} | {r['both']['top']:+.3f} | — | {r['pos']['mass']:.0f} / {r['neg']['mass']:.0f} |")
        else:
            out.append(f"| top-{k} by |AtP| | {r['pos']['top']:+.3f} | {r['neg']['top']:+.3f} | {r['both']['top']:+.3f} | "
                       f"{r['pos']['rand']:+.3f} / {r['neg']['rand']:+.3f} / {r['both']['rand']:+.3f} | {r['pos']['top_mass']:.1f} / {r['neg']['top_mass']:.1f} |")
    out.append("")
    out.append("Attribution sign split: " + "; ".join(
        f"top-{k}: {v['abs_attr_from_negative_activation']:.2f} of |AtP| from negative activations, {v['n_negative_side']} negative-side / {v['n_positive_side']} positive-side"
        for k, v in d["neg_share_of_attribution"].items()))
    out.append("")
    out.append("| Santa analog (zero both parts) | share of decision lost | random sign-matched |")
    out.append("|---|---|---|")
    for k, v in d["santa"].items():
        out.append(f"| {k} | {v['share_lost']:+.3f} | {v['rand']:+.3f} |")
    return "\n".join(out) + "\n"


def t1b_tables():
    d = J("t1b_llama31_8b.json")
    if not d or "summary" not in d:
        return "(t1b incomplete)\n"
    S = d["summary"]
    out = []
    names = {"pos": "aggregator positions (top-16)", "whead": "write heads (top-20)", "neuron": "MLP neurons (top-30)",
             "sae": "SAE L14 features (top-30)", "rhead": "read heads (top-12)"}
    out.append("Ablation-size curve (retention of the decision on the clean run; patch-ablation, mean over pairs):\n")
    for typ, s in S.items():
        cv = s["curve"]
        for m, c in cv.items():
            out.append(f"- {names[typ]} [{m}]: " + ", ".join(f"k={k}: {v:.3f}" for k, v in sorted(c.items(), key=lambda x: int(x[0]))))
    out.append("")
    out.append("| carrier | mean single suff | max single suff | mean single nec | max single nec | # suff>0.3 & nec<0.1 | corr(suff,nec) | Σ single nec | joint nec | Σ single suff | joint suff | compensation ratio (pooled / median) | self effect after ablation |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for typ, s in S.items():
        si, ad, co = s["single"], s["additivity"], s["compensation"]
        extra = f" (all heads ≥ layer: {co.get('pool_all_heads_ge_layer_pooled_ratio'):+.3f})" if typ == "rhead" and co.get("pool_all_heads_ge_layer_pooled_ratio") is not None else ""
        out.append(f"| {names[typ]} | {si['suff_mean']:+.3f} | {si['suff_max']:+.3f} | {si['nec_mean']:+.3f} | {si['nec_max']:+.3f} | "
                   f"{si['n_high_suff_low_nec(suff>0.3,nec<0.1)']}/{si['n']} | {si['corr_suff_nec'] if si['corr_suff_nec'] is None else round(si['corr_suff_nec'], 2)} | "
                   f"{ad['sum_single_nec']:+.3f} | {ad['joint_nec']:+.3f} | {ad['sum_single_suff']:+.3f} | {ad['joint_suff']:+.3f} | "
                   f"{r3(co['pooled_ratio'])} / {r3(co['median_ratio'])}{extra} | {r3(co.get('self_after_over_before'))} |")
    return "\n".join(out) + "\n"


def ov_table():
    d = J("ov_llama31_8b.json")
    if not d:
        return "(no ov)\n"
    out = ["| read head | set | n | OV gain top / rand | |Δlogit| per unit, account_role top / rand | safety_mode | subscription_tier |", "|---|---|---|---|---|---|---|"]
    for h, sets in d["results"].items():
        for name, v in sets.items():
            t, r = v["top"], v["rand_layer_matched"]
            cells = " | ".join(f"{t['abs_logitdiff_per_unit_mean'][s]:.4f} / {r['abs_logitdiff_per_unit_mean'][s]:.4f}" for s in t["abs_logitdiff_per_unit_mean"])
            out.append(f"| {h} | {name} | {v['n']} | {t['ov_gain_mean']:.3f} / {r['ov_gain_mean']:.3f} | {cells} |")
    return "\n".join(out) + "\n"


def split_line():
    d = J("t1_split_llama31_8b.json")
    if not d:
        return "(no split)\n"
    s = d["summary"]
    return (f"Attention vs MLP at agg8 (24 pairs): AtP attention share = {s['atp_attn_share_of_attn_plus_mlp']:.3f} "
            f"(AtP attn {s['atp_attn_mean']:+.3f}, MLP {s['atp_mlp_mean']:+.3f}); actual patch of all attention outputs at agg8 = "
            f"{s['patch_all_attn_mean']:+.3f}, all MLP outputs = {s['patch_all_mlp_mean']:+.3f}. Repo's direct-projection share (circ_components, T=14): 0.599.\n")


if __name__ == "__main__":
    sec = sys.argv[1]
    print({"causal": lambda: causal_table(sys.argv[2] if len(sys.argv) > 2 else "llama31_8b"), "labels": labels_table,
           "gao": gao_summary, "t7": t7_tables, "t1b": t1b_tables, "ov": ov_table, "split": split_line}[sec]())
