"""Analyze all memory-KV experiment outputs with statistical rigor.
Reads results/e*.jsonl, applies cluster bootstrap / TOST / GEE / McNemar / BH-FDR, writes
results/summary.json and prints a report. Run: python analyze.py
"""
import os, sys, json, glob, math
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import stats as S

R = os.path.join(os.path.dirname(__file__), "results")
DELTA = 0.03  # pre-registered decision-equivalence margin
COS_EQ = 0.98


def load(prefix):
    recs = []
    for fp in sorted(glob.glob(os.path.join(R, f"{prefix}_*.jsonl"))):
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a partially-written trailing line during live runs
    return recs


def rate(recs, key):
    return float(np.mean([r[key] for r in recs])) if recs else float("nan")


def boot_rate(recs, key, cluster="persona"):
    if not recs:
        return (float("nan"),) * 4
    return S.cluster_bootstrap_stat(recs, lambda rs: np.mean([r[key] for r in rs]), cluster, n_boot=4000)


# ----------------------------- E2 -----------------------------
def analyze_e2():
    recs = load("e2")
    if not recs:
        return {}
    # primary endpoint = dec_agree (yes/no decision governance). top1_agree saturates (argmax is a
    # format token); cos is dominated by the bulk vector and is far less sensitive than dec_agree.
    out = {"seam_doseresponse": {}, "equivalence": {}, "naive_control": {}, "min_seam_equiv": {}}
    models = sorted(set(r["model"] for r in recs))
    for m in models:
        mr = [r for r in recs if r["model"] == m]
        for pl in ("early", "late"):
            pr = [r for r in mr if r["placement"] == pl]
            if not pr:
                continue
            key = f"{m}|{pl}"
            seam_vals = sorted(set(int(r["method"][4:]) for r in pr if r["method"].startswith("seam")))
            dr = {}; min_seam = None
            for sv in seam_vals:
                sr = [r for r in pr if r["method"] == f"seam{sv}"]
                cos = boot_rate(sr, "cos"); da = boot_rate(sr, "dec_agree")
                eqv = (1 - da[1]) <= DELTA          # decision-equivalence: disagreement upper-CI <= delta
                dr[sv] = dict(cos=round(cos[0], 4), cos_lo=round(cos[1], 4),
                              dec_agree=round(da[0], 4), dec_agree_lo=round(da[1], 4),
                              top1_agree=round(rate(sr, "top1_agree"), 4),
                              equiv_decision=bool(eqv), equiv_cos=bool(cos[1] >= COS_EQ), n=len(sr))
                if eqv and min_seam is None:
                    min_seam = sv
            out["seam_doseresponse"][key] = dr
            out["min_seam_equiv"][key] = min_seam
            # headline equivalence at the smallest equivalent seam (proposed deployment)
            if min_seam is not None:
                out["equivalence"][key] = dr[min_seam]
            # naive control (no re-rotation) by placement
            nz = [r for r in pr if r["method"] == "naive"]
            if nz:
                da = boot_rate(nz, "dec_agree")
                out["naive_control"][key] = dict(cos=round(rate(nz, "cos"), 4),
                    dec_agree=round(da[0], 4), dec_agree_lo=round(da[1], 4), n=len(nz))
    return out


# ----------------------------- E1 -----------------------------
def analyze_e1():
    recs = load("e1")
    if not recs:
        return {}
    out = {"accuracy": {}, "accuracy_by_len": {}, "placement_gee": {}, "margin": {},
           "oracle_competence": {}, "fdr": {}}
    models = sorted(set(r["model"] for r in recs))
    gee_pvals = []  # (key, term, p) for BH-FDR across placement effects
    for m in models:
        for reg in sorted(set(r["reasoning"] for r in recs if r["model"] == m)):
            cell = [r for r in recs if r["model"] == m and r["reasoning"] == reg]
            key = f"{m}|{reg}"
            # accuracy by (placement, nf) pooled over length, and by (placement, nf, mtotal)
            acc = {}; acc_len = {}
            mts = sorted(set(r["mtotal"] for r in cell))
            for pl in ("early", "late"):
                for nf in sorted(set(r["n_facts"] for r in cell)):
                    sub = [r for r in cell if r["placement"] == pl and r["n_facts"] == nf]
                    if sub:
                        b = boot_rate(sub, "correct")
                        acc[f"{pl}_nf{nf}"] = dict(acc=round(b[0], 3), lo=round(b[1], 3), hi=round(b[2], 3), n=len(sub))
                    for mt in mts:
                        s2 = [r for r in cell if r["placement"] == pl and r["n_facts"] == nf and r["mtotal"] == mt]
                        if s2:
                            acc_len[f"{pl}_nf{nf}_mt{mt}"] = dict(acc=round(rate(s2, "correct"), 3), n=len(s2))
            out["accuracy"][key] = acc
            out["accuracy_by_len"][key] = acc_len
            # oracle competence (pooled) for the inclusion gate
            comp = rate(cell, "correct")
            out["oracle_competence"][key] = round(comp, 3)
            # placement effect via GEE on COMPETENT CoT cells only (pre-registered inclusion gate >=0.80)
            try:
                ys = [r["correct"] for r in cell]
                if reg == "cot" and comp >= 0.80 and 0 < sum(ys) < len(ys):
                    g = S.gee_logit(cell, "correct", "C(placement) * n_facts + np.log(mtotal)", cluster_key="persona")
                    terms = {}
                    for k in g["params"]:
                        if "placement" in k:
                            terms[k] = dict(coef=round(g["params"][k], 3), p=round(g["pvalues"][k], 4),
                                            ci=[round(g["ci_lo"][k], 3), round(g["ci_hi"][k], 3)])
                            gee_pvals.append((key, k, g["pvalues"][k]))
                    out["placement_gee"][key] = dict(terms=terms, n=g["n"], n_clusters=g["n_clusters"])
            except Exception as e:
                out["placement_gee"][key] = {"error": str(e)[:120]}
            # direct-mode gold margin (paired early vs late) with cluster bootstrap
            if reg == "direct":
                pers = defaultdict(dict)
                for r in cell:
                    if "margin_gold" in r:
                        pers[r["persona"]].setdefault(r["placement"], []).append(r["margin_gold"])
                diffs = [np.mean(v["early"]) - np.mean(v["late"]) for v in pers.values()
                         if "early" in v and "late" in v]
                if diffs:
                    out["margin"][key] = dict(mean_early_minus_late=round(float(np.mean(diffs)), 3), n=len(diffs))
    # BH-FDR across the confirmatory family (competent CoT placement terms; drop any nan)
    gee_pvals = [(k, t, p) for (k, t, p) in gee_pvals if p == p]  # filter nan
    if gee_pvals:
        rej, adj = S.bh_fdr([p for _, _, p in gee_pvals])
        out["fdr"] = [dict(key=k, term=t, p=round(p, 4), p_adj=round(pa, 4), reject=bool(rj))
                      for (k, t, p), pa, rj in zip(gee_pvals, adj, rej)]
    return out


# ----------------------------- E3 -----------------------------
def analyze_e3():
    recs = load("e3")
    if not recs:
        return {}
    out = {"by_model": {}, "scale_inplace": {}}
    models = sorted(set(r["model"] for r in recs))
    methods = ["full_recompute", "stale", "in_place", "erratum", "recompile_chunk", "selective@4", "selective@16"]
    for m in models:
        mr = [r for r in recs if r["model"] == m]
        # join oracle (full_recompute) per persona
        oracle = {r["persona"]: r["pred"] for r in mr if r["method"] == "full_recompute"}
        per_method = {}
        for meth in methods:
            sub = [r for r in mr if r["method"] == meth]
            if not sub:
                continue
            # agreement with oracle + correctness + cost
            for r in sub:
                r["_agree_oracle"] = int(r["pred"] == oracle.get(r["persona"], r["pred"]))
            corr = boot_rate(sub, "correct"); agr = boot_rate(sub, "_agree_oracle")
            per_method[meth] = dict(correct=round(corr[0], 3), correct_lo=round(corr[1], 3),
                                    agree_oracle=round(agr[0], 3), agree_lo=round(agr[1], 3),
                                    recompute_tok=round(float(np.median([r["recompute_tok"] for r in sub])), 1),
                                    n=len(sub))
        out["by_model"][m] = per_method
        # McNemar in_place vs erratum (correctness), BH later
        ip = {r["persona"]: r["correct"] for r in mr if r["method"] == "in_place"}
        er = {r["persona"]: r["correct"] for r in mr if r["method"] == "erratum"}
        common = set(ip) & set(er)
        b = sum(1 for p in common if er[p] and not ip[p]); c = sum(1 for p in common if ip[p] and not er[p])
        out["by_model"][m]["_mcnemar_inplace_vs_erratum"] = S.mcnemar_exact(b, c)
        # scale: in_place correctness per model size
        ipr = [r for r in mr if r["method"] == "in_place"]
        if ipr:
            out["scale_inplace"][m] = round(rate(ipr, "correct"), 3)
    return out


# ----------------------------- E4 -----------------------------
def analyze_e4():
    recs = load("e4")
    if not recs:
        return {}
    out = {"by_model": {}}
    for m in sorted(set(r["model"] for r in recs)):
        mr = [r for r in recs if r["model"] == m]
        per_S = {}
        for sval in sorted(set(r["S"] for r in mr)):
            sub = [r for r in mr if r["S"] == sval]
            cos = boot_rate(sub, "cos"); a1 = boot_rate(sub, "top1_agree"); da = boot_rate(sub, "dec_agree")
            per_S[sval] = dict(cos=round(cos[0], 4), cos_lo=round(cos[1], 4),
                               top1_agree=round(a1[0], 4), dec_agree=round(da[0], 4),
                               edit_cost_tok=round(float(np.median([r["edit_cost_tok"] for r in sub])), 1),
                               full_cost_tok=round(float(np.median([r["full_cost_tok"] for r in sub])), 1),
                               n=len(sub))
        out["by_model"][m] = per_S
    return out


# ----------------------------- E5 -----------------------------
def analyze_e5():
    recs = load("e5")
    if not recs:
        return {}
    out = {"by_model": {}}
    for m in sorted(set(r["model"] for r in recs)):
        mr = [r for r in recs if r["model"] == m]
        def med(k):
            return round(float(np.median([r[k] for r in mr])), 2)
        # cumulative per session
        sess = defaultdict(lambda: defaultdict(float))
        for r in mr:
            for meth in ("oracle", "front", "end", "proposed"):
                sess[r["session"]][meth] += r[f"ttft_{meth}"]
        spd_front = np.median([sess[s]["front"] / sess[s]["proposed"] for s in sess])
        spd_end = np.median([sess[s]["end"] / sess[s]["proposed"] for s in sess])
        spd_oracle = np.median([sess[s]["oracle"] / sess[s]["proposed"] for s in sess])
        cot = [r for r in mr if "cot_agree" in r]
        out["by_model"][m] = dict(
            n_decisions=len(mr), n_sessions=len(sess),
            ttft_proposed_ms=med("ttft_proposed"), ttft_front_ms=med("ttft_front"),
            ttft_end_ms=med("ttft_end"), ttft_oracle_ms=med("ttft_oracle"),
            cum_speedup_vs_front=round(float(spd_front), 2), cum_speedup_vs_end=round(float(spd_end), 2),
            cum_speedup_vs_oracle=round(float(spd_oracle), 2),
            proposed_top1_agree=round(rate(mr, "top1_agree"), 4), proposed_cos=round(rate(mr, "cos"), 4),
            cot_agree=round(rate(cot, "cot_agree"), 3) if cot else None, n_cot=len(cot))
    return out


def analyze_negctrl():
    recs = load("negctrl")
    if not recs:
        return {}
    out = {}
    for m in sorted(set(r["model"] for r in recs)):
        mr = [r for r in recs if r["model"] == m]
        out[m] = {}
        for meth in sorted(set(r["method"] for r in mr)):
            sub = [r for r in mr if r["method"] == meth]
            b = boot_rate(sub, "stable")
            out[m][meth] = dict(stable=round(b[0], 3), lo=round(b[1], 3), n=len(sub))
    return out


def analyze_locomo():
    recs = load("locomo")
    if not recs:
        return {}
    out = {}
    for m in sorted(set(r["model"] for r in recs)):
        mr = [r for r in recs if r["model"] == m]
        cf = boot_rate(mr, "correct_full", cluster="conv")
        ct = boot_rate(mr, "correct_transplant", cluster="conv")
        # paired accuracy-parity test (transplant vs full) + answer-token fidelity
        eq = S.tost_proportions_paired(mr, "correct_transplant", "correct_full", DELTA,
                                       cluster_key="conv", n_boot=4000)
        out[m] = dict(n=len(mr), n_conv=len({r["conv"] for r in mr}),
                      acc_full=round(cf[0], 3), acc_full_ci=[round(cf[1], 3), round(cf[2], 3)],
                      acc_transplant=round(ct[0], 3), acc_transplant_ci=[round(ct[1], 3), round(ct[2], 3)],
                      acc_parity_diff=round(eq["diff"], 3), acc_parity_ci=[round(eq["lo"], 3), round(eq["hi"], 3)],
                      acc_parity_equivalent=bool(eq["equivalent"]),
                      ans_cos=round(rate(mr, "ans_cos"), 4), ans_top1_agree=round(rate(mr, "ans_top1_agree"), 3),
                      answer_agree=round(rate(mr, "answer_agree"), 3),
                      median_mem_tok=int(np.median([r["L_mem"] for r in mr])))
    return out


def main():
    summary = dict(e1=analyze_e1(), e2=analyze_e2(), e3=analyze_e3(), e4=analyze_e4(),
                   e5=analyze_e5(), negctrl=analyze_negctrl(), locomo=analyze_locomo())
    json.dump(summary, open(os.path.join(R, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("ANALYZE_DONE")


if __name__ == "__main__":
    main()
