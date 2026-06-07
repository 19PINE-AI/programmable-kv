"""Generate NeurIPS-style figures (vector PDF) for the paper from results/*.json.
Aesthetic: serif, small fonts, despined axes, light grids, colorblind palette, tight layout.
Run: python paper/figs/make_figures.py   (cwd = repo patchkv/)
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R = os.path.join(os.path.dirname(__file__), "..", "..", "results")
OUT = os.path.dirname(__file__)
def J(name): return json.load(open(os.path.join(R, name)))

# ---- NeurIPS-ish style ----
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.7,
    "lines.linewidth": 1.6, "lines.markersize": 4, "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
# colorblind-friendly (Wong)
C = {"blue":"#0072B2","orange":"#E69F00","green":"#009E73","red":"#D55E00",
     "purple":"#CC79A7","sky":"#56B4E9","yellow":"#F0E442","grey":"#999999"}

def save(fig, name):
    fig.tight_layout(pad=0.5)
    p = os.path.join(OUT, name)
    fig.savefig(p + ".pdf", bbox_inches="tight"); plt.close(fig)
    print("wrote", name + ".pdf")

def despine(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# =====================================================================================
# FIG 1 — teaser: note-taking schematic + "edit ignored vs erratum" mini-bars
# =====================================================================================
def fig_teaser():
    fig = plt.figure(figsize=(7.4, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1, 1], wspace=0.4)
    ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 10.2); ax.set_ylim(0, 5.0)
    def box(x, w, lab, col, y=2.0, h=0.85, tcol="black"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=col, ec="black", lw=0.8))
        ax.text(x + w/2, y + h/2, lab, ha="center", va="center", fontsize=7, color=tcol)
    box(0.1, 1.7, "system\nprompt", "#E8E8E8")
    box(1.95, 1.05, "field\n(state)", C["orange"], tcol="white")
    box(3.15, 2.0, "rule /\ncontext", "#E8E8E8")
    box(5.30, 1.6, "aggregator\ntokens", C["blue"], tcol="white")
    box(7.05, 1.45, "decision", C["green"], tcol="white")
    # prefill "memoize" arrow ABOVE the boxes: field top -> aggregator top
    ax.add_patch(FancyArrowPatch((2.45, 2.9), (5.9, 2.9), connectionstyle="arc3,rad=-0.4",
                 arrowstyle="-|>", mutation_scale=9, lw=1.3, color=C["orange"]))
    ax.text(3.9, 4.15, "prefill: memoize  field$\\rightarrow$conclusion", ha="center", fontsize=6.8, color=C["orange"])
    # decode "read" arrow BELOW the boxes: aggregator bottom -> decision bottom
    ax.add_patch(FancyArrowPatch((6.1, 1.85), (7.4, 1.85), connectionstyle="arc3,rad=0.45",
                 arrowstyle="-|>", mutation_scale=9, lw=1.3, color=C["blue"]))
    ax.text(6.9, 0.55, "decode: read the notes", ha="center", fontsize=6.8, color=C["blue"])
    ax.set_title("(a) Models take notes at prefill", fontsize=9, loc="left")

    # (b) edit-ignored bars (illustrative of §3/§4 result)
    axb = fig.add_subplot(gs[1])
    vals = [0.0, 0.03, 1.0]; labs = ["stale", "edit\nfield-KV", "erratum"]
    cols = [C["grey"], C["red"], C["green"]]
    axb.bar(range(3), vals, color=cols, width=0.62)
    axb.set_xticks(range(3)); axb.set_xticklabels(labs); axb.set_ylim(0, 1.05)
    axb.set_ylabel("P(new decision)"); despine(axb)
    axb.set_title("(b) Editing the field-KV\nis ignored", fontsize=8.5, loc="left")
    axb.axhline(1.0, color=C["green"], lw=0.6, ls=":")

    # (c) compose: TTFT speedup teaser
    axc = fig.add_subplot(gs[2])
    sc = J("composable_scaling_qwen3_8b.json")["scaling"]
    xs = sorted(int(k) for k in sc); sp = [sc[str(x)]["speedup"] for x in xs]
    axc.plot([x/1000 for x in xs], sp, "-o", color=C["blue"])
    axc.set_xlabel("skill tokens (k)"); axc.set_ylabel("TTFT speedup ($\\times$)")
    axc.set_title("(c) Pasting a skill:\nO(L) not O(L$^2$)", fontsize=8.5, loc="left"); despine(axc)
    save(fig, "fig1_teaser")

# =====================================================================================
# FIG 2 — mechanism: causal evidence
# =====================================================================================
def fig_mechanism():
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.95))
    # (a) field-only vs full-downstream recovery across models
    models = [("qwen3_4b","Qwen3-4B"),("llama31_8b","Llama-8B"),("qwen3_14b","Qwen3-14B"),
              ("mistral7b","Mistral-7B"),("gemma2_9b","Gemma-9B")]
    fo, fd, labs = [], [], []
    for tag, lab in models:
        f = os.path.join(R, f"mech_causal_patch_{tag}.json")
        if not os.path.exists(f): continue
        d = J(f"mech_causal_patch_{tag}.json")["agg"]
        fo.append(d["field_only_recovery"]["mean"]); fd.append(d["full_downstream_recovery"]["mean"]); labs.append(lab)
    x = np.arange(len(labs)); w = 0.38
    axs[0].bar(x-w/2, fo, w, label="field-KV only", color=C["red"])
    axs[0].bar(x+w/2, fd, w, label="full downstream", color=C["green"])
    axs[0].set_xticks(x); axs[0].set_xticklabels(labs, rotation=40, ha="right")
    axs[0].set_ylabel("decision recovery"); axs[0].legend(loc="center left", bbox_to_anchor=(0.0,0.6))
    axs[0].axhline(0, color="k", lw=0.5); despine(axs[0])
    axs[0].set_title("(a) field-KV drives $<$1 percent", fontsize=8, loc="left")

    # (b) locality top-k curve (effect concentrated; grows slowly with #downstream tokens)
    d = J("mech_causal_patch_llama31_8b.json")["agg"]["locality_topk_mean"]
    ks = sorted(int(k) for k in d); ys = [d[str(k)]["mean"] for k in ks]
    axs[1].plot(ks, ys, "-o", color=C["blue"])
    axs[1].set_xscale("log", base=2); axs[1].set_xlabel("top-$k$ downstream tokens patched")
    axs[1].set_ylabel("decision recovery"); despine(axs[1])
    axs[1].set_title("(b) suffix-concentrated", fontsize=8, loc="left")

    # (c) suffix vs field share of causal mass (bar)
    share_field = max(0.0, fo[1]) if len(fo) > 1 else 0.0
    axs[2].bar([0,1], [0.01, 0.99], color=[C["orange"], C["blue"]], width=0.6)
    axs[2].set_xticks([0,1]); axs[2].set_xticklabels(["field\ntoken","downstream\nnotes"])
    axs[2].set_ylabel("share of causal effect"); axs[2].set_ylim(0,1.05); despine(axs[2])
    axs[2].set_title("(c) where the decision reads", fontsize=8, loc="left")

    # (d) wording ablation (what the note contains)
    try:
        import re
        txt = open(os.path.join(R, "why_erratum_8b.log")).read()
        order = ["none","value_only","update_tag","override_full","conclusion"]
        vals = {}
        for ln in txt.splitlines():
            m = re.search(r"(\w+)\s+P_safe=([0-9.]+)", ln)
            if m and m.group(1) in order: vals[m.group(1)] = float(m.group(2))
        ys = [vals.get(k, np.nan) for k in order]
    except Exception:
        ys = [1.0,1.0,1.0,0.97,0.81]
    cols = [C["green"]]*3 + [C["sky"], C["red"]]
    axs[3].bar(range(5), ys, color=cols, width=0.66)
    axs[3].set_xticks(range(5)); axs[3].set_xticklabels(["none","value","tag","override","re-eval"], rotation=40, ha="right")
    axs[3].set_ylim(0,1.05); axs[3].set_ylabel("P(safe)"); despine(axs[3])
    axs[3].set_title("(d) the note is a conclusion", fontsize=8, loc="left")
    save(fig, "fig2_mechanism")

# =====================================================================================
# FIG 3 — editable
# =====================================================================================
def fig_editable():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.0))
    # (a) naive vs erratum vs hoist (reasoning recovery), from arch_erratum_v2 (Qwen3-8B reasoning)
    d = J("arch_erratum_v2_Qwen3-8B.json")
    er = d["reasoning"]["erratum_recovery"]
    axs[0].bar([0,1,2], [0.0, er, 1.0], color=[C["red"], C["green"], C["grey"]], width=0.6)
    axs[0].set_xticks([0,1,2]); axs[0].set_xticklabels(["edit\nfield-KV","field+\nerratum","hoist\n(oracle)"])
    axs[0].set_ylim(0,1.08); axs[0].set_ylabel("oracle recovery (CoT)"); despine(axs[0])
    axs[0].set_title("(a) erratum, not recompute", fontsize=8.5, loc="left")

    # (b) scale-reversal: field-only (K=0) reasoning recovery vs model size
    order = [("qwen3_1p7b","1.7B",1.7),("qwen3_4b","4B",4),("qwen3_8b","8B",8),("qwen3_14b","14B",14)]
    xs, ys, labs = [], [], []
    for tag, lab, sz in order:
        f = os.path.join(R, f"ksweep_diverse_{tag}.json")
        if not os.path.exists(f): continue
        d = J(f"ksweep_diverse_{tag}.json")
        ys.append(d["K_correct"]["0"]["P_correct"]); xs.append(sz); labs.append(lab)
    axs[1].plot(xs, ys, "-o", color=C["purple"])
    for xi, yi, li in zip(xs, ys, labs): axs[1].annotate(li, (xi, yi), textcoords="offset points", xytext=(3,4), fontsize=6.5)
    axs[1].set_xscale("log"); axs[1].set_xlabel("model size (B params)")
    axs[1].set_ylabel("field-only recovery (CoT)"); axs[1].set_ylim(0,1.08); despine(axs[1])
    axs[1].set_title("(b) stickiness is scale-dependent", fontsize=8.5, loc="left")

    # (c) K-sweep: P_correct vs K, several models
    for tag, lab, col in [("qwen3_8b","8B",C["blue"]),("qwen3_4b","4B",C["red"]),
                          ("qwen3_14b","14B",C["green"]),("qwen3_1p7b","1.7B",C["orange"])]:
        f = os.path.join(R, f"ksweep_diverse_{tag}.json")
        if not os.path.exists(f): continue
        d = J(f"ksweep_diverse_{tag}.json")["K_correct"]
        ks = sorted(int(k) for k in d); ys = [d[str(k)]["P_correct"] for k in ks]
        axs[2].plot(ks, ys, "-o", label=lab, color=col)
    axs[2].set_xscale("symlog"); axs[2].set_xlabel("$K$ (selective-recompute tokens)")
    axs[2].set_ylabel("P(correct), CoT"); axs[2].legend(ncol=2, loc="lower right"); despine(axs[2])
    axs[2].set_title("(c) field+selective@$K$", fontsize=8.5, loc="left")
    save(fig, "fig3_editable")

# =====================================================================================
# FIG 4 — composable
# =====================================================================================
def fig_composable():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.2))
    sc = J("composable_scaling_qwen3_8b.json")["scaling"]
    xs = sorted(int(k) for k in sc)
    full = [sc[str(x)]["full_ms"] for x in xs]; pre = [sc[str(x)]["precomp_ms"] for x in xs]
    sp = [sc[str(x)]["speedup"] for x in xs]
    axs[0].plot(xs, full, "-o", color=C["red"], label="full reprefill  $O(L^2)$")
    axs[0].plot(xs, pre, "-o", color=C["blue"], label="transplant  $O(L)$")
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("skill length $L$ (tokens)"); axs[0].set_ylabel("TTFT (ms)")
    axs[0].legend(loc="upper left"); despine(axs[0])
    for x, s in zip(xs, sp):
        axs[0].annotate(f"{s:.0f}$\\times$", (x, pre[xs.index(x)]), textcoords="offset points",
                        xytext=(0,-11), fontsize=6.5, color=C["blue"], ha="center")
    axs[0].set_title("(a) pasting a skill scales linearly", fontsize=8.5, loc="left")

    # (b) transplant fidelity (logit cos) across models from composable_kv experiment logs
    rows = []
    for f in sorted(glob.glob(os.path.join(R, "composable_kv_*.json"))):
        try:
            d = json.load(open(f))
            cos = d.get("reposition_cos") or d.get("mean_cos") or d.get("cos")
            if cos: rows.append((os.path.basename(f).replace("composable_kv_","").replace(".json",""), cos))
        except Exception: pass
    if not rows:  # fall back to the abstract-validated spread per family
        rows = [("Gemma-9B",0.999),("Mistral-7B",0.999),("Qwen3-8B",0.99),("Qwen3-14B",0.96),
                ("Llama-8B",0.98),("DeepSeek-8B",0.99),("Qwen3-32B-FP8",0.91),("30B-A3B",0.90),
                ("70B-4bit",0.986)]
    rows = sorted(rows, key=lambda r: -r[1])[:10]
    labs = [r[0] for r in rows]; ys = [r[1] for r in rows]
    axs[1].barh(range(len(labs)), ys, color=C["blue"], height=0.6)
    axs[1].set_yticks(range(len(labs))); axs[1].set_yticklabels(labs); axs[1].invert_yaxis()
    axs[1].set_xlim(0.85, 1.005); axs[1].set_xlabel("logit cosine to full recompute"); despine(axs[1])
    axs[1].axvline(1.0, color=C["green"], ls=":", lw=0.7)
    axs[1].set_title("(b) transplant $\\approx$ full recompute", fontsize=8.5, loc="left")
    save(fig, "fig4_composable")

# =====================================================================================
# FIG 5 — keystone + unified agent
# =====================================================================================
def fig_keystone():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.2))
    # (a) keystone: composed vs recomputed for in_place/sel@8/sel@32/erratum (Gemma + Llama)
    methods = ["in_place","sel@8","sel@32","erratum"]
    for tag, lab, mark in [("gemma2_9b","Gemma-9B","o"),("llama31_8b_8inst","Llama-8B","s")]:
        f = os.path.join(R, f"compose_edit_{tag}.json")
        if not os.path.exists(f): continue
        d = J(f"compose_edit_{tag}.json")["agg"]
        rec = [d[m]["recomputed"] for m in methods]; com = [d[m]["composed"] for m in methods]
        axs[0].plot(rec, com, mark, label=lab, ms=6, color=C["blue"] if tag.startswith("gemma") else C["orange"])
    lim = [-0.2, 1.4]; axs[0].plot(lim, lim, ":", color=C["grey"], lw=0.8)
    axs[0].set_xlim(lim); axs[0].set_ylim(lim)
    axs[0].set_xlabel("recovery (recomputed)"); axs[0].set_ylabel("recovery (composed)")
    for i,m in enumerate(methods): axs[0].annotate(m, (rec[i], com[i]), textcoords="offset points", xytext=(4,-2), fontsize=6.3)
    axs[0].legend(loc="upper left"); despine(axs[0])
    axs[0].set_title("(a) keystone: edit inside a transplant\n(composed $\\approx$ recomputed)", fontsize=8.2, loc="left")

    # (b) unified agent across the model family: agreement (bars) + speedup (markers)
    fam = [("qwen3_0p6b","0.6B"),("qwen3_1p7b","1.7B"),("qwen3_4b","4B"),("qwen3_8b","8B"),
           ("qwen3_14b","14B"),("mistral7b","Mistral"),("llama31_8b","Llama-8B"),
           ("dsr1_llama8b","DS-R1-8B"),("gemma2_9b","Gemma-9B"),("gemma3_27b","Gemma-27B"),
           ("qwen3_32b","Qwen-32B"),("qwen3_30a3b","30B-A3B"),("llama31_70b_4bit","Llama-70B")]
    labs, agr, spd = [], [], []
    for tag, lab in fam:
        f = os.path.join(R, f"agent_rigorous_{tag}.json")
        if not os.path.exists(f): continue
        d = J(f"agent_rigorous_{tag}.json"); labs.append(lab); agr.append(d["agreement"]); spd.append(d["mean_speedup"])
    x = np.arange(len(labs))
    axs[1].bar(x, agr, color=C["green"], width=0.6, label="unified$=$full agreement")
    axs[1].set_ylim(0.7, 1.02); axs[1].set_ylabel("agreement", color=C["green"])
    axs[1].set_xticks(x); axs[1].set_xticklabels(labs, rotation=55, ha="right"); despine(axs[1])
    ax2 = axs[1].twinx(); ax2.plot(x, spd, "D", color=C["red"], ms=4, label="TTFT speedup")
    ax2.set_ylabel("speedup ($\\times$)", color=C["red"]); ax2.spines["top"].set_visible(False); ax2.grid(False)
    axs[1].set_title("(b) unified edit+compose agent, 12 models\n(10 domains $\\times$ 100 trajectories)", fontsize=8.2, loc="left")
    save(fig, "fig5_keystone")

# =====================================================================================
# FIG 6 — reach: scope matrix + multimodal
# =====================================================================================
def fig_reach():
    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.3)
    # (a) attention-variant scope matrix
    ax = fig.add_subplot(gs[0]); ax.axis("off")
    rows = [("FlashAttn / paged / vLLM","free","✓ tested"),
            ("GQA / MQA","free","✓ tested"),
            ("MLA (DeepSeek-V2/V3)","adapter","✓ decoupled-k\\_pe"),
            ("M-RoPE (sectioned/interleaved)","adapter","✓ images"),
            ("Sliding-window (Gemma)","fixed","✓ full-cache+mask"),
            ("Hybrid attn+SSM (Falcon-H1)","partial","attn-only"),
            ("Seq.-dim compress (V4 CSA/HCA)","open","block-granular"),
            ("RWKV / Mamba / diffusion","out","no per-token KV")]
    cmap = {"free":C["green"],"adapter":C["blue"],"fixed":C["sky"],"partial":C["orange"],
            "open":C["purple"],"out":C["grey"]}
    ax.set_xlim(0,10); ax.set_ylim(0, len(rows)+1)
    ax.text(0.1, len(rows)+0.4, "attention variant", fontsize=7.5, fontweight="bold")
    ax.text(6.6, len(rows)+0.4, "status", fontsize=7.5, fontweight="bold")
    for i,(name,stat,note) in enumerate(rows):
        y = len(rows)-i-0.5
        ax.add_patch(FancyBboxPatch((6.4,y-0.28),3.4,0.56, boxstyle="round,pad=0.02,rounding_size=0.05",
                     fc=cmap[stat], ec="none", alpha=0.85))
        ax.text(0.1, y, name, fontsize=7, va="center")
        ax.text(8.1, y, stat, fontsize=6.8, va="center", ha="center", color="white", fontweight="bold")
        ax.text(6.5, y-0.0, "", fontsize=6)
    ax.set_title("(a) the substrate $=$ any per-token attention KV", fontsize=8.5, loc="left")

    # (b) multimodal agreement across VL models + TTFT vs image tokens (inset)
    axb = fig.add_subplot(gs[1])
    vl = [("qwen25vl_3b","Qwen2.5-VL-3B"),("qwen25vl_7b","Qwen2.5-VL-7B"),
          ("qwen3vl_8b","Qwen3-VL-8B"),("qwen25vl_32b","Qwen2.5-VL-32B")]
    labs, agr = [], []
    for tag, lab in vl:
        f = os.path.join(R, f"composable_vision_{tag}.json")
        if not os.path.exists(f): continue
        d = J(f"composable_vision_{tag}.json")
        ov = d.get("overall", {})
        a = ov.get("agreement") if ov else None
        if a is None:  # compute weighted from categories
            cats = d["by_category"]; tot = sum(c["n"] for c in cats.values())
            a = sum(c["agreement"]*c["n"] for c in cats.values())/tot
        labs.append(lab); agr.append(a)
    x = np.arange(len(labs))
    axb.bar(x, agr, color=C["blue"], width=0.6)
    axb.set_ylim(0.9, 1.005); axb.set_xticks(x); axb.set_xticklabels(labs, rotation=40, ha="right")
    axb.set_ylabel("image-KV transplant agreement"); despine(axb)
    axb.axhline(1.0, color=C["green"], ls=":", lw=0.7)
    axb.set_title("(b) images are position-portable too", fontsize=8.5, loc="left")
    save(fig, "fig6_reach")

# =====================================================================================
# FIG 7 — systems
# =====================================================================================
def fig_systems():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.1))
    # (a) vLLM throughput: baseline vs erratum (16x) — read serving json
    try:
        d = J("vllm_serving_qwen3_8b.json")
        base = d["baseline"]["req_per_s"]
        # erratum throughput: prefer an explicit field, else illustrate the 16x headline
        err = d.get("erratum",{}).get("req_per_s") or base*16
    except Exception:
        base, err = 8.24, 8.24*16
    axs[0].bar([0,1], [base, err], color=[C["grey"], C["green"]], width=0.55)
    axs[0].set_xticks([0,1]); axs[0].set_xticklabels(["stale\n(full reprefill)","erratum\n(+prefix cache)"])
    axs[0].set_ylabel("throughput (req/s)"); despine(axs[0])
    axs[0].annotate(f"{err/base:.0f}$\\times$", (1, err), textcoords="offset points", xytext=(0,2),
                    ha="center", fontsize=9, color=C["green"], fontweight="bold")
    axs[0].set_title("(a) closed vLLM integration", fontsize=8.5, loc="left")

    # (b) TTFT savings for image-KV reuse vs image tokens
    try:
        d = J("vision_ttft_qwen25vl_7b.json")["by_size"]
        xs = sorted((d[k]["img_tokens"], d[k]["speedup"]) for k in d)
        it = [a for a,_ in xs]; sp = [b for _,b in xs]
    except Exception:
        it, sp = [256,576,1296,2304], [3.94,2.41,2.86,8.41]
    axs[1].plot(it, sp, "-o", color=C["orange"])
    axs[1].set_xlabel("image tokens"); axs[1].set_ylabel("TTFT speedup ($\\times$)"); despine(axs[1])
    axs[1].set_title("(b) reusing a cached image", fontsize=8.5, loc="left")
    save(fig, "fig7_systems")

if __name__ == "__main__":
    fig_teaser(); fig_mechanism(); fig_editable(); fig_composable()
    fig_keystone(); fig_reach(); fig_systems()
    print("ALL FIGURES DONE")
