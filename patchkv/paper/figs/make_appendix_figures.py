"""Appendix figures (NeurIPS style) for the paper — frontier, heatmaps, grouped bars.
Run: python paper/figs/make_appendix_figures.py  (cwd = repo patchkv/)
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

R = os.path.join(os.path.dirname(__file__), "..", "..", "results")
OUT = os.path.dirname(__file__)
def J(n): return json.load(open(os.path.join(R, n)))
def has(n): return os.path.exists(os.path.join(R, n))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.7,
    "lines.linewidth": 1.6, "lines.markersize": 5, "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
C = {"blue":"#0072B2","orange":"#E69F00","green":"#009E73","red":"#D55E00",
     "purple":"#CC79A7","sky":"#56B4E9","yellow":"#F0E442","grey":"#999999"}
# pleasant sequential map (white -> teal -> deep blue)
CMAP = LinearSegmentedColormap.from_list("tealblue", ["#f7fbff","#9ecae1","#2171b5","#08306b"])
GYR = LinearSegmentedColormap.from_list("gyr", ["#d55e00","#f0e442","#009e73"])  # red->yellow->green

def save(fig, name):
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight"); plt.close(fig)
    print("wrote", name + ".pdf")
def despine(ax): ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ---------- A1: baseline frontier (cost vs correctness, Pareto) ----------
def figA1():
    d = J("baseline_table_qwen3_8b.json")["methods"]
    name_map = {"full_reprefill":"full reprefill","stale":"stale","in_place":"in-place edit",
                "cacheblend@15%":"CacheBlend@15%","hoist_to_end":"hoist-to-end","erratum":"erratum",
                "field+erratum":"field+erratum"}
    surgery = {"hoist_to_end"}  # needs prompt surgery
    # per-method label offsets (points cluster at top-left and bottom-left)
    off = {"full_reprefill":(8,-2),"stale":(6,8),"in_place":(8,-12),"cacheblend@15%":(8,-2),
           "hoist_to_end":(-6,12),"erratum":(8,10),"field+erratum":(8,-13)}
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for k, v in d.items():
        x, y = v["recompute_frac"]*100, v["P_correct"]
        col = C["green"] if y >= 0.99 else (C["red"] if y < 0.2 else C["orange"])
        mk = "D" if k in surgery else "o"
        ax.scatter(x, y, s=70, marker=mk, color=col, edgecolor="black", lw=0.7, zorder=3)
        ax.annotate(name_map[k] + (" *" if k in surgery else ""), (x, y),
                    textcoords="offset points", xytext=off.get(k,(6,4)), fontsize=7)
    ax.set_xlabel("recompute (\\% of context)"); ax.set_ylabel("P(correct decision)")
    ax.set_ylim(-0.08, 1.16); ax.set_xlim(-5, 116); despine(ax)
    ax.text(0.98, 0.02, "$*$ requires prompt surgery", transform=ax.transAxes, ha="right",
            fontsize=6.8, color=C["grey"])
    ax.set_title("Baseline frontier (Qwen3-8B, 8 tasks)", fontsize=9, loc="left")
    save(fig, "figA1_frontier")

# ---------- A2: K-sweep heatmap (model x K -> P_correct) ----------
def figA2():
    order = [("qwen3_1p7b","Qwen3-1.7B"),("qwen3_4b","Qwen3-4B"),("qwen3_8b","Qwen3-8B"),
             ("qwen3_14b","Qwen3-14B"),("llama31_8b","Llama-3.1-8B"),("gemma2_9b","Gemma-2-9B"),
             ("dsr1_llama8b","DS-R1-Llama-8B")]
    Ks = [0,4,8,16,32,64]; rows, labs = [], []
    for tag, lab in order:
        f = f"ksweep_diverse_{tag}.json"
        if not has(f): continue
        d = J(f)["K_correct"]; rows.append([d[str(k)]["P_correct"] for k in Ks]); labs.append(lab)
    M = np.array(rows)
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    im = ax.imshow(M, cmap=GYR, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(Ks))); ax.set_xticklabels(["$K{=}%d$" % k for k in Ks])
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6.6,
                    color="black" if M[i,j]>0.45 else "white")
    ax.set_title("field+selective@$K$ recovery under reasoning", fontsize=8.6, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("P(correct)", fontsize=7.5)
    ax.grid(False)
    save(fig, "figA2_ksweep")

# ---------- A3: composable agreement heatmap (model x content-type) ----------
def figA3():
    models = [("qwen3_8b","Qwen3-8B"),("mistral7b","Mistral-7B"),("llama31_8b","Llama-3.1-8B"),
              ("gemma2_9b","Gemma-2-9B"),("gemma3_27b","Gemma-3-27B"),("qwen3_32b","Qwen3-32B-FP8"),
              ("qwen3_30a3b","Qwen3-30B-A3B")]
    cols = ["facts (early)","facts (late)","agentic tool-call"]
    rows, labs = [], []
    for tag, lab in models:
        ff, fa = f"composable_facts_{tag}.json", f"composable_agentic_{tag}.json"
        fe = J(ff)["results"]["early"]["agreement"] if has(ff) else np.nan
        fl = J(ff)["results"]["late"]["agreement"] if has(ff) else np.nan
        ag = J(fa)["toolcall_agreement"] if has(fa) else np.nan
        if np.all(np.isnan([fe,fl,ag])): continue
        rows.append([fe,fl,ag]); labs.append(lab)
    M = np.array(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    im = ax.imshow(M, cmap=GYR, vmin=0.3, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=20, ha="right")
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i,j]
            ax.text(j, i, "--" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                    fontsize=6.8, color="black" if (not np.isnan(v) and v>0.62) else "white")
    ax.set_title("Composable: decision agreement vs. full recompute", fontsize=8.4, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("agreement", fontsize=7.5)
    ax.grid(False)
    save(fig, "figA3_scorecard")

# ---------- A4: architecture erratum recovery (attention vs hybrid vs SSM) ----------
def figA4():
    items = [("Qwen3-8B","attention-GQA",C["blue"]),("qwen3_14b","attention-GQA",C["blue"]),
             ("llama31_8b","attention-GQA",C["blue"]),("mistral7b","attention-GQA",C["blue"]),
             ("gemma2_9b","attn (sliding)",C["sky"]),("gemma3_27b_bf16","attn (sliding)",C["sky"]),
             ("Falcon-H1-1_5B-Instruct","hybrid attn+SSM",C["orange"]),
             ("falcon-mamba-7b-instruct","pure SSM",C["red"])]
    labs, ys, cs = [], [], []
    for tag, cls, col in items:
        f = f"arch_erratum_v2_{tag}.json"
        if not has(f): continue
        d = J(f); rec = d.get("reasoning",{}).get("erratum_recovery")
        if rec is None: rec = 0.0
        nm = d.get("arch", tag).split("(")[0].strip()
        labs.append(tag.replace("arch_erratum_v2_","").replace("-Instruct","").replace("_"," ")); ys.append(rec); cs.append(col)
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    ax.bar(range(len(labs)), ys, color=cs, width=0.66, edgecolor="black", lw=0.5)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=40, ha="right")
    ax.set_ylim(0,1.08); ax.set_ylabel("erratum recovery (CoT)"); despine(ax)
    import matplotlib.patches as mp
    handles = [mp.Patch(color=C["blue"],label="attention (GQA)"), mp.Patch(color=C["sky"],label="sliding-window"),
               mp.Patch(color=C["orange"],label="hybrid attn+SSM"), mp.Patch(color=C["red"],label="pure SSM")]
    ax.legend(handles=handles, ncol=2, loc="lower left", fontsize=6.6)
    ax.set_title("editkv is an attention-architecture method", fontsize=8.6, loc="left")
    save(fig, "figA4_architecture")

# ---------- A5: multimodal per-category (full vs precompiled) ----------
def figA5():
    vl = [("qwen25vl_3b","Qwen2.5-VL-3B"),("qwen25vl_7b","Qwen2.5-VL-7B"),
          ("qwen3vl_8b","Qwen3-VL-8B"),("qwen25vl_32b","Qwen2.5-VL-32B")]
    cats = ["perception","reasoning","agentic"]
    fig, axs = plt.subplots(1, len(vl), figsize=(7.4, 2.3), sharey=True)
    for ax,(tag,lab) in zip(axs, vl):
        f = f"composable_vision_{tag}.json"
        if not has(f): ax.axis("off"); continue
        bc = J(f)["by_category"]; x = np.arange(len(cats)); w=0.38
        full = [bc.get(c,{}).get("full",np.nan) for c in cats]
        pre  = [bc.get(c,{}).get("precompiled",np.nan) for c in cats]
        ax.bar(x-w/2, full, w, label="full", color=C["grey"])
        ax.bar(x+w/2, pre, w, label="transplant", color=C["blue"])
        ax.set_xticks(x); ax.set_xticklabels(cats, rotation=30, ha="right"); ax.set_ylim(0,1.08)
        ax.set_title(lab, fontsize=8); despine(ax)
    axs[0].set_ylabel("VQA accuracy"); axs[0].legend(loc="upper left", fontsize=6.8)
    fig.suptitle("Image-KV transplant by task category", fontsize=9, x=0.02, ha="left")
    save(fig, "figA5_multimodal")

# ---------- A6: MLA fidelity + composable scaling across models ----------
def figA6():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
    # MLA
    mla = [("dsv2lite_mla","DeepSeek-V2-Lite"),("dscoderv2_mla","DeepSeek-Coder-V2-Lite")]
    labs, agr, cos = [], [], []
    for tag, lab in mla:
        f = f"mla_composable_{tag}.json"
        if not has(f): continue
        d = J(f); labs.append(lab); agr.append(d["composed_vs_full_agreement"]); cos.append(d["mean_logit_cos"])
    x = np.arange(len(labs)); w=0.36
    axs[0].bar(x-w/2, agr, w, label="decision agreement", color=C["green"])
    axs[0].bar(x+w/2, cos, w, label="logit cosine", color=C["blue"])
    axs[0].set_xticks(x); axs[0].set_xticklabels(labs, rotation=15, ha="right"); axs[0].set_ylim(0,1.08)
    axs[0].legend(loc="lower left", fontsize=6.8); despine(axs[0])
    axs[0].set_title("(a) MLA transplant (decoupled-$k_{pe}$ adapter)", fontsize=8.2, loc="left")
    # scaling across models
    for tag, lab, col in [("qwen3_8b","Qwen3-8B",C["blue"]),("llama31_8b","Llama-3.1-8B",C["orange"]),
                          ("qwen3_1p7b","Qwen3-1.7B",C["green"]),("dsr1_llama8b","DS-R1-8B",C["purple"])]:
        f = f"composable_scaling_{tag}.json"
        if not has(f): continue
        sc = J(f)["scaling"]; xs=sorted(int(k) for k in sc); sp=[sc[str(x)]["speedup"] for x in xs]
        axs[1].plot(xs, sp, "-o", label=lab, color=col)
    axs[1].set_xscale("log"); axs[1].set_xlabel("skill length (tokens)"); axs[1].set_ylabel("TTFT speedup ($\\times$)")
    axs[1].legend(loc="upper left", fontsize=6.8); despine(axs[1])
    axs[1].set_title("(b) transplant TTFT speedup across models", fontsize=8.2, loc="left")
    save(fig, "figA6_mla_scaling")

if __name__ == "__main__":
    figA1(); figA2(); figA3(); figA4(); figA5(); figA6()
    print("APPENDIX FIGURES DONE")
