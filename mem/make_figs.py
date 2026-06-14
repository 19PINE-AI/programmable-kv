"""Generate figures for the memory-KV experiments from results/*.jsonl (Times-coherent style)."""
import os, sys, json, glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Style matched to paper/figs/make_figures.py so the memory figures are visually
# consistent with the rest of the paper (STIX serif, Wong palette, same sizes).
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 4, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8, "axes.edgecolor": "#3a3a3a",
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "lines.linewidth": 1.9, "lines.markersize": 4.5, "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.16, "grid.linewidth": 0.6, "grid.color": "#8a8a8a",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.axisbelow": True, "figure.facecolor": "white", "savefig.facecolor": "white",
})
R = os.path.join(os.path.dirname(__file__), "results")
# Render straight into the paper's figure directory (the single source of truth the
# LaTeX build includes) so the committed PDFs are always exactly this script's output
# from results/*.jsonl. Override with FIGS_OUT to render elsewhere.
F = os.environ.get("FIGS_OUT") or os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "paper", "figs"))
os.makedirs(F, exist_ok=True)
# colorblind-friendly (Wong), matching the paper palette
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", gray="#9a9a9a", purple="#CC79A7")


def load(prefix):
    recs = []
    for fp in sorted(glob.glob(os.path.join(R, f"{prefix}_*.jsonl"))):
        for line in open(fp):
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
    return recs


def short(m):
    return m.split("/")[-1].replace("-Instruct", "").replace("unsloth_", "").replace("Meta-", "")


def fig_e2_seam():
    recs = load("e2")
    if not recs:
        return
    models = sorted(set(r["model"] for r in recs))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
    # (a) seam dose-response: dec_agree vs seam, late placement, per model
    ax = axes[0]
    for i, m in enumerate(models):
        seams = sorted(set(int(r["method"][4:]) for r in recs if r["method"].startswith("seam") and r["model"] == m))
        ys = []
        for s in seams:
            sr = [r for r in recs if r["model"] == m and r["placement"] == "late" and r["method"] == f"seam{s}"]
            ys.append(np.mean([r["dec_agree"] for r in sr]) if sr else np.nan)
        ax.plot(seams, ys, "-o", ms=4, label=short(m))
    ax.axhline(0.97, ls="--", c=C["gray"], lw=1)
    ax.set_xlabel("seam-repair tokens $K$"); ax.set_ylabel("decision agreement vs full")
    ax.set_title("(a) Seam repair (late placement)"); ax.set_ylim(0, 1.03)
    ax.legend(fontsize=6, ncol=2)
    # (b) naive vs rotated(seam0) vs seam1, late, decision agreement, per model
    ax = axes[1]
    labels = ["naive\n(no rotate)", "rotated\n(seam0)", "rotated\n+seam1"]
    x = np.arange(len(models)); w = 0.26
    for j, (meth, col) in enumerate([("naive", C["red"]), ("seam0", C["orange"]), ("seam1", C["green"])]):
        ys = [np.mean([r["dec_agree"] for r in recs if r["model"] == m and r["placement"] == "late" and r["method"] == meth] or [np.nan]) for m in models]
        ax.bar(x + (j - 1) * w, ys, w, color=col, label=labels[j])
    ax.set_xticks(x); ax.set_xticklabels([short(m) for m in models], rotation=35, ha="right", fontsize=6)
    ax.set_ylabel("decision agreement vs full"); ax.set_title("(b) Re-rotation + 1 seam token suffice")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(F, "fig_e2_faithfulness.pdf")); plt.close()
    print("wrote fig_e2_faithfulness.pdf")


def fig_e3_editing():
    recs = load("e3")
    if not recs:
        return
    models = sorted(set(r["model"] for r in recs))
    methods = ["stale", "in_place", "selective@4", "selective@16", "erratum", "recompile_chunk", "full_recompute"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
    ax = axes[0]
    x = np.arange(len(methods)); w = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        ys = [np.mean([r["correct"] for r in recs if r["model"] == m and r["method"] == meth] or [np.nan]) for meth in methods]
        ax.bar(x + i * w, ys, w, label=short(m))
    ax.set_xticks(x + w * (len(models) - 1) / 2); ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=6)
    ax.set_ylabel("decision correct (vs new gold)"); ax.set_title("(a) Editing a memory fact (CoT)")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=6)
    # (b) cost vs correctness frontier (one model)
    ax = axes[1]
    m = models[-1]
    for meth, col in zip(methods, [C["gray"], C["red"], C["orange"], C["purple"], C["green"], C["blue"], "#000"]):
        sub = [r for r in recs if r["model"] == m and r["method"] == meth]
        if sub:
            ax.scatter(np.median([r["recompute_tok"] for r in sub]), np.mean([r["correct"] for r in sub]),
                       c=col, s=40); ax.annotate(meth, (np.median([r["recompute_tok"] for r in sub]), np.mean([r["correct"] for r in sub])), fontsize=6)
    ax.set_xscale("symlog"); ax.set_xlabel("recompute tokens (median)"); ax.set_ylabel("decision correct")
    ax.set_title(f"(b) Cost/correctness frontier ({short(m)})"); ax.set_ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(os.path.join(F, "fig_e3_editing.pdf")); plt.close()
    print("wrote fig_e3_editing.pdf")


def fig_e1_placement():
    recs = load("e1")
    if not recs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    models = sorted(set(r["model"] for r in recs))
    for i, (ax, reg) in enumerate(zip(axes, ["direct", "cot"])):
        for m in models:
            for pl, ls in [("early", "-o"), ("late", "--s")]:
                nfs = sorted(set(r["n_facts"] for r in recs if r["model"] == m))
                ys = [np.mean([r["correct"] for r in recs if r["model"] == m and r["reasoning"] == reg and r["placement"] == pl and r["n_facts"] == nf] or [np.nan]) for nf in nfs]
                # label only once (panel a) so the shared legend has no duplicates
                ax.plot(nfs, ys, ls, ms=4, label=(f"{short(m)} {pl}" if i == 0 else None))
        ax.set_xlabel("integration depth (n_facts)"); ax.set_ylabel("decision accuracy")
        ax.set_title(f"({'a' if reg=='direct' else 'b'}) {reg}"); ax.set_ylim(0, 1.05)
    # one shared legend outside (below) both panels -- no overlap with the lines
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handlelength=1.7, handletextpad=0.5)
    fig.tight_layout(rect=[0, 0.24, 1, 1])
    plt.savefig(os.path.join(F, "fig_e1_placement.pdf")); plt.close()
    print("wrote fig_e1_placement.pdf")


def fig_e4_granularity():
    recs = load("e4")
    if not recs:
        return
    models = sorted(set(r["model"] for r in recs))
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    ax2 = ax.twinx()
    for m in models:
        Ss = sorted(set(r["S"] for r in recs if r["model"] == m))
        da = [np.mean([r["dec_agree"] for r in recs if r["model"] == m and r["S"] == s]) for s in Ss]
        cost = [np.median([r["edit_cost_tok"] for r in recs if r["model"] == m and r["S"] == s]) for s in Ss]
        ax.plot(Ss, da, "-o", ms=4, c=C["blue"], label=f"{short(m)} agree")
        ax2.plot(Ss, cost, "--s", ms=4, c=C["orange"], label=f"{short(m)} edit cost")
    ax.set_xlabel("sub-blocks $S$"); ax.set_ylabel("decision agreement vs full", color=C["blue"])
    ax2.set_ylabel("localized-edit tokens", color=C["orange"]); ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.05); ax.set_title("E4: granularity cost/fidelity")
    plt.tight_layout(); plt.savefig(os.path.join(F, "fig_e4_granularity.pdf")); plt.close()
    print("wrote fig_e4_granularity.pdf")


def fig_e5_systems():
    recs = load("e5")
    if not recs:
        return
    models = sorted(set(r["model"] for r in recs))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
    ax = axes[0]
    x = np.arange(len(models)); w = 0.2
    for j, (meth, col) in enumerate([("oracle", "#000"), ("front", C["gray"]), ("end", C["red"]), ("proposed", C["green"])]):
        ys = [np.median([r[f"ttft_{meth}"] for r in recs if r["model"] == m]) for m in models]
        ax.bar(x + (j - 1.5) * w, ys, w, color=col, label=meth)
    ax.set_xticks(x); ax.set_xticklabels([short(m) for m in models], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("median TTFT (ms)"); ax.set_title("(a) Per-decision TTFT"); ax.legend(fontsize=7)
    ax = axes[1]
    xb = np.arange(len(models)); wb = 0.36
    sf_all, se_all = [], []
    for m in models:
        mr = [r for r in recs if r["model"] == m]
        sess = defaultdict(lambda: defaultdict(float))
        for r in mr:
            for meth in ("front", "end", "proposed"):
                sess[r["session"]][meth] += r[f"ttft_{meth}"]
        sf_all.append(np.median([sess[s]["front"] / sess[s]["proposed"] for s in sess]))
        se_all.append(np.median([sess[s]["end"] / sess[s]["proposed"] for s in sess]))
    ax.bar(xb - wb / 2, sf_all, wb, color=C["gray"], label="vs front placement")
    ax.bar(xb + wb / 2, se_all, wb, color=C["red"], label="vs end placement")
    ax.axhline(1, ls="--", c="#000", lw=1)
    ax.set_xticks(xb); ax.set_xticklabels([short(m) for m in models], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("cumulative TTFT speedup (×)")
    ax.set_title("(b) Cumulative speedup"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(F, "fig_e5_systems.pdf")); plt.close()
    print("wrote fig_e5_systems.pdf")


def fig_locomo():
    recs = load("locomo")
    if not recs:
        return
    models = sorted(set(r["model"] for r in recs))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
    ax = axes[0]; x = np.arange(len(models)); w = 0.38
    af = [np.mean([r["correct_full"] for r in recs if r["model"] == m]) for m in models]
    at = [np.mean([r["correct_transplant"] for r in recs if r["model"] == m]) for m in models]
    ax.bar(x - w / 2, af, w, color=C["gray"], label="full recompute")
    ax.bar(x + w / 2, at, w, color=C["green"], label="transplant")
    ax.set_xticks(x); ax.set_xticklabels([short(m) for m in models], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("LoCoMo QA accuracy"); ax.set_title("(a) Real memory: accuracy parity"); ax.set_ylim(0, 1); ax.legend(fontsize=7)
    ax = axes[1]
    cos = [np.mean([r["ans_cos"] for r in recs if r["model"] == m]) for m in models]
    t1 = [np.mean([r["ans_top1_agree"] for r in recs if r["model"] == m]) for m in models]
    ax.bar(x - w / 2, cos, w, color=C["blue"], label="answer-token cos")
    ax.bar(x + w / 2, t1, w, color=C["orange"], label="top-1 agree")
    ax.set_xticks(x); ax.set_xticklabels([short(m) for m in models], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("transplant vs full"); ax.set_title("(b) Real memory: answer fidelity"); ax.set_ylim(0, 1.05); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(F, "fig_locomo.pdf")); plt.close()
    print("wrote fig_locomo.pdf")


if __name__ == "__main__":
    for fn in [fig_e2_seam, fig_e3_editing, fig_e1_placement, fig_e4_granularity, fig_e5_systems, fig_locomo]:
        try:
            fn()
        except Exception as e:
            print("skip", fn.__name__, repr(e)[:120])
    print("FIGS_DONE")
