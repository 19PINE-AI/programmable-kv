"""Generate figures for the memory-KV experiments from results/*.jsonl (Times-coherent style)."""
import os, sys, json, glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})
R = os.path.join(os.path.dirname(__file__), "results")
F = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(F, exist_ok=True)
C = dict(blue="#2c6fbb", orange="#e07b39", green="#3a9d6e", red="#c0392b", gray="#7f8c8d", purple="#8e44ad")


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
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
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
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
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
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    models = sorted(set(r["model"] for r in recs))
    for ax, reg in zip(axes, ["direct", "cot"]):
        for m in models:
            for pl, ls in [("early", "-o"), ("late", "--s")]:
                nfs = sorted(set(r["n_facts"] for r in recs if r["model"] == m))
                ys = [np.mean([r["correct"] for r in recs if r["model"] == m and r["reasoning"] == reg and r["placement"] == pl and r["n_facts"] == nf] or [np.nan]) for nf in nfs]
                ax.plot(nfs, ys, ls, ms=4, label=f"{short(m)} {pl}")
        ax.set_xlabel("integration depth (n_facts)"); ax.set_ylabel("decision accuracy")
        ax.set_title(f"({'a' if reg=='direct' else 'b'}) {reg}"); ax.set_ylim(0, 1.05); ax.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(F, "fig_e1_placement.pdf")); plt.close()
    print("wrote fig_e1_placement.pdf")


def fig_e4_granularity():
    recs = load("e4")
    if not recs:
        return
    models = sorted(set(r["model"] for r in recs))
    fig, ax = plt.subplots(figsize=(5, 3.4))
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
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    x = np.arange(len(models)); w = 0.2
    for j, (meth, col) in enumerate([("oracle", "#000"), ("front", C["gray"]), ("end", C["red"]), ("proposed", C["green"])]):
        ys = [np.median([r[f"ttft_{meth}"] for r in recs if r["model"] == m]) for m in models]
        ax.bar(x + (j - 1.5) * w, ys, w, color=col, label=meth)
    ax.set_xticks(x); ax.set_xticklabels([short(m) for m in models], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("median TTFT (ms)"); ax.set_title("(a) Per-decision TTFT"); ax.legend(fontsize=7)
    ax = axes[1]
    for m in models:
        mr = [r for r in recs if r["model"] == m]
        sess = defaultdict(lambda: defaultdict(float))
        for r in mr:
            for meth in ("front", "end", "proposed"):
                sess[r["session"]][meth] += r[f"ttft_{meth}"]
        sf = np.median([sess[s]["front"] / sess[s]["proposed"] for s in sess])
        se = np.median([sess[s]["end"] / sess[s]["proposed"] for s in sess])
        ax.bar(short(m) + "\nvs front", sf, color=C["gray"]); ax.bar(short(m) + "\nvs end", se, color=C["red"])
    ax.axhline(1, ls="--", c="#000", lw=1); ax.set_ylabel("cumulative TTFT speedup (×)")
    ax.set_title("(b) Cumulative speedup"); plt.setp(ax.get_xticklabels(), fontsize=6, rotation=20)
    plt.tight_layout(); plt.savefig(os.path.join(F, "fig_e5_systems.pdf")); plt.close()
    print("wrote fig_e5_systems.pdf")


def fig_locomo():
    recs = load("locomo")
    if not recs:
        return
    models = sorted(set(r["model"] for r in recs))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
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
