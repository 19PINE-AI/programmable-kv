"""Circuit-level figure (fig8_circuit.pdf) from the circ_*.json results.
Panels: (a) read/write head cumulative recovery; (b) causal conclusion direction across layers;
(c) attn-vs-MLP write share by readout layer; (d) causal-scrubbing faithfulness vs interchange;
(e) SAE sparse-decode vs distributed-cause; (f) read-head decision->aggregator attention.
Run: python paper/figs/make_circuit_figure.py   (cwd = repo patchkv/)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "..", "results")
OUT = os.path.dirname(__file__)
def J(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 4, "axes.labelsize": 8.5, "legend.fontsize": 7.0,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8, "axes.edgecolor": "#3a3a3a", "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
    "lines.linewidth": 1.9, "lines.markersize": 4.5, "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.16, "grid.linewidth": 0.6, "grid.color": "#8a8a8a",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.axisbelow": True, "figure.facecolor": "white", "savefig.facecolor": "white",
})
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#9a9a9a"}


FAM = [("llama31_8b", "Llama-3.1-8B", C["blue"]), ("qwen3_8b", "Qwen3-8B", C["orange"]),
       ("gemma2_9b", "Gemma-2-9B", C["green"]), ("mistral_7b", "Mistral-7B", C["purple"])]


def panel_heads(ax):
    for tag, name, c in FAM:
        d = J(f"circ_heads_{tag}.json")
        if not d:
            continue
        s = d["summary"]
        rk = sorted(int(k) for k in s["read_cumk"])
        ax.plot(rk, [s["read_cumk"][str(k)]["mean"] for k in rk], "o-", color=c, label=name)
        wk = sorted(int(k) for k in s["write_cumk"])
        ax.plot(wk, [s["write_cumk"][str(k)]["mean"] for k in wk], "s:", color=c, alpha=0.55, markersize=3)
    ax.plot([], [], "ko-", label="read (solid)"); ax.plot([], [], "ks:", alpha=0.6, label="write (dotted)")
    ax.set_xlabel("# named heads patched jointly (top-$k$)")
    ax.set_ylabel("decision recovery")
    ax.set_title("(a) read & write head circuit")
    ax.set_ylim(-0.05, 0.9); ax.legend(loc="upper left", fontsize=5.6, ncol=2)


def panel_direction(ax):
    d = J("circ_direction_llama31_8b.json")
    if not d:
        ax.set_visible(False); return
    pl = d["summary"]["per_layer"]
    layers = sorted(int(k) for k in pl)
    full = [pl[str(L)]["dm"]["full"]["mean"] for L in layers]
    along = [pl[str(L)]["dm"]["along"]["mean"] for L in layers]
    rand = [pl[str(L)]["dm"]["random"]["mean"] for L in layers]
    ax.plot(layers, full, "o-", color=C["grey"], label="full residual (ceiling)")
    ax.plot(layers, along, "o-", color=C["blue"], label="along $\\hat d$ (1-D conclusion dir.)")
    ax.plot(layers, rand, "o-", color=C["red"], label="random 1-D")
    ax.set_xlabel("layer"); ax.set_ylabel("single-site recovery")
    ax.set_title("(b) causal conclusion direction"); ax.set_ylim(-0.05, 0.85)
    ax.legend(loc="upper right", fontsize=6.4)


def panel_components(ax):
    for tag, name, c in FAM:
        Ts, attn = [], []
        for T in [12, 14, 16, 18, 20, 22, 28]:
            d = J(f"circ_components_{tag}_T{T}.json")
            if not d:
                continue
            Ts.append(T); attn.append(d["summary"]["attn_share"]["mean"])
        if Ts:
            ax.plot(Ts, attn, "o-", color=c, label=name)
    ax.axhline(0.5, color="#bbbbbb", ls=":", lw=1)
    ax.set_xlabel("readout layer $T$"); ax.set_ylabel("attention share of write")
    ax.set_title("(c) attention writes the note"); ax.set_ylim(0.3, 0.95)
    ax.legend(loc="lower left", fontsize=6.0)


def panel_scrub(ax):
    labels = ["faithful drift\n(same concl.)", "swap note\n(opposite)", "swap rest\n(opposite)"]
    x = np.arange(len(labels)); nF = len(FAM); w = 0.8 / nF
    for i, (tag, name, c) in enumerate(FAM):
        d = J(f"circ_scrub_{tag}.json")
        if not d:
            continue
        s = d["summary"]
        vals = [s["faithfulness_drift"]["drift_all_same"]["mean"],
                s["interchange_recovery"]["rec_note_opp"]["mean"],
                s["interchange_recovery"]["rec_rest_opp"]["mean"]]
        ax.bar(x + (i - (nF - 1) / 2) * w, vals, w, color=c, label=name)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.0)
    ax.set_ylabel("drift (faithful≈0) / recovery"); ax.set_title("(d) causal scrubbing (4 families)")
    ax.set_ylim(0, 1.0); ax.legend(loc="upper center", fontsize=5.6, ncol=2)


def panel_sae(ax):
    d = J("circ_sae_llama31_8b_L14.json")
    if not d:
        ax.set_visible(False); return
    s = d["summary"]
    Ks = sorted(int(k) for k in s["sufficiency_recovery_byK"])
    suff = [s["sufficiency_recovery_byK"][str(k)]["mean"] for k in Ks]
    ctrl = [s["sufficiency_control_byK"][str(k)]["mean"] for k in Ks]
    ax.plot(Ks, suff, "o-", color=C["green"], label="top-$k$ conclusion feats")
    ax.plot(Ks, ctrl, "o--", color=C["red"], label="random-$k$ feats", alpha=0.7)
    ax.set_xscale("log"); ax.set_xticks(Ks); ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("# SAE features clamped"); ax.set_ylabel("decision recovery")
    ax.set_title(f"(e) SAE: decode$\\neq$cause (best AUC={s['best_single_auc']:.2f})")
    ax.set_ylim(-0.1, 0.7); ax.legend(loc="upper left", fontsize=6.6)


def panel_readattn(ax):
    """Top read heads' decision->aggregator attention vs causal recovery (Llama)."""
    d = J("circ_heads_llama31_8b.json")
    if not d:
        ax.set_visible(False); return
    rh = d["summary"]["read_heads_ranked"][:10]
    rec = [h["rec_mean"] for h in rh]; att = [h["attn_mean"] for h in rh]
    sc = ax.scatter(att, rec, c=C["purple"], s=22, zorder=3)
    for h in rh[:5]:
        ax.annotate(h["head"], (h["attn_mean"], h["rec_mean"]), fontsize=5.6,
                    xytext=(3, 2), textcoords="offset points", color="#444")
    ax.set_xlabel("decision$\\to$aggregator attention"); ax.set_ylabel("causal recovery")
    ax.set_title("(f) read heads attend the note")


def main():
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6))
    panel_heads(axes[0, 0]); panel_direction(axes[0, 1]); panel_components(axes[0, 2])
    panel_scrub(axes[1, 0]); panel_sae(axes[1, 1]); panel_readattn(axes[1, 2])
    fig.tight_layout(pad=0.6)
    p = os.path.join(OUT, "fig8_circuit.pdf")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
