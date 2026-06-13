"""Composable-axis figures: TTFT scaling + keystone (composed vs recomputed) across models."""
import json, os, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
F = os.path.join(os.path.dirname(__file__), "..", "figures")
NAMES = {"qwen3_8b": "Qwen3-8B", "dsr1_llama8b": "DeepSeek-R1-Llama-8B", "qwen3_1p7b": "Qwen3-1.7B", "llama31_8b": "Llama-3.1-8B", "qwen3_4b": "Qwen3-4B"}


def fig_scaling():
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for p in sorted(glob.glob(os.path.join(R, "composable_scaling_*.json"))):
        d = json.load(open(p)); sc = d["scaling"]
        tag = os.path.basename(p)[len("composable_scaling_"):-5]
        ks = sorted(int(k) for k in sc); ys = [sc[str(k)]["speedup"] for k in ks]
        ax.plot(ks, ys, "o-", label=NAMES.get(tag, tag))
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("skill length (tokens)"); ax.set_ylabel("TTFT speedup (full reprefill / precompiled)")
    ax.set_title("Composable KV: precompiled-skill TTFT speedup vs skill length\n(full prefill O(L²) → transplant O(L))")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_composable_scaling.png")); plt.close(fig)
    print("ok: fig_composable_scaling")


def fig_keystone():
    order = [("qwen3_1p7b", "1.7B"), ("qwen3_4b", "4B"), ("qwen3_8b", "8B"), ("qwen3_14b", "14B"),
             ("dsr1_llama8b", "DeepSeek-8B"), ("mistral7b", "Mistral-7B"), ("llama31_8b", "Llama-3.1-8B")]
    labels, rec_e, com_e, rec_s, com_s = [], [], [], [], []
    for tag, nm in order:
        p = os.path.join(R, f"compose_edit_{tag}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p)).get("agg", {})
        if not d:
            continue
        labels.append(nm)
        rec_e.append(d["erratum"]["recomputed"]); com_e.append(d["erratum"]["composed"])
        rec_s.append(d["sel@32"]["recomputed"]); com_s.append(d["sel@32"]["composed"])
    if not labels:
        print("no keystone data"); return
    import numpy as np
    x = np.arange(len(labels)); w = 0.2
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(x - 1.5 * w, rec_s, w, label="sel@32 recomputed", color="#9ecae1")
    ax.bar(x - 0.5 * w, com_s, w, label="sel@32 composed", color="#3182bd")
    ax.bar(x + 0.5 * w, rec_e, w, label="erratum recomputed", color="#fdae6b")
    ax.bar(x + 1.5 * w, com_e, w, label="erratum composed", color="#e6550d")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("edit recovery (D1-style ratio)")
    ax.set_title("Keystone: editing a field INSIDE a transplanted skill\ncomposed ≈ recomputed (edit+compose on one substrate)")
    ax.set_ylim(-1, 2.6); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, axis="y"); ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_keystone.png")); plt.close(fig)
    print("ok: fig_keystone")


fig_scaling(); fig_keystone()
print("COMPOSABLE_FIGS_DONE")
