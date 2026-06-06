"""Generate publication figures from the result JSONs. Outputs to ../figures/*.png.
Run: python esys/make_figures.py
"""
import json, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
F = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(F, exist_ok=True)
def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def fig_memoization_map():
    d = load("mech_causal_patch_qwen3_8b.json")
    if not d:
        return
    agg = d["agg"]
    suf = sorted(((float(k), v["mean"], v["ci"]) for k, v in agg["cum_suffix_mean"].items()))
    pre = sorted(((float(k), v["mean"], v["ci"]) for k, v in agg["cum_prefix_mean"].items()))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    xs = [x for x, _, _ in suf]
    ax.plot(xs, [m for _, m, _ in suf], "o-", color="#1f77b4", label="suffix (last k% of downstream)")
    ax.fill_between(xs, [c[0] for _, _, c in suf], [c[1] for _, _, c in suf], color="#1f77b4", alpha=0.15)
    ax.plot([x for x, _, _ in pre], [m for _, m, _ in pre], "s--", color="#ff7f0e", label="prefix (first k%)")
    fo = agg["field_only_recovery"]["mean"]
    ax.axhline(fo, color="crimson", ls=":", lw=1.5)
    ax.text(0.5, fo + 0.03, f"field-only (in_place) = {fo:.3f}", color="crimson", fontsize=9)
    ax.set_xlabel("fraction of downstream KV recomputed (patched to new)")
    ax.set_ylabel("decision-flip recovery")
    ax.set_title("Causal memoization map (Qwen3-8B, n=12)\nthe field's effect is suffix-concentrated")
    ax.legend(loc="center right", fontsize=8); ax.set_ylim(-0.05, 1.05)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_memoization_map.png")); plt.close(fig)


def fig_d1_generalization():
    order = [("qwen3_4b", "Qwen3-4B"), ("qwen3_8b", "Qwen3-8B"), ("qwen3_14b", "Qwen3-14B"),
             ("qwen3_32b", "Qwen3-32B"), ("gemma2_9b", "Gemma2-9B"), ("gemma2_27b", "Gemma2-27B"),
             ("gemma3_27b_bf16", "Gemma3-27B"), ("mistral7b", "Mistral-7B")]
    labels, fo, full = [], [], []
    for tag, name in order:
        d = load(f"mech_causal_patch_{tag}.json")
        if not d:
            continue
        labels.append(name); fo.append(d["agg"]["field_only_recovery"]["mean"])
        full.append(d["agg"]["full_downstream_recovery"]["mean"])
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], fo, 0.4, label="field-only (in_place)", color="crimson")
    ax.bar([i + 0.2 for i in x], full, 0.4, label="full-downstream (=full reprefill)", color="#1f77b4")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("decision-flip recovery"); ax.set_ylim(0, 1.1)
    ax.set_title("in_place is causally inert across 7 models (full-downstream = 1.0)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_d1_generalization.png")); plt.close(fig)


def fig_dose_response():
    d = load("mech_dose_response_qwen3_8b.json")
    if not d:
        return
    items = sorted(d["by_pos"].items(), key=lambda kv: int(kv[0]))
    xs = [r["label"] for _, r in items]; ys = [r["field_only_mean"] for _, r in items]
    ci = [r.get("field_only_ci", [y, y]) for (_, r), y in zip(items, ys)]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.errorbar(range(len(xs)), ys, yerr=[[y - c[0] for y, c in zip(ys, ci)], [c[1] - y for y, c in zip(ys, ci)]],
                fmt="o-", color="#2ca02c", capsize=3)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([x.split("(")[0] for x in xs], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("in_place recovery"); ax.set_xlabel("field position (less conditioned text after it →)")
    ax.set_title("Dose-response: in_place recovers more as the field moves later (D6)")
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_dose_response.png")); plt.close(fig)


def fig_surgical():
    order = [("Qwen3-8B", "8B"), ("Qwen3-14B", "14B"), ("Qwen3-32B", "32B")]
    nr, rr, rci, labels = [], [], [], []
    for tag, name in order:
        d = load(f"surgical_suffices_{tag}.json")
        if not d:
            continue
        labels.append(name); nr.append(d["non_reasoning"]["in_place_correct"])
        rr.append(d["reasoning"]["in_place_correct"])
        rci.append(d["reasoning"].get("in_place_boot_ci") or d["reasoning"].get("in_place_ci") or [rr[-1], rr[-1]])
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], nr, 0.4, label="non-reasoning", color="#7f7f7f")
    yerr = [[r - c[0] for r, c in zip(rr, rci)], [c[1] - r for r, c in zip(rr, rci)]]
    ax.bar([i + 0.2 for i in x], rr, 0.4, yerr=yerr, capsize=3, label="reasoning", color="#1f77b4")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels); ax.set_ylim(0, 1.1)
    ax.set_ylabel("P(surgical in_place == oracle), no erratum")
    ax.set_title("When the surgical edit alone suffices (§5e)\nreasoning-only, scale-dependent")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_surgical_suffices.png")); plt.close(fig)


def fig_architecture():
    rows = [("arch_erratum_v2_Qwen3-8B.json", "attention\n(Qwen3-8B)"),
            ("arch_erratum_v2_Falcon-H1-1_5B-Instruct.json", "hybrid\n(Falcon-H1)"),
            ("arch_erratum_v2_falcon-mamba-7b-instruct.json", "pure SSM\n(Falcon-Mamba)")]
    labels, nr, rr, nrci, rrci = [], [], [], [], []
    for fn, name in rows:
        d = load(fn)
        if not d:
            continue
        labels.append(name)
        nr.append(d["non_reasoning"]["erratum_recovery"]); rr.append(d["reasoning"]["erratum_recovery"])
        nrci.append(d["non_reasoning"]["erratum_recovery_ci"]); rrci.append(d["reasoning"]["erratum_recovery_ci"])
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    x = range(len(labels))
    def yerr(vals, cis):
        return [[(v - c[0]) if (v is not None and c) else 0 for v, c in zip(vals, cis)],
                [(c[1] - v) if (v is not None and c) else 0 for v, c in zip(vals, cis)]]
    nrp = [v if v is not None else 0 for v in nr]
    ax.bar([i - 0.2 for i in x], nrp, 0.4, yerr=yerr(nr, nrci), capsize=3, label="non-reasoning", color="#7f7f7f")
    ax.bar([i + 0.2 for i in x], rr, 0.4, yerr=yerr(rr, rrci), capsize=3, label="reasoning (CoT)", color="#1f77b4")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels); ax.set_ylim(0, 1.15)
    ax.set_ylabel("erratum recovery (P | oracle flips)")
    ax.set_title("editkv is an attention-architecture method (§6b)\nCoT partially rescues pure SSM")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_architecture.png")); plt.close(fig)


def fig_serving():
    d = load("serving_bench_qwen3_8b.json") or load("serving_bench_qwen3_8b_32k.json")
    if not d:
        return
    rows = [r for r in d["rows"] if "skipped" not in r]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for bs in sorted(set(r["bs"] for r in rows)):
        rs = sorted([r for r in rows if r["bs"] == bs], key=lambda r: r["T"])
        ax.plot([r["T"] for r in rs], [r["speedup_erratum"] for r in rs], "o-", label=f"erratum, bs={bs}")
    ax.set_xscale("log"); ax.set_xlabel("context length (tokens)"); ax.set_ylabel("TTFT speedup vs full reprefill")
    ax.set_title("Serving: erratum TTFT speedup grows with context & batch (§8b)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_serving.png")); plt.close(fig)


def fig_baseline_frontier():
    d = load("baseline_table_qwen3_8b.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for m, v in d["methods"].items():
        x = 100 * v["recompute_frac"]; y = v["P_correct"]
        ax.scatter(x, y, s=60)
        ax.annotate(m, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("recompute (% of tokens)"); ax.set_ylabel("P(correct decision)")
    ax.set_title("Correctness vs cost (n=8, NON-reasoning)\nhoist & field+erratum reach 1.0 cheaply; in_place needs reasoning (§5e)")
    ax.set_xscale("symlog", linthresh=1); ax.set_ylim(-0.05, 1.08)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_baseline_frontier.png")); plt.close(fig)


def fig_online_load():
    d = load("vllm_online_load_qwen3_8b.json")
    if not d:
        return
    rows = sorted(d["rows"], key=lambda r: r["N"])
    Ns = [r["N"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(Ns, [r["baseline"]["throughput_req_s"] for r in rows], "s--", color="#d62728",
            label="baseline (new field in prefix)")
    ax.plot(Ns, [r["erratum"]["throughput_req_s"] for r in rows], "o-", color="#1f77b4",
            label="erratum (append-only, prefix reused)")
    ax.set_xscale("log", base=2); ax.set_xlabel("offered concurrency (requests)")
    ax.set_ylabel("throughput (req/s)")
    ax.set_title("Online load on vLLM: baseline saturates (compute-bound),\nerratum scales (cache-bound)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_online_load.png")); plt.close(fig)


for fn in [fig_memoization_map, fig_d1_generalization, fig_dose_response, fig_surgical,
           fig_architecture, fig_serving, fig_baseline_frontier, fig_online_load]:
    try:
        fn(); print("ok:", fn.__name__)
    except Exception as e:
        print("FAIL:", fn.__name__, type(e).__name__, e)
print("FIGURES_DONE; wrote to", F)
