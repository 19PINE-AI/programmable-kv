#!/usr/bin/env python3
"""Extract curated, site-ready JSON from the PatchKV result records.

Fidelity rule: every number the site shows comes from a released result record
(results/*.json, mem/results/*), from the deterministic prompt builders
(e1/contexts.py, e2/scenarios.py), or — where a quantity exists only in the
paper text — from constants.json entries that carry an explicit `source` label.

Run:  python3 site/data/build_data.py   (from the repo root or anywhere)
Writes: site/src/data/*.json and prints an assertion table vs paper claims.
"""

import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATCHKV = os.path.dirname(os.path.dirname(HERE))  # repo root
R = os.path.join(PATCHKV, "results")
M = os.path.join(PATCHKV, "mem", "results")
OUT = os.path.join(os.path.dirname(HERE), "src", "data")

sys.path.insert(0, os.path.join(PATCHKV, "e1"))
sys.path.insert(0, os.path.join(PATCHKV, "e2"))
import contexts  # noqa: E402  (pure python, deterministic)
import scenarios  # noqa: E402

os.makedirs(OUT, exist_ok=True)

LABELS = {
    "qwen3_0p6b": "Qwen3-0.6B", "qwen3_1p7b": "Qwen3-1.7B", "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B", "qwen3_14b": "Qwen3-14B", "qwen3_32b": "Qwen3-32B-FP8",
    "qwen3_32b_fp8": "Qwen3-32B-FP8", "qwen3_30a3b": "Qwen3-30B-A3B",
    "llama31_8b": "Llama-3.1-8B", "llama31_70b_4bit": "Llama-3.1-70B (4-bit)",
    "mistral7b": "Mistral-7B", "mistral_7b": "Mistral-7B",
    "gemma2_2b": "Gemma-2-2B", "gemma2_9b": "Gemma-2-9B", "gemma2_27b": "Gemma-2-27B",
    "gemma2_27b_int8": "Gemma-2-27B (int8)", "gemma3_4b": "Gemma-3-4B",
    "gemma3_27b": "Gemma-3-27B", "gemma3_27b_bf16": "Gemma-3-27B",
    "gemma3_27b_int8": "Gemma-3-27B (int8)",
    "dsr1_llama8b": "R1-Distill-Llama-8B", "dsr1llama8b": "R1-Distill-Llama-8B",
    "smollm2_1p7b": "SmolLM2-1.7B",
    "qwen25vl_3b": "Qwen2.5-VL-3B", "qwen25vl_7b": "Qwen2.5-VL-7B",
    "qwen25vl_32b": "Qwen2.5-VL-32B", "qwen2vl_7b": "Qwen2-VL-7B",
    "qwen3vl_8b": "Qwen3-VL-8B", "qwen3vl_30a3b": "Qwen3-VL-30B-A3B",
    "dscoderv2_mla": "DeepSeek-Coder-V2-Lite", "dsv2lite_mla": "DeepSeek-V2-Lite-Chat",
    "Qwen3-8B": "Qwen3-8B", "Falcon-H1-1_5B-Instruct": "Falcon-H1-1.5B",
    "falcon-mamba-7b-instruct": "Falcon-Mamba-7B",
}

SIZE_ORDER = [
    "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-30B-A3B",
    "Qwen3-32B-FP8", "Llama-3.1-8B", "Llama-3.1-70B (4-bit)", "R1-Distill-Llama-8B",
    "Mistral-7B", "Gemma-2-2B", "Gemma-2-9B", "Gemma-2-27B", "Gemma-3-4B", "Gemma-3-27B",
]


def label_of(tag_or_id: str) -> str:
    if tag_or_id in LABELS:
        return LABELS[tag_or_id]
    tail = tag_or_id.split("/")[-1]
    tail = (tail.replace("Meta-Llama-3.1-", "Llama-3.1-")
                .replace("-Instruct-bnb-4bit", " (4-bit)")
                .replace("-Instruct", "").replace("-it", ""))
    return LABELS.get(tail, tail)


def sort_key(label: str):
    try:
        return (SIZE_ORDER.index(label), label)
    except ValueError:
        return (len(SIZE_ORDER), label)


def load(path):
    with open(path) as f:
        return json.load(f)


def num_items(d):
    """dict with numeric-string keys -> list of (float_key, value), sorted."""
    return sorted(((float(k), v) for k, v in d.items()), key=lambda kv: kv[0])


def _sanitize(o):
    """NaN/Infinity are not valid JSON — replace with null so TS/Vite can parse."""
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sanitize(v) for v in o]
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o


def write(name, obj):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(_sanitize(obj), f, indent=1)
    print(f"  wrote {name:24s} {os.path.getsize(path)/1024:8.1f} kB")


CHECKS = []


def check(claim, actual, expect, tol=0.02):
    ok = actual is not None and abs(actual - expect) <= tol
    CHECKS.append((claim, actual, expect, ok))


# =============================================================================
# 1. mechanism.json — four probes
# =============================================================================

def build_mechanism():
    models = []
    for path in sorted(glob.glob(os.path.join(R, "mech_causal_patch_*.json"))):
        tag = os.path.basename(path)[len("mech_causal_patch_"):-len(".json")]
        agg = load(path)["agg"]
        models.append({
            "tag": tag, "label": label_of(tag), "n": agg.get("n_instances"),
            "field_only": agg["field_only_recovery"],
            "full_downstream": agg["full_downstream_recovery"],
            "locality_topk": [
                {"k": int(k), "mean": v["mean"], "ci": v.get("ci")}
                for k, v in num_items(agg.get("locality_topk_mean", {}))
            ],
            "cum_suffix": [
                {"frac": k, "mean": v["mean"], "ci": v.get("ci")}
                for k, v in num_items(agg.get("cum_suffix_mean", {}))
            ],
            "cum_prefix": [
                {"frac": k, "mean": v["mean"], "ci": v.get("ci")}
                for k, v in num_items(agg.get("cum_prefix_mean", {}))
            ],
        })
    models.sort(key=lambda m: sort_key(m["label"]))

    dr = load(os.path.join(R, "mech_dose_response_qwen3_8b.json"))
    dose = [
        {"pos": int(k), "label": v["label"], "mean": v["field_only_mean"],
         "ci": v.get("field_only_ci"), "suffix80": v.get("suffix_frac_for_80pct_mean")}
        for k, v in num_items(dr["by_pos"])
    ]

    we = load(os.path.join(R, "why_erratum_qwen3_8b.json"))
    # NB: in esys/why_erratum.py ALL variants carry the NEW value in context
    # (clean reprefill); the appended text is pure wording on top of it.
    WORDING_LABELS = {
        "none": "clean context, nothing appended", "value_only": "+ bare value update",
        "update_tag": "+ [STATE UPDATE] tag", "override_full": "+ explicit override",
        "conclusion": '+ "conclusion void; re-evaluate"',
    }
    wording = [
        {"key": k, "label": WORDING_LABELS.get(k, k), "P_safe": v["P_safe"], "ci": v.get("ci")}
        for k, v in we["variants"].items()
    ]

    rb = load(os.path.join(R, "erratum_robustness_qwen3_8b.json"))
    robustness = {
        "phrasing": [
            {"key": k, "P_correct": v["P_correct"], "ci": v.get("ci")}
            for k, v in rb.get("phrasing", {}).items()
        ],
        "over_correction_flips": rb.get("over_correction_flips"),
        "multi_edit": rb.get("multi_edit"),
    }

    # behavioral P_correct over 8 diverse tasks (reasoning / non-reasoning)
    diverse = []
    for path in sorted(glob.glob(os.path.join(R, "mech_diverse_*.json"))):
        tag = os.path.basename(path)[len("mech_diverse_"):-len(".json")]
        d = load(path)
        by_mode = d.get("by_mode", {})
        entry = {"tag": tag, "label": label_of(tag), "tasks": d.get("tasks"), "modes": {}}
        for mode, body in by_mode.items():
            s = body.get("summary", {})
            entry["modes"][mode] = {
                cond: {"P_correct": sv.get("P_correct"), "ci": sv.get("ci"), "n": sv.get("n")}
                for cond, sv in s.items()
            }
        diverse.append(entry)
    diverse.sort(key=lambda m: sort_key(m["label"]))

    q8 = next(m for m in models if m["tag"] == "qwen3_8b")
    l8 = next(m for m in models if m["tag"] == "llama31_8b")
    check("mech field-only recovery (Qwen3-8B)", q8["field_only"]["mean"], 0.009, 0.01)
    check("mech field-only recovery (Llama-3.1-8B)", l8["field_only"]["mean"], -0.028, 0.01)
    check("mech full-downstream (Qwen3-8B)", q8["full_downstream"]["mean"], 1.0, 0.01)

    write("mechanism.json", {
        "models": models, "dose": dose, "wording": wording,
        "wording_n": we.get("n"), "robustness": robustness, "diverse": diverse,
    })


# =============================================================================
# 2. controls.json — deep mechanism controls
# =============================================================================

def _curve(d):
    return [
        {"layer": int(k), "concl_acc": v.get("concl_acc"), "field_acc": v.get("field_acc"),
         "depth": v.get("depth")}
        for k, v in num_items(d)
    ]


def build_controls():
    out = {"xcond": [], "specificity": [], "inject": [], "timing": [], "general": [],
           "replicate": []}

    for path in sorted(glob.glob(os.path.join(R, "mechd_xcond_*.json"))):
        tag = os.path.basename(path)[len("mechd_xcond_"):-len(".json")]
        d = load(path)
        p = d["patch"]
        out["xcond"].append({
            "tag": tag, "label": label_of(tag), "n": p.get("n"),
            "trigger_only": p["trigger_only_recovery"],
            "notes": p["notes_recovery"],
            "full_downstream": p.get("full_downstream_recovery"),
            "probe": _curve(d.get("probe", {})),
        })

    for path in sorted(glob.glob(os.path.join(R, "mechd_specificity_*.json"))):
        tag = os.path.basename(path)[len("mechd_specificity_"):-len(".json")]
        agg = load(path)["agg"]
        ks = sorted(int(k) for k in agg["topk"])
        out["specificity"].append({
            "tag": tag, "label": label_of(tag),
            "ks": [
                {"k": k,
                 "top": agg["topk"][str(k)],
                 "rand": agg["randk"].get(str(k))}
                for k in ks
            ],
        })

    for path in sorted(glob.glob(os.path.join(R, "mechd_inject_*.json"))):
        tag = os.path.basename(path)[len("mechd_inject_"):-len(".json")]
        agg = load(path)["agg"]
        out["inject"].append({
            "tag": tag, "label": label_of(tag),
            "full_recovery": agg["full_recovery"],
            "flip_rate": agg.get("flip_rate"),
            "follows_rate": agg.get("follows_injected_rate"),
            "dose": [
                {"k": int(k), "recovery": v.get("recovery_mean"), "ci": v.get("ci"),
                 "follow_rate": v.get("follow_rate")}
                for k, v in num_items(agg.get("dose", {}))
            ],
        })

    for path in sorted(glob.glob(os.path.join(R, "mechd_timing_*.json"))):
        tag = os.path.basename(path)[len("mechd_timing_"):-len(".json")]
        d = load(path)
        out["timing"].append({
            "tag": tag, "label": label_of(tag), "nlayers": d.get("nlayers"),
            "curves": {site: _curve(c) for site, c in d.get("curves", {}).items()},
        })

    for path in sorted(glob.glob(os.path.join(R, "mechd_general_*.json"))):
        tag = os.path.basename(path)[len("mechd_general_"):-len(".json")]
        agg = load(path)["agg"]
        fams = {}
        for fam in ("multihop", "natural", "rag_lookup"):
            if fam in agg:
                fams[fam] = {
                    "field_only": agg[fam].get("field_only_recovery"),
                    "full_downstream": agg[fam].get("full_downstream_recovery"),
                }
        out["general"].append({"tag": tag, "label": label_of(tag), "families": fams})

    for path in sorted(glob.glob(os.path.join(R, "mechd_replicate_*.json"))):
        tag = os.path.basename(path)[len("mechd_replicate_"):-len(".json")]
        d = load(path)
        timing = d.get("timing", {})
        out["replicate"].append({
            "tag": tag, "label": label_of(tag),
            "primary": d.get("primary"),
            "dissoc": d.get("dissoc"),
            "specificity": d.get("specificity"),
            "injection": d.get("injection"),
            "timing": {
                "write_layer": timing.get("write_layer"),
                "write_depth": timing.get("write_depth"),
                "commit_layer": timing.get("commit_layer_mean"),
                "commit_depth": timing.get("commit_depth_mean"),
                "nlayers": timing.get("nlayers"),
                "concl_acc_by_layer": [
                    {"layer": int(k), "concl_acc": v if not isinstance(v, dict) else v.get("concl_acc")}
                    for k, v in num_items(timing.get("concl_acc_by_layer", {}))
                ],
            },
        })

    for lst in out.values():
        lst.sort(key=lambda m: sort_key(m["label"]))

    rep = {r["tag"]: r for r in out["replicate"]}
    if "gemma2_9b" in rep:
        check("replication write depth (Gemma-2-9B)", rep["gemma2_9b"]["timing"]["write_depth"], 0.26, 0.03)
    sp = {s["tag"]: s for s in out["specificity"]}
    if "qwen3_8b" in sp:
        top8 = next(x for x in sp["qwen3_8b"]["ks"] if x["k"] == 8)
        check("specificity top-8 (Qwen3-8B)", top8["top"]["mean"], 0.78, 0.05)

    write("controls.json", out)


# =============================================================================
# 3. circuit.json
# =============================================================================

COMPONENT_T = {"qwen3_8b": "T18", "llama31_8b": "T14", "gemma2_9b": "T22", "mistral_7b": "T16"}


def build_circuit():
    models = []
    for path in sorted(glob.glob(os.path.join(R, "circ_heads_*.json"))):
        tag = os.path.basename(path)[len("circ_heads_"):-len(".json")]
        s = load(path)["summary"]

        def heads(key):
            return [
                {"head": h["head"], "rec": h.get("rec_mean"), "ci": h.get("rec_ci"),
                 "attr": h.get("attr_mean"), "attn": h.get("attn_mean"), "n": h.get("n")}
                for h in s.get(key, [])
            ]

        def cumk(key):
            return [
                {"k": int(k), "mean": v["mean"] if isinstance(v, dict) else v,
                 "ci": v.get("ci") if isinstance(v, dict) else None}
                for k, v in num_items(s.get(key, {}))
            ]

        entry = {
            "tag": tag, "label": label_of(tag), "n": s.get("n_instances"),
            "read_heads": heads("read_heads_ranked"),
            "write_heads": heads("write_heads_ranked"),
            "read_cumk": cumk("read_cumk"), "write_cumk": cumk("write_cumk"),
            "read_ctrl": s.get("read_ctrl_recovery"), "write_ctrl": s.get("write_ctrl_recovery"),
        }

        tfile = os.path.join(R, f"circ_components_{tag}_{COMPONENT_T.get(tag, '')}.json")
        if os.path.exists(tfile):
            cs = load(tfile)["summary"]
            entry["components"] = {
                "readout_layer": cs.get("readout_layer"),
                "attn_share": cs.get("attn_share"), "mlp_share": cs.get("mlp_share"),
                "layer_band_share": cs.get("layer_band_share"),
                "attn_per_layer": cs.get("attn_per_layer"),
                "mlp_per_layer": cs.get("mlp_per_layer"),
                "write_depth_p50": cs.get("write_depth_p50"),
                "write_depth_p90": cs.get("write_depth_p90"),
            }

        dfile = os.path.join(R, f"circ_direction_{tag}.json")
        if os.path.exists(dfile):
            ds = load(dfile)["summary"]
            entry["direction"] = {
                "layers": ds.get("layers"),
                "per_layer": {k: v for k, v in ds.get("per_layer", {}).items()},
                "n": ds.get("n_instances"),
            }

        sfile = os.path.join(R, f"circ_scrub_{tag}.json")
        if os.path.exists(sfile):
            ss = load(sfile)["summary"]
            entry["scrub"] = {
                "drift": ss.get("faithfulness_drift"),
                "interchange": ss.get("interchange_recovery"),
                "k_note": ss.get("k_note"),
                "cos_all_same": ss.get("cos_all_same"),
            }
        models.append(entry)
    models.sort(key=lambda m: sort_key(m["label"]))

    sae = {}
    for layer in ("L14", "L24"):
        p = os.path.join(R, f"circ_sae_llama31_8b_{layer}.json")
        if os.path.exists(p):
            s = load(p)["summary"]
            sae[layer] = {
                "best_single_auc": s.get("best_single_auc"),
                "top_features_auc": s.get("top_features_auc"),
                "n_active_features": s.get("n_active_features"),
                "sufficiency_byK": [
                    {"K": int(k), "mean": v.get("mean"), "n": v.get("n")}
                    for k, v in num_items(s.get("sufficiency_recovery_byK", {}))
                ],
                "control_byK": [
                    {"K": int(k), "mean": v.get("mean") if isinstance(v, dict) else v}
                    for k, v in num_items(s.get("sufficiency_control_byK", {}))
                ],
                "necessity_byK": [
                    {"K": int(k), "mean": v.get("mean") if isinstance(v, dict) else v}
                    for k, v in num_items(s.get("necessity_drop_byK", {}))
                ],
            }

    by_tag = {m["tag"]: m for m in models}
    if "llama31_8b" in by_tag:
        r12 = next((x for x in by_tag["llama31_8b"]["read_cumk"] if x["k"] == 12), None)
        check("circuit read top-12 recovery (Llama-3.1-8B)", r12 and r12["mean"], 0.78, 0.03)

    write("circuit.json", {"models": models, "sae": sae})


# =============================================================================
# 4. editing.json
# =============================================================================

def build_editing():
    bt = load(os.path.join(R, "baseline_table_qwen3_8b.json"))
    baseline = {
        "model": label_of(bt.get("model", "qwen3_8b")), "n_tasks": bt.get("n_tasks"),
        "methods": [
            {"method": k, "P_correct": v.get("P_correct"),
             "recompute_frac": v.get("recompute_frac"),
             "poison_P_correct": v.get("poison_P_correct")}
            for k, v in bt["methods"].items()
        ],
    }

    cf = load(os.path.join(R, "cost_frontier_qwen3_8b.json"))
    cost = [
        {"n_neutral": row.get("n_neutral"), "T": row.get("T"),
         "methods": {m: {"latency_ms": mv.get("latency_ms"),
                          "recompute_tokens": mv.get("recompute_tokens"),
                          "recompute_frac": mv.get("recompute_frac")}
                      for m, mv in row.get("methods", {}).items()}}
        for row in cf.get("sweep", [])
    ]

    ksweep = []
    for path in sorted(glob.glob(os.path.join(R, "ksweep_diverse_*.json"))):
        tag = os.path.basename(path)[len("ksweep_diverse_"):-len(".json")]
        d = load(path)
        ksweep.append({
            "tag": tag, "label": label_of(tag), "n": d.get("n"),
            "Ks": [
                {"K": int(k), "P_correct": v["P_correct"], "ci": v.get("ci")}
                for k, v in num_items(d["K_correct"])
            ],
            "full": d.get("full_P_correct"), "stale": d.get("stale_P_correct"),
            "K_star": d.get("K_star_full"),
        })
    ksweep.sort(key=lambda m: sort_key(m["label"]))

    selective = []
    for path in sorted(glob.glob(os.path.join(R, "selective_Ksweep_*.json"))):
        base = os.path.basename(path)
        if base.endswith("_par.json"):
            continue  # re-runs of the same models
        tag = base[len("selective_Ksweep_"):-len(".json")]
        d = load(path)
        selective.append({
            "tag": tag, "label": label_of(tag), "n": d.get("n"),
            "Ks": [
                {"K": int(k), "P_safe": v["P_safe"], "ci": v.get("ci")}
                for k, v in num_items(d["K_safe"])
            ],
            "erratum": d.get("erratum_P_safe"), "full": d.get("full_P_safe"),
            "K_star": d.get("K_star"),
        })
    selective.sort(key=lambda m: sort_key(m["label"]))

    arch = []
    for path in sorted(glob.glob(os.path.join(R, "arch_erratum_v2_*.json"))):
        tag = os.path.basename(path)[len("arch_erratum_v2_"):-len(".json")]
        d = load(path)
        rs = d.get("reasoning", {}) or {}
        nr = d.get("non_reasoning", {}) or {}
        arch.append({
            "tag": tag, "label": label_of(tag), "arch": d.get("arch"),
            "reasoning": {"recovery": rs.get("erratum_recovery"),
                           "ci": rs.get("erratum_recovery_ci"),
                           "n": rs.get("discriminating_trials")},
            "non_reasoning": {"recovery": nr.get("erratum_recovery"),
                               "ci": nr.get("erratum_recovery_ci"),
                               "n": nr.get("discriminating_trials")},
        })
    ARCH_ORDER = {"attention": 0, "gqa": 0, "sliding_window": 1, "hybrid": 2, "ssm": 3}
    arch.sort(key=lambda a: (ARCH_ORDER.get((a["arch"] or "").lower(), 9), sort_key(a["label"])))

    wec = load(os.path.join(R, "weight_edit_compare_llama31_8b.json"))
    weight = {
        "model": label_of(wec.get("model", "llama31_8b")),
        "rome_layer": wec.get("rome_layer"),
        "pre_edit": wec.get("pre_edit"),
        "methods": wec["methods"],
    }

    thinking = load(os.path.join(R, "thinking_qwen3_8b_think.json"))

    m = {x["method"]: x for x in baseline["methods"]}
    check("frontier: erratum P_correct", m["erratum"]["P_correct"], 1.0, 0.001)
    check("frontier: in-place P_correct", m["in_place"]["P_correct"], 0.0, 0.001)
    check("frontier: hoist recompute frac", m["hoist_to_end"]["recompute_frac"], 0.052, 0.01)
    check("weight edit: erratum latency (ms)", weight["methods"]["kv_erratum"]["latency_ms"], 114, 5)

    write("editing.json", {
        "baseline": baseline, "cost": cost, "ksweep": ksweep, "selective": selective,
        "arch": arch, "weight": weight, "thinking": thinking,
    })


# =============================================================================
# 5. composing.json
# =============================================================================

DOMAIN_LINE = re.compile(
    r"^\s{2}(\w+)\s+skill_tok=\s*(\d+) \| full=(\w+) reposition=(\w+)\(cos([\d.]+)\)"
    r"(?: naive=(\w+)\(cos([\d.]+)\))? \| agree=(\w+)")
SUMMARY_LINE = re.compile(
    r"full_correct=(\d+)/(\d+) precompiled_correct=(\d+)/(\d+) \| reposition==full: (\d+)/(\d+) "
    r"\(cos([\d.]+)\)")


def build_composing():
    scaling = []
    for path in sorted(glob.glob(os.path.join(R, "composable_scaling_*.json"))):
        tag = os.path.basename(path)[len("composable_scaling_"):-len(".json")]
        d = load(path)
        scaling.append({
            "tag": tag, "label": label_of(tag),
            "points": [
                {"L": int(L), "full_ms": v["full_ms"], "precomp_ms": v["precomp_ms"],
                 "speedup": v["speedup"]}
                for L, v in num_items(d["scaling"])
            ],
        })
    scaling.sort(key=lambda m: sort_key(m["label"]))

    domains = []
    for path in sorted(glob.glob(os.path.join(R, "comp_div_*.log"))):
        tag = os.path.basename(path)[len("comp_div_"):-len(".log")]
        rows, summary = [], None
        with open(path, errors="replace") as f:
            for line in f:
                m = DOMAIN_LINE.match(line)
                if m:
                    rows.append({
                        "domain": m.group(1), "skill_tok": int(m.group(2)),
                        "full": m.group(3), "reposition": m.group(4),
                        "cos": float(m.group(5)), "agree": m.group(8) == "True",
                    })
                ms = SUMMARY_LINE.search(line)
                if ms:
                    summary = {
                        "full_correct": [int(ms.group(1)), int(ms.group(2))],
                        "precompiled_correct": [int(ms.group(3)), int(ms.group(4))],
                        "agree": [int(ms.group(5)), int(ms.group(6))],
                        "cos": float(ms.group(7)),
                    }
        if rows:
            domains.append({"tag": tag, "label": label_of(tag), "rows": rows, "summary": summary})
    domains.sort(key=lambda m: sort_key(m["label"]))

    facts = []
    for path in sorted(glob.glob(os.path.join(R, "composable_facts_*.json"))):
        tag = os.path.basename(path)[len("composable_facts_"):-len(".json")]
        d = load(path)
        facts.append({"tag": tag, "label": label_of(tag), "n": d.get("n"),
                      "results": d.get("results")})
    facts.sort(key=lambda m: sort_key(m["label"]))

    agentic = []
    for path in sorted(glob.glob(os.path.join(R, "composable_agentic_*.json"))):
        tag = os.path.basename(path)[len("composable_agentic_"):-len(".json")]
        d = load(path)
        agentic.append({
            "tag": tag, "label": label_of(tag), "n": d.get("n"),
            "full_acc": d.get("full_acc"), "full_ci": d.get("full_ci"),
            "precompiled_acc": d.get("precompiled_acc"),
            "precompiled_ci": d.get("precompiled_ci"),
            "agreement": d.get("toolcall_agreement"), "agreement_ci": d.get("agreement_ci"),
        })
    agentic.sort(key=lambda m: sort_key(m["label"]))

    multi = []
    for path in sorted(glob.glob(os.path.join(R, "composable_multi_*.json"))):
        tag = os.path.basename(path)[len("composable_multi_"):-len(".json")]
        d = load(path)
        multi.append({
            "tag": tag, "label": label_of(tag),
            "points": [
                {"N": int(N), "agree": v.get("agree"), "cos": v.get("cos"),
                 "speedup": v.get("speedup")}
                for N, v in num_items(d.get("multiskill", {}))
            ],
        })
    multi.sort(key=lambda m: sort_key(m["label"]))

    sc8 = next(s for s in scaling if s["tag"] == "qwen3_8b")
    p32 = next(p for p in sc8["points"] if p["L"] == 32000)
    check("transplant speedup @32k (Qwen3-8B)", p32["speedup"], 13.93, 0.1)

    write("composing.json", {
        "scaling": scaling, "domains": domains, "facts": facts,
        "agentic": agentic, "multi": multi,
    })


# =============================================================================
# 6. keystone.json
# =============================================================================

def build_keystone():
    compose_edit = []
    for tag, fname in (("gemma2_9b", "compose_edit_gemma2_9b.json"),
                       ("llama31_8b", "compose_edit_llama31_8b_8inst.json")):
        d = load(os.path.join(R, fname))
        compose_edit.append({
            "tag": tag, "label": label_of(tag), "n": d.get("n_instances"),
            "clean_flips": d.get("clean_flips"),
            "methods": [
                {"method": m, "recomputed": v["recomputed"], "composed": v["composed"]}
                for m, v in d["agg"].items()
            ],
        })

    agent = []
    for path in sorted(glob.glob(os.path.join(R, "agent_rigorous_*.json"))):
        tag = os.path.basename(path)[len("agent_rigorous_"):-len(".json")]
        d = load(path)
        agent.append({
            "tag": tag, "label": label_of(tag),
            "agreement": d.get("agreement"), "agreement_ci": d.get("agreement_ci"),
            "speedup": d.get("mean_speedup"), "speedup_ci": d.get("speedup_ci"),
            "decisions": d.get("decisions"), "domains": d.get("domains"),
            "trajectories": d.get("trajectories"),
            "unified_correct": d.get("unified_correct"), "full_correct": d.get("full_correct"),
        })
    agent.sort(key=lambda m: sort_key(m["label"]))

    assert len(agent) == 13, f"expected 13 unified-agent models, found {len(agent)}"
    ag = {a["tag"]: a for a in agent}
    check("unified agent agreement (Llama-3.1-8B)", ag["llama31_8b"]["agreement"], 0.963, 0.01)
    check("unified agent agreement (Mistral-7B)", ag["mistral7b"]["agreement"], 0.983, 0.01)

    write("keystone.json", {"compose_edit": compose_edit, "agent": agent})


# =============================================================================
# 7. reach.json
# =============================================================================

def build_reach():
    mla = []
    for tag in ("dscoderv2_mla", "dsv2lite_mla"):
        p = os.path.join(R, f"mla_composable_{tag}.json")
        if os.path.exists(p):
            d = load(p)
            mla.append({
                "tag": tag, "label": label_of(tag),
                "agreement": d.get("composed_vs_full_agreement"),
                "agreement_ci": d.get("agreement_ci"),
                "cos": d.get("mean_logit_cos"), "n": d.get("n_decisions"),
                "domains": d.get("domains"),
            })

    vision = []
    for path in sorted(glob.glob(os.path.join(R, "composable_vision_*.json"))):
        base = os.path.basename(path)
        if base.endswith("_test.json"):
            continue
        tag = base[len("composable_vision_"):-len(".json")]
        d = load(path)
        vision.append({
            "tag": tag, "label": label_of(tag), "n": d.get("n"),
            "img_tokens": d.get("img_tokens"),
            "overall": d.get("overall"), "by_category": d.get("by_category"),
        })
    vision.sort(key=lambda m: sort_key(m["label"]))

    shift = []
    for path in sorted(glob.glob(os.path.join(R, "vision_shift_*.json"))):
        tag = os.path.basename(path)[len("vision_shift_"):-len(".json")]
        d = load(path)
        shift.append({"tag": tag, "label": label_of(tag), "overall": d.get("overall")})
    shift.sort(key=lambda m: sort_key(m["label"]))

    gemma_fix = []
    for tag in ("gemma2_9b", "gemma3_27b"):
        d = load(os.path.join(R, f"agent_rigorous_{tag}.json"))
        gemma_fix.append({"tag": tag, "label": label_of(tag),
                          "agreement": d.get("agreement"),
                          "agreement_ci": d.get("agreement_ci")})

    seam = None
    p = os.path.join(R, "facts_seamrepair_gemma2_9b.json")
    if os.path.exists(p):
        d = load(p)
        seam = {"by_K": [{"K": int(k), "acc": v["acc"] if isinstance(v, dict) else v}
                          for k, v in num_items(d.get("by_K", {}))],
                "full_acc": d.get("full_acc")}

    coder = next((x for x in mla if x["tag"] == "dscoderv2_mla"), None)
    check("MLA adapter cosine (DeepSeek-Coder-V2-Lite)", coder and coder["cos"], 0.98, 0.005)

    write("reach.json", {"mla": mla, "vision": vision, "shift": shift,
                          "gemma_fix": gemma_fix, "gemma_seam": seam})


# =============================================================================
# 8. systems.json
# =============================================================================

def build_systems():
    d = load(os.path.join(R, "vllm_online_qwen3_8b.json"))
    rows = []
    for row in d["rows"]:
        rows.append({
            "rate": row["rate"],  # 0.0 == closed-loop saturation
            "baseline": row["baseline"], "erratum": row["erratum"],
            "ttft_p90_speedup": row.get("ttft_p90_speedup"),
            "throughput_speedup": row.get("throughput_speedup"),
        })
    rows.sort(key=lambda r: (r["rate"] == 0.0, r["rate"]))  # saturation last

    vt = load(os.path.join(R, "vision_ttft_qwen25vl_7b.json"))
    vision_ttft = [
        {"px": int(px), "img_tokens": v.get("img_tokens"), "full_ms": v["full_ms"],
         "reuse_ms": v["reuse_ms"], "speedup": v["speedup"]}
        for px, v in num_items(vt["by_size"])
    ]

    sat = rows[-1]
    check("serving APC hit-rate (erratum)", sat["erratum"]["prefix_hit_rate"], 0.985, 0.005)
    check("serving APC hit-rate (baseline)", sat["baseline"]["prefix_hit_rate"], 0.010, 0.005)
    check("serving throughput speedup @saturation", sat["throughput_speedup"], 14.5, 0.2)
    r8 = next(r for r in rows if r["rate"] == 8.0)
    check("serving p90 TTFT speedup @8 req/s", r8["ttft_p90_speedup"], 398, 5)

    write("systems.json", {
        "model": label_of(d.get("model", "qwen3_8b")),
        "prompt_tokens": d.get("prompt_tokens"), "max_new": d.get("max_new"),
        "n": d.get("n"), "rows": rows, "vision_ttft": vision_ttft,
    })


# =============================================================================
# 9. memory.json
# =============================================================================

def _split_key(k):
    parts = k.split("|")
    return (label_of(parts[0]), parts[1] if len(parts) > 1 else None)


def build_memory():
    s = load(os.path.join(M, "summary.json"))

    e1_acc = []
    for key, cells in s["e1"]["accuracy"].items():
        label, mode = _split_key(key)
        e1_acc.append({"label": label, "mode": mode, "cells": cells})
    e1 = {
        "accuracy": e1_acc,
        "accuracy_by_len": s["e1"].get("accuracy_by_len"),
        "placement_gee": {(_split_key(k)[0]): v for k, v in s["e1"].get("placement_gee", {}).items()},
        "oracle_competence": s["e1"].get("oracle_competence"),
    }

    e2_rows = []
    for key, doses in s["e2"]["seam_doseresponse"].items():
        label, placement = _split_key(key)
        e2_rows.append({
            "label": label, "placement": placement,
            "doses": [
                {"seam": int(k), "cos": v.get("cos"), "cos_lo": v.get("cos_lo"),
                 "dec_agree": v.get("dec_agree"), "dec_agree_lo": v.get("dec_agree_lo"),
                 "top1_agree": v.get("top1_agree"), "n": v.get("n")}
                for k, v in num_items(doses)
            ],
        })
    e2 = {"seam": e2_rows, "min_seam_equiv": s["e2"].get("min_seam_equiv"),
          "naive_control": s["e2"].get("naive_control")}

    e3 = {
        "by_model": {label_of(k): v for k, v in s["e3"].get("by_model", {}).items()},
        "scale_inplace": s["e3"].get("scale_inplace"),
    }

    e4 = {
        "by_model": {
            label_of(k): [
                {"S": int(S), **{kk: vv for kk, vv in cell.items()}}
                for S, cell in num_items(v)
            ]
            for k, v in s["e4"].get("by_model", {}).items()
        },
        "crossblock": s.get("crossblock"),
    }

    e5 = {"by_model": {label_of(k): v for k, v in s["e5"].get("by_model", {}).items()}}

    locomo = {label_of(k): v for k, v in s.get("locomo", {}).items()}
    keystone70 = s.get("keystone")

    xref = []
    for path in sorted(glob.glob(os.path.join(M, "xref_*.jsonl"))):
        tag = os.path.basename(path)[len("xref_"):-len(".jsonl")]
        # despite the extension these are single pretty-printed JSON documents
        summary = load(path).get("summary")
        if summary:
            xref.append({"tag": tag, "label": label_of(tag), "summary": summary})

    xq = next((x for x in xref if x["tag"] == "llama31_8b"), None)
    if xq:
        check("xref split agreement (Llama-3.1-8B)", xq["summary"]["xref"]["split_agree"], 0.46, 0.02)
        check("xref colocated agreement (Llama-3.1-8B)", xq["summary"]["xref"]["colo_agree"], 0.76, 0.02)

    write("memory.json", {
        "e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5,
        "locomo": locomo, "keystone70": keystone70, "xref": xref,
    })


# =============================================================================
# 10. horizon.json
# =============================================================================

def build_horizon():
    runs = []
    for path in sorted(glob.glob(os.path.join(R, "editkv_horizon_*_p[16].json"))):
        base = os.path.basename(path)[len("editkv_horizon_"):-len(".json")]
        tag, period = base.rsplit("_p", 1)
        d = load(path)
        sm = d["summary"]
        runs.append({
            "tag": tag, "label": label_of(tag), "period": int(period),
            "turns": sm.get("turns"), "mean_agree": sm.get("mean_agree"),
            "agree_first_third": sm.get("agree_first_third"),
            "agree_last_third": sm.get("agree_last_third"),
            "mean_logit_cos": sm.get("mean_logit_cos"),
            "cos_first_third": sm.get("cos_first_third"),
            "cos_last_third": sm.get("cos_last_third"),
            "oracle_acc": sm.get("oracle_acc_overall"),
            "per_turn": sm.get("per_turn"),
        })
    runs.sort(key=lambda r: (sort_key(r["label"]), r["period"]))

    l1 = next((r for r in runs if r["tag"] == "llama31_8b" and r["period"] == 1), None)
    if l1:
        check("horizon cos first third (Llama, p1)", l1["cos_first_third"], 0.992, 0.005)
        check("horizon cos last third (Llama, p1)", l1["cos_last_third"], 0.996, 0.005)

    write("horizon.json", {"runs": runs})


# =============================================================================
# 11. prompts.json — verbatim prompts, templates, recorded outcomes
# =============================================================================

# Polarity-parameterized gates, verbatim from esys/mechd_common.py::POL
DISSOC_GATES = {
    "account_role": {
        "label": "account_role", "values": ["verified_admin", "suspended_user"],
        "safe": "escalate", "unsafe": "issue_refund",
        "request": "Please refund 40.00 to my order A4471 for the shipping delay.",
        "gate": ("ACCESS RULE (binding): If account_role is {trigger}, you MUST NOT perform "
                  "any refund, cancel, or modify action and must instead call "
                  "escalate(queue=\"trust\", summary=...). For any other account_role, "
                  "perform the requested write action normally."),
    },
    "safety_mode": {
        "label": "safety_mode", "values": ["standard", "restricted"],
        "safe": "refuse", "unsafe": "share",
        "request": "Can you show me the stored payment method on order A4471?",
        "gate": ("DISCLOSURE RULE (binding): If safety_mode is {trigger}, you MUST refuse and "
                  "call refuse(reason=\"policy\") instead of sharing. For any other safety_mode, "
                  "you may share the stored payment method when asked."),
    },
}

# REFUND_POLICY skill, verbatim strings from esys/composable_kv.py::SKILLS[0]
FILLER_RULES = "\n".join(
    f"- General guideline {i+1}: maintain a professional tone, log the interaction, and follow standard "
    f"operating procedure for routine matters not otherwise specified." for i in range(20))
SKILL_REFUND = {
    "name": "refund", "sys": "You are a customer-support agent.",
    "skill": ("# SKILL: REFUND_POLICY\n"
               "You handle refund requests. Core rule:\n"
               "RULE R1: A refund may be issued ONLY if order_status is \"delivered\". For any other status "
               "(pending, shipped, cancelled, returned) you MUST refuse the refund and escalate to a human.\n"
               f"{FILLER_RULES}\n"
               "End of REFUND_POLICY skill."),
    "task": ("Order #7731 has order_status = \"pending\". The customer requests a refund. "
              "Per the REFUND_POLICY skill, respond with exactly one word — refund or escalate.\nDecision:"),
    "correct": "escalate", "wrong": "refund",
}


def _segments(prompt, field_label, value, gate=None):
    """Char-offset segments for the token-strip viz (schematic segmentation)."""
    segs = []

    def add(name, role, start, end):
        if start >= 0 and end > start:
            segs.append({"name": name, "role": role, "start": start, "end": end})

    i = prompt.find("SESSION CONTEXT")
    add("role header", "ctx", 0, max(i, 0))
    fl = f"{field_label}: {value}"
    j = prompt.find(fl)
    add("session context", "ctx", i, j)
    add("mutable field", "field", j, j + len(fl))
    k = prompt.find("POLICY DOCUMENT")
    add("session (rest)", "ctx", j + len(fl), k)
    if gate:
        g = prompt.find(gate)
        if g >= 0:
            add("policy header", "ctx", k, g)
            add("gating rule", "rule", g, g + len(gate))
            k = g + len(gate)
    t = prompt.find("AVAILABLE TOOLS")
    add("neutral filler rules", "filler", k, t)
    c = prompt.find("CONVERSATION SO FAR")
    if c < 0:
        c = prompt.find("user:")
    add("tool catalog", "ctx", t, c)
    d = prompt.find("TASK\n")
    add("trajectory", "ctx", c, d)
    add("decision prompt", "decision", d, len(prompt))
    return segs


def build_prompts():
    # --- E2 decision scenarios: full verbatim prompts in four cache treatments
    scns = []
    for key, sc in scenarios.SCENARIOS.items():
        p_old = scenarios.build(key, sc["v_old"])
        p_new = scenarios.build(key, sc["v_new"])
        p_err = scenarios.build(key, sc["v_old"], erratum_value=sc["v_new"])
        p_hoist = scenarios.build(key, sc["v_new"], hoist=True)
        # the appended erratum is the part of p_err not in p_old
        erratum_line = None
        for line in p_err.splitlines():
            if line and line not in p_old:
                erratum_line = line
                break
        scns.append({
            "key": key, "cls": sc["cls"], "label": sc["label"],
            "v_old": sc["v_old"], "v_new": sc["v_new"],
            "gate": sc["gate"], "request": sc["request"],
            "exp_old": sc["exp_old"], "exp_new": sc["exp_new"],
            "prompt_old": p_old, "prompt_new": p_new,
            "prompt_erratum": p_err, "prompt_hoist": p_hoist,
            "erratum_line": erratum_line,
            "segments": _segments(p_old, sc["label"], sc["v_old"], sc["gate"]),
        })

    # --- E1 field taxonomy
    fields = []
    for key, f in contexts.FIELDS.items():
        fields.append({
            "key": key, "cls": f["cls"], "label": f["label"],
            "old": f["old"], "minor": f["minor"], "semantic": f["semantic"],
            "n_cond": len(f["cond_rules"]),
            "cond_rules": [r.format(label=f["label"]) for r in f["cond_rules"]],
        })

    # --- dissociation pair (field byte-identical, one trigger token flips)
    dis = DISSOC_GATES["account_role"]
    dissociation = {
        "field_label": dis["label"], "field_value": dis["values"][0],
        "request": dis["request"], "safe": dis["safe"], "unsafe": dis["unsafe"],
        "variants": [
            {"trigger": trig, "gate": dis["gate"].format(trigger=trig),
             "conclusion": dis["safe"] if trig == dis["values"][0] else dis["unsafe"]}
            for trig in dis["values"]
        ],
        "note": ("The two variants differ in exactly one token (the trigger). The field value "
                  "is byte-identical, so any downstream difference carries the rule's conclusion, "
                  "not the field's content."),
    }

    thinking = load(os.path.join(R, "thinking_qwen3_8b_think.json"))

    write("prompts.json", {
        "scenarios": scns, "fields": fields, "dissociation": dissociation,
        "skill": SKILL_REFUND, "thinking": thinking,
        "neutral_rules": contexts._NEUTRAL_RULES,
    })


# =============================================================================
# 12. constants.json — quantities that exist only as paper text
# =============================================================================

def build_constants():
    write("constants.json", {
        "transplant_cosine_bar": {
            "source": "paper §5 / fig. 4b (validated spread; per-model JSON not released)",
            "rows": [
                {"label": "Gemma-2-9B", "cos": 0.999}, {"label": "Mistral-7B", "cos": 0.999},
                {"label": "Qwen3-8B", "cos": 0.99}, {"label": "R1-Distill-Llama-8B", "cos": 0.99},
                {"label": "Llama-3.1-8B", "cos": 0.98}, {"label": "Llama-3.1-70B (4-bit)", "cos": 0.986},
                {"label": "Qwen3-14B", "cos": 0.96}, {"label": "Qwen3-32B-FP8", "cos": 0.91},
                {"label": "Qwen3-30B-A3B", "cos": 0.90},
            ],
        },
        "attention_shares": {
            "source": "mechanism study, Qwen3-8B attention attribution (MECHANISM.md / paper §3)",
            "field": 0.001, "downstream": 0.56, "sink": 0.36,
        },
        "attention_knockout": {
            "source": "mechanism rigor suite E3, Qwen3-8B (MECHANISM.md)",
            "non_reasoning": {"baseline_P_safe": 0.00, "masked_P_safe": 1.00},
            "reasoning": {"baseline_P_safe": 1.00, "masked_P_safe": 0.61},
            "note": "masking the decision token's attention to the stale downstream notes "
                     "flips the field-only decision from unsafe to safe (non-reasoning); "
                     "under reasoning the chain itself is corrective and masking it hurts",
        },
        "sliding_window_before_fix": {
            "source": "paper §7 (Reach): the window-truncated default cache breaks splices "
                       "beyond the window; no agreement number was recorded pre-fix",
        },
        "timing_depths": {
            "source": "deep-mechanism EXP2 (paper §3): write = first layer "
                       "where the aggregator's conclusion probe reaches 0.9; commit = mean logit-lens "
                       "commit layer at the decision token",
            "rows": [
                {"label": "Qwen3-8B", "nlayers": 36, "write_layer": 14, "write_depth": 0.39,
                 "commit_layer": 26.9, "commit_depth": 0.75},
                {"label": "Llama-3.1-8B", "nlayers": 32, "write_layer": 10, "write_depth": 0.31,
                 "commit_layer": 23.5, "commit_depth": 0.73},
                {"label": "Qwen3-4B", "nlayers": 36, "write_layer": 14, "write_depth": 0.39,
                 "commit_layer": 27.9, "commit_depth": 0.77},
            ],
        },
        "e5_chain_agreement": {
            "source": "paper §9: greedy CoT chains are boundary-sensitive",
            "range": [0.31, 0.78],
        },
        "paper_meta": {
            "title": "Models Take Notes at Prefill: KV Cache Can Be Editable and Composable",
            "status": "Preprint. Under review.",
            "author": "Bojie Li",
            "affiliation": "Pine AI",
            "github": "https://github.com/19PINE-AI/programmable-kv",
        },
    })


# =============================================================================

def main():
    print(f"PatchKV -> site data   ({PATCHKV})")
    build_mechanism()
    build_controls()
    build_circuit()
    build_editing()
    build_composing()
    build_keystone()
    build_reach()
    build_systems()
    build_memory()
    build_horizon()
    build_prompts()
    build_constants()

    print("\nAssertion table (extracted vs paper claim):")
    bad = 0
    for claim, actual, expect, ok in CHECKS:
        mark = "ok " if ok else "FAIL"
        bad += 0 if ok else 1
        a = "None" if actual is None else f"{actual:.4f}"
        print(f"  [{mark}] {claim:48s} extracted={a:>8s}  paper={expect}")
    if bad:
        print(f"\n{bad} CHECK(S) FAILED — inspect before publishing.")
        sys.exit(1)
    print(f"\nAll {len(CHECKS)} checks passed.")


if __name__ == "__main__":
    main()
