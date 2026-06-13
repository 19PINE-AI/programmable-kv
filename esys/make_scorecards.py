"""Two scorecards across models:
 (1) FULL-SUBSTRATE: per model, editable (D1 field-only/full + erratum recovery) AND composable
     (keystone sel@32 + erratum, composed) -> shows the substrate validates on MANY models, not one.
 (2) FIELD+SELECTIVE: per model, the field+selective edit recovery (keystone composed sel@8/sel@32)
     -> unreliable, but shows which models it works for.
Reads results/{mech_causal_patch,arch_erratum_v2,compose_edit}_*.json. Run: python esys/make_scorecards.py
"""
import json, os
R = os.path.join(os.path.dirname(__file__), "..", "results")


def L(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


# (label, d1_tag, erratum_tag, keystone_tag)
MODELS = [
    ("Qwen3-4B", "qwen3_4b", "qwen3_4b", "qwen3_4b"),
    ("Qwen3-8B", "qwen3_8b", "Qwen3-8B", "qwen3_8b"),
    ("Qwen3-14B", "qwen3_14b", "qwen3_14b", "qwen3_14b"),
    ("Qwen3-32B-FP8", "qwen3_32b", None, None),
    ("Gemma-2-9B", "gemma2_9b", "gemma2_9b", "gemma2_9b"),
    ("Gemma-3-27B", "gemma3_27b_bf16", "gemma3_27b_bf16", None),
    ("Mistral-7B", "mistral7b", "mistral7b", "mistral7b"),
    ("Llama-3.1-8B", "llama31_8b", "llama31_8b", "llama31_8b_8inst"),
    ("DeepSeek-R1-Llama-8B", None, None, "dsr1_llama8b"),
]


def fmt(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else "—"


def main():
    print("\n=== (1) FULL-SUBSTRATE SCORECARD (editable + composable per model) ===")
    print(f"{'model':22s} | {'D1 field':>8} {'D1 full':>7} | {'err(nr)':>7} {'err(r)':>7} | "
          f"{'key sel@32':>10} {'key erratum':>11}")
    rows = []
    for label, d1, er, ks in MODELS:
        d = L(f"mech_causal_patch_{d1}.json") if d1 else None
        fo = d["agg"]["field_only_recovery"]["mean"] if d else None
        fu = d["agg"]["full_downstream_recovery"]["mean"] if d else None
        e = L(f"arch_erratum_v2_{er}.json") if er else None
        enr = e["non_reasoning"]["erratum_recovery"] if e else None
        err = e["reasoning"]["erratum_recovery"] if e else None
        k = L(f"compose_edit_{ks}.json") if ks else None
        ks32 = k["agg"]["sel@32"]["composed"] if k else None
        kerr = k["agg"]["erratum"]["composed"] if k else None
        complete = all(v is not None for v in [fo, fu, enr, ks32])
        rows.append((label, fo, fu, enr, err, ks32, kerr, complete))
        print(f"{label:22s} | {fmt(fo):>8} {fmt(fu):>7} | {fmt(enr):>7} {fmt(err):>7} | "
              f"{fmt(ks32):>10} {fmt(kerr):>11}" + ("  [FULL]" if complete else ""))
    nfull = sum(r[7] for r in rows)
    print(f"  -> {nfull} models with the COMPLETE substrate (D1 + erratum + keystone).")

    print("\n=== (2) FIELD+SELECTIVE SCORECARD (composed recovery; unreliable but works for some) ===")
    print(f"{'model':22s} | {'in_place':>8} {'sel@8':>7} {'sel@32':>7} | verdict")
    for label, d1, er, ks in MODELS:
        k = L(f"compose_edit_{ks}.json") if ks else None
        if not k:
            continue
        ip = k["agg"]["in_place"]["composed"]; s8 = k["agg"]["sel@8"]["composed"]; s32 = k["agg"]["sel@32"]["composed"]
        verdict = "WORKS" if s32 >= 0.6 else ("partial" if s32 >= 0.3 else "fails")
        print(f"{label:22s} | {fmt(ip):>8} {fmt(s8):>7} {fmt(s32):>7} | {verdict}")
    print("  (field+selective@32 composed: WORKS>=0.6, partial>=0.3, else fails; in_place~0 everywhere = memoization)")
    print("SCORECARDS_DONE")


if __name__ == "__main__":
    main()
