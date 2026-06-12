"""E-horizon - Compounding error over a long trajectory (the research plan's novel axis).

CacheBlend et al. measure F1/Rouge, not degeneration over steps. We ask: if we maintain ONE
evolving KV cache (reuse the static prefix forever, apply state changes as appended errata, leave
all downstream KV stale) over a long agent trajectory, does the decision drift from a full-reprefill
oracle as turns-after-patch accumulate?

Clean isolation: the patched path and the oracle see the BYTE-IDENTICAL token sequence (errata
included); the ONLY difference is whether downstream KV was recomputed after each state change
(oracle: full reprefill every turn; patched: never recompute the prefix). So per-turn decision
agreement measures exactly the cost of leaving KV stale, as a function of trajectory length.

Stress design: a single gated field (clearance) toggles between a granting and a denying value
every `period` turns, so (a) the correct decision flips, and (b) by turn t the cache holds t stale
errata and t stale earlier conclusions that the latest erratum must override.
Run: python esys/editkv_horizon.py --model Qwen/Qwen3-8B --turns 24
"""
import argparse, os, sys, json
import torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM, AutoTokenizer
from composable_kv import prefill, cache_slice, forward_suffix

SYS = ("You are an access-control agent. Apply the RULE to the CURRENT clearance.\n"
       "RULE (binding): grant a CONFIDENTIAL record only if clearance is L4 or L5; "
       "for clearance L1, L2, or L3 you MUST deny.")
REQS = [
    "the auditor asks to view record #{n}.", "please pull the confidential file for case {n}.",
    "I need the sealed report {n} for review.", "open the restricted dossier {n}.",
    "show me the classified memo {n}.", "retrieve confidential exhibit {n}.",
    "grant me the protected record {n}.", "access the secure document {n}.",
]
PROBE = ("\n[REQUEST] Per the RULE and the CURRENT clearance, answer with exactly one word — "
         "grant or deny.\nDecision:")
GRANT_VAL, DENY_VAL = "L5", "L2"


def boot_ci(xs, B=2000):
    n = len(xs)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (round(float(xs[0]), 3), round(float(xs[0]), 3))
    m = sorted(sum(xs[(b * 2654435761 + j * 40503) % n] for j in range(n)) / n for b in range(B))
    return (round(m[int(0.025 * B)], 3), round(m[int(0.975 * B)], 3))


@torch.no_grad()
def run(model, tok, T, period, seed_shift=0):
    tg = tok("grant", add_special_tokens=False)["input_ids"][0]
    td = tok("deny", add_special_tokens=False)["input_ids"][0]
    probe_ids = tok(PROBE, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")

    clr = GRANT_VAL
    policy = SYS + f"\n\nSESSION: clearance = {clr}.\n"
    pre = tok(policy, add_special_tokens=True, return_tensors="pt")["input_ids"].to("cuda")
    cache_p = prefill(model, pre); pos_p = pre.shape[1]
    oracle_text = policy
    per_turn = []

    for t in range(T):
        n = 1000 + t + seed_shift
        turn = f"\n[TURN {t + 1}] user: {REQS[(t + seed_shift) % len(REQS)].format(n=n)}"
        if t > 0 and t % period == 0:
            clr = DENY_VAL if clr == GRANT_VAL else GRANT_VAL
            turn += (f"\n[STATE UPDATE] clearance is now {clr}; this overrides any earlier "
                     f"clearance value and any earlier decision.")
        turn += "\nassistant: acknowledged."
        turn_ids = tok(turn, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")

        # patched: extend the persistent cache (never recompute the prefix)
        cache_p = forward_suffix(model, cache_p, turn_ids, pos_p).past_key_values
        pos_p += turn_ids.shape[1]
        cp = cache_slice(cache_p, 0, pos_p)
        op = forward_suffix(model, cp, probe_ids, pos_p)
        lg_p = op.logits[0, -1].float()
        dp = "grant" if lg_p[tg] >= lg_p[td] else "deny"

        # oracle: full reprefill of the identical text so far + probe
        oracle_text += turn
        full_ids = tok(oracle_text + PROBE, add_special_tokens=True, return_tensors="pt")["input_ids"].to("cuda")
        lg_o = model(input_ids=full_ids, use_cache=False).logits[0, -1].float()
        do = "grant" if lg_o[tg] >= lg_o[td] else "deny"

        correct = "grant" if clr == GRANT_VAL else "deny"
        cos = float(torch.nn.functional.cosine_similarity(lg_p, lg_o, dim=0))
        per_turn.append({"t": t + 1, "clr": clr, "patched": dp, "oracle": do, "correct": correct,
                         "agree": dp == do, "patched_correct": dp == correct,
                         "oracle_correct": do == correct, "logit_cos": round(cos, 4),
                         "margin_p": round(float(lg_p[tg] - lg_p[td]), 2)})
    return per_turn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--turns", type=int, default=24)
    ap.add_argument("--period", type=int, default=1, help="toggle clearance every `period` turns")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    quant = any(q in args.model.upper() for q in ("FP8", "-INT8", "GPTQ", "AWQ", "W8A", "W4A"))
    kw = dict(device_map="cuda", attn_implementation="sdpa", trust_remote_code=True)
    if not quant:
        kw["dtype"] = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()

    reps = [run(model, tok, args.turns, args.period, seed_shift=r) for r in range(args.reps)]
    # aggregate per-turn across reps
    agg = []
    for ti in range(args.turns):
        rows = [rep[ti] for rep in reps]
        agg.append({"t": ti + 1,
                    "agree": round(sum(r["agree"] for r in rows) / len(rows), 3),
                    "patched_correct": round(sum(r["patched_correct"] for r in rows) / len(rows), 3),
                    "oracle_correct": round(sum(r["oracle_correct"] for r in rows) / len(rows), 3),
                    "logit_cos": round(sum(r["logit_cos"] for r in rows) / len(rows), 4)})
    # decay test: agreement & correctness in first vs last third
    third = max(1, args.turns // 3)
    def band(rows, key, lo, hi):
        vals = [a[key] for a in rows[lo:hi]]
        return round(sum(vals) / len(vals), 3)
    flat_agree = [a["agree"] for a in agg]
    summary = {
        "model": args.model, "turns": args.turns, "period": args.period, "reps": args.reps,
        "mean_agree": round(sum(flat_agree) / len(flat_agree), 3),
        "agree_first_third": band(agg, "agree", 0, third),
        "agree_last_third": band(agg, "agree", args.turns - third, args.turns),
        "patched_acc_first_third": band(agg, "patched_correct", 0, third),
        "patched_acc_last_third": band(agg, "patched_correct", args.turns - third, args.turns),
        "oracle_acc_overall": round(sum(a["oracle_correct"] for a in agg) / len(agg), 3),
        "mean_logit_cos": round(sum(a["logit_cos"] for a in agg) / len(agg), 4),
        "cos_first_third": band(agg, "logit_cos", 0, third),
        "cos_last_third": band(agg, "logit_cos", args.turns - third, args.turns),
        "per_turn": agg,
    }
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    json.dump({"summary": summary, "reps": reps}, open(os.path.join(
        os.path.dirname(__file__), "..", "results", f"editkv_horizon_{tag}.json"), "w"), indent=2)
    print(f"=== E-horizon ({args.model}) T={args.turns} period={args.period} reps={args.reps} ===")
    print(f"  oracle accuracy overall: {summary['oracle_acc_overall']} (sanity: should be high)")
    print(f"  patched==oracle agreement: mean {summary['mean_agree']}  "
          f"(first third {summary['agree_first_third']} -> last third {summary['agree_last_third']})")
    print(f"  patched accuracy: first third {summary['patched_acc_first_third']} -> "
          f"last third {summary['patched_acc_last_third']}")
    print(f"  decision-logit cosine: mean {summary['mean_logit_cos']}  "
          f"(first {summary['cos_first_third']} -> last {summary['cos_last_third']})")
    decayed = summary["agree_last_third"] < summary["agree_first_third"] - 0.1
    print(f"  COMPOUNDING DECAY: {'YES (>0.1 drop)' if decayed else 'NO (stays flat)'}")
    print("EDITKV_HORIZON_DONE")


if __name__ == "__main__":
    main()
