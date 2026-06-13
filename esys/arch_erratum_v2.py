"""Rigorous architecture x {reasoning, non-reasoning} erratum test with BOOTSTRAP CIs.

Replaces the earlier N=1 single-mode check. For each model and each mode we run MULTIPLE gating
scenarios x K stochastic samples, and report the erratum recovery rate (P(erratum == the oracle's
flipped action), over scenarios where the oracle actually flips) with a bootstrap 95% CI (B=10000).

  non-reasoning mode: ask for the one-word answer directly.
  reasoning  mode   : CoT-prompt ("think step by step ... end with 'Answer: <word>'") and read the
                      final answer. (These backbones are not native thinking models, so we elicit
                      CoT by prompt — which is exactly the interesting test for pure SSM: the CoT is
                      generated AFTER the erratum, so the recurrent state processes the override last.)

Question: is the erratum's failure on PURE SSM (no attention to look back) rescued by CoT
re-derivation, or is it a hard architectural limit? Run on attention / hybrid / pure-SSM models.
Run: MECH_ATTN=sdpa python esys/arch_erratum_v2.py --model <id> --arch "<label>" --K 8
"""
import argparse, os, sys, json, re, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# scenarios: (policy, field, old_val, new_val, old_action, new_action)  -- simple, distinct words
SCEN = [
    ("An order may be CANCELLED only if its order_status is 'pending'; if 'processed' it CANNOT be "
     "cancelled and you must deny.", "order_status", "pending", "processed", "cancel", "deny"),
    ("Process a refund only if account_role is 'admin'; if 'suspended' you must not refund and must "
     "deny.", "account_role", "admin", "suspended", "refund", "deny"),
    ("Grant access to the document only if clearance is 'secret'; if 'public' you must deny.",
     "clearance", "secret", "public", "grant", "deny"),
    ("Ship the item only if stock_status is 'in_stock'; if 'out_of_stock' you must backorder.",
     "stock_status", "in_stock", "out_of_stock", "ship", "backorder"),
]
ERR = ("\n\n[STATE UPDATE] The {field} has just changed to '{nv}'. This overrides any earlier value "
       "AND any earlier conclusion. Apply the current value.")


def build(policy, field, val, a, b, mode, erratum_nv=None):
    body = (f"You are an operations agent. Binding policy: {policy}\n\nThe current {field} is: {val}." +
            (ERR.format(field=field, nv=erratum_nv) if erratum_nv else "") +
            f"\n\nGiven the policy and the CURRENT {field}, what is the correct action: '{a}' or '{b}'?")
    if mode == "reasoning":
        return body + " Think step by step, then end your response with exactly 'Answer: <word>'."
    return body + f" Answer with exactly one word — '{a}' or '{b}'."


def extract(text, a, b):
    t = text.lower()
    m = re.search(r"answer:\s*([a-z_]+)", t)
    if m:
        w = m.group(1)
        if a in w or w in a:
            return a
        if b in w or w in b:
            return b
    # fallback: last mention of either action word
    ia, ib = t.rfind(a), t.rfind(b)
    if ia < 0 and ib < 0:
        return "?"
    return a if ia > ib else b


@torch.no_grad()
def gen(model, tok, content, mode, seed, greedy):
    msgs = [{"role": "user", "content": content}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = content + "\nAnswer:"
    ids = tok(text, return_tensors="pt").to(model.device)
    mn = 200 if mode == "reasoning" else 8
    if greedy:
        out = model.generate(**ids, max_new_tokens=mn, do_sample=False, pad_token_id=tok.eos_token_id or 0)
    else:
        torch.manual_seed(seed)
        out = model.generate(**ids, max_new_tokens=mn, do_sample=True, temperature=0.7, top_p=0.95,
                             pad_token_id=tok.eos_token_id or 0)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def boot_ci(xs, B=10000, seed=0):
    """Proper bootstrap 95% CI: B resamples WITH REPLACEMENT (fixed seed -> reproducible)."""
    n = len(xs)
    if n == 0:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(B))
    return [round(means[int(0.025 * B)], 3), round(means[int(0.975 * B)], 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arch", default="?")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    impl = os.environ.get("MECH_ATTN", "sdpa")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    kw = dict(device_map="cuda", attn_implementation=impl, trust_remote_code=True)
    quantized = any(q in args.model.upper() for q in ("FP8", "-INT8", "GPTQ", "AWQ", "QUANTIZED.W", "W8A", "W4A"))
    if os.environ.get("BNB_8BIT"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif not quantized:
        kw["dtype"] = torch.bfloat16
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()
    except (ValueError, KeyError):
        kw.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()
    res = {"model": args.model, "arch": args.arch, "K": args.K}
    print(f"=== {args.arch}: {args.model} (K={args.K} samples/scenario) ===", flush=True)
    for mode in ["non_reasoning", "reasoning"]:
        recov, flips = [], []   # recov: 1 if erratum==new_action (only on flipping scenarios)
        oracle_new, stale_old = [], []
        for (policy, field, ov, nv, a, b) in SCEN:
            for k in range(args.K):
                greedy = (k == 0)
                o = extract(gen(model, tok, build(policy, field, nv, a, b, mode), mode, 100 + k, greedy), a, b)
                s = extract(gen(model, tok, build(policy, field, ov, a, b, mode), mode, 200 + k, greedy), a, b)
                e = extract(gen(model, tok, build(policy, field, ov, a, b, mode, erratum_nv=nv), mode, 300 + k, greedy), a, b)
                oracle_new.append(o == b); stale_old.append(s == a)
                flipped = (o == b and s == a)            # this trial's scenario discriminates
                flips.append(flipped)
                if flipped:
                    recov.append(1 if e == b else 0)
        n_flip = sum(flips)
        res[mode] = {
            "n_trials": len(flips), "oracle_picks_new": round(sum(oracle_new) / len(oracle_new), 3),
            "stale_picks_old": round(sum(stale_old) / len(stale_old), 3),
            "discriminating_trials": n_flip,
            "erratum_recovery": round(sum(recov) / len(recov), 3) if recov else None,
            "erratum_recovery_ci": boot_ci(recov) if recov else None,
            "recov_raw": recov}
        r = res[mode]
        print(f"  [{mode:13s}] oracle_new={r['oracle_picks_new']} stale_old={r['stale_picks_old']} "
              f"| discriminating={n_flip}/{len(flips)} | ERRATUM_RECOVERY={r['erratum_recovery']} "
              f"CI{r['erratum_recovery_ci']}", flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"arch_erratum_v2_{tag}.json"), "w"), indent=2)
    print("ARCH_ERRATUM_V2_DONE")


if __name__ == "__main__":
    main()
