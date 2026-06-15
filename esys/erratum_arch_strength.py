"""Is the erratum weaker on PURE SSM (no attention to "look back" at the override)?

Earlier finding (pure-SSM erratum strength): on pure Mamba the oracle flips (the model tracks the field)
but the standard erratum FAILS to override, while hybrid (Falcon-H1) and attention models recover. Mechanism
(§7): the erratum works because the decision ATTENDS to the recent authoritative override; a pure
SSM has only a fixed-size recurrent state with no look-back. We confirm by sweeping erratum strength
(weak -> very strong/imperative) across one model per architecture class and reporting which
phrasings recover the oracle 'deny'. Run: MECH_ATTN=sdpa python esys/erratum_arch_strength.py
"""
import argparse, os, sys, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

POLICY = ("You are a retail support agent. Binding policy: an order may be CANCELLED only if its "
          "order_status is exactly 'pending'. If the status is 'processed' or 'delivered', the order "
          "CANNOT be cancelled and you must DENY the request.")
Q = ("\n\nThe customer asks to cancel order #W123. Per the policy and the order's CURRENT status, "
     "answer with exactly one word — 'cancel' or 'deny'.")
PHRASINGS = {
    "none(stale)": "",
    "weak": "\n\n[STATE UPDATE] order_status is now 'processed'.",
    "standard": "\n\n[STATE UPDATE] The order_status has changed to 'processed'; this overrides any earlier value AND any earlier conclusion.",
    "imperative": "\n\nIMPORTANT CORRECTION: IGNORE the order_status stated above. The order_status is NOW 'processed', NOT 'pending'. Per policy a processed order CANNOT be cancelled — you MUST deny.",
    "restate_rule": "\n\nUPDATE: the order_status is now 'processed'. Recall the rule: only 'pending' orders may be cancelled; 'processed' orders must be denied. Therefore the correct action is to deny.",
}
MODELS = [
    ("Qwen/Qwen3-8B", "attention (GQA)"),
    ("tiiuae/Falcon-H1-1.5B-Instruct", "hybrid attn+Mamba2"),
    ("tiiuae/falcon-mamba-7b-instruct", "pure Mamba / no KV"),
]


def fw(t):
    t = t.strip().lower()
    return "cancel" if "cancel" in t[:24] else ("deny" if ("deny" in t[:24] or "cannot" in t[:24]) else "?")


@torch.no_grad()
def decide(model, tok, content):
    msgs = [{"role": "user", "content": content}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = content + "\nAnswer:"
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id or 0)
    return fw(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))


def main():
    impl = os.environ.get("MECH_ATTN", "sdpa")
    summary = {}
    for mid, arch in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
            try:
                model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda",
                                                             attn_implementation=impl, trust_remote_code=True).eval()
            except (ValueError, KeyError):
                model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda",
                                                             trust_remote_code=True).eval()
        except Exception as e:
            print(f"[skip {mid}: {type(e).__name__}]", flush=True); continue
        base = f"{POLICY}\n\nThe order #W123 current order_status is: pending.{{erratum}}{Q}"
        oracle = decide(model, tok, f"{POLICY}\n\nThe order #W123 current order_status is: processed.{Q}")
        row = {"arch": arch, "oracle(processed)": oracle}
        print(f"\n=== {arch}: {mid} | oracle(processed)={oracle} (target for erratum) ===")
        for name, ph in PHRASINGS.items():
            d = decide(model, tok, base.replace("{erratum}", ph))
            recovered = (d == oracle) and name != "none(stale)"
            row[name] = d
            print(f"  {name:14s} -> {d:7s} {'(recovers)' if recovered else ''}", flush=True)
        summary[mid] = row
        del model; torch.cuda.empty_cache()
    json.dump(summary, open(os.path.join(os.path.dirname(__file__), "..", "results",
              "erratum_arch_strength.json"), "w"), indent=2)
    print("\nERRATUM_ARCH_STRENGTH_DONE")


if __name__ == "__main__":
    main()
