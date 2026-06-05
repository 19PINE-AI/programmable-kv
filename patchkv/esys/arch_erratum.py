"""Erratum universality across novel attention/sequence-mixing architectures.

The erratum is *append-only*: it keeps the (stale) field in place, appends a short authoritative
override, and recomputes only the suffix forward through ALL layers. That makes it architecture-
universal — independent of how a layer stores history (full attention, GQA, MLA's compressed
latent KV, sliding-window, a Mamba/SSM recurrent state, or a hybrid of these). The cheap in_place
KV edit, by contrast, is architecture-specific and degrades exactly where the field-conditioned
memory is NOT a directly-editable KV entry: MLA compresses it, hybrid models keep much of it in a
recurrent SSM state, and pure SSM models have NO KV cache to edit at all.

This script runs the behavioral erratum check (stale->old, oracle->new, erratum->matches oracle)
across architectures via plain HF generation (the erratum needs no cache surgery). Run:
  MECH_ATTN=sdpa python esys/arch_erratum.py --model <hf-id> --arch "<label>"
"""
import argparse, os, sys, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

POLICY = ("You are a retail support agent. Binding policy: an order may be CANCELLED only if its "
          "order_status is exactly 'pending'. If the status is 'processed' or 'delivered', the order "
          "CANNOT be cancelled and you must DENY the request.")
ERR = ("\n\n[STATE UPDATE] The order_status has just changed to 'processed'. This overrides any earlier "
       "value AND any earlier conclusion. Apply the current value.")


def user_msg(status, erratum=False):
    return (f"{POLICY}\n\nThe order #W123 current order_status is: {status}." + (ERR if erratum else "") +
            "\n\nThe customer asks to cancel order #W123. Per the policy and the order's CURRENT status, "
            "answer with exactly one word — 'cancel' or 'deny'.")


def first_word(t):
    t = t.strip().lower()
    return "cancel" if "cancel" in t[:24] else ("deny" if ("deny" in t[:24] or "cannot" in t[:24]) else
                                                (t.split()[0] if t.split() else "?"))


@torch.no_grad()
def decide(model, tok, content):
    msgs = [{"role": "user", "content": content}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = content + "\nAnswer:"
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=8, do_sample=False,
                         pad_token_id=tok.eos_token_id or 0)
    gen = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return first_word(gen), gen.strip()[:30]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arch", default="?")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    impl = os.environ.get("MECH_ATTN", "sdpa")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                     attn_implementation=impl, trust_remote_code=True).eval()
    except (ValueError, KeyError):
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                     trust_remote_code=True).eval()
    cases = {"stale (pending)": user_msg("pending"), "oracle (processed)": user_msg("processed"),
             "erratum (pending+update->processed)": user_msg("pending", erratum=True)}
    res, raw = {}, {}
    print(f"=== erratum on {args.arch}: {args.model} ===")
    for label, c in cases.items():
        d, r = decide(model, tok, c)
        res[label] = d; raw[label] = r
        print(f"  {label:38s} -> {d:7s} (raw: {r!r})", flush=True)
    ok = (res["stale (pending)"] == "cancel" and res["oracle (processed)"] == "deny" and
          res["erratum (pending+update->processed)"] == res["oracle (processed)"])
    discriminates = res["stale (pending)"] != res["oracle (processed)"]
    print(f"  oracle flips (stale!=oracle): {discriminates} | erratum matches oracle: "
          f"{res['erratum (pending+update->processed)'] == res['oracle (processed)']} | clean PASS: {ok}")
    json.dump({"model": args.model, "arch": args.arch, "results": res, "discriminates": discriminates,
               "erratum_matches_oracle": res["erratum (pending+update->processed)"] == res["oracle (processed)"],
               "clean_pass": ok}, open(os.path.join(os.path.dirname(__file__), "..", "results",
               f"arch_erratum_{tag}.json"), "w"), indent=2)
    print("ARCH_ERRATUM_DONE")


if __name__ == "__main__":
    main()
