"""End-to-end-style multi-turn editkv test on the REAL tau2-bench retail policy.

Scenario (a documented tau2 retail rule): "an order can only be cancelled if its status is
'pending'." The mutable field `order_status` is buried early in the long (6699-char) policy
and changes `pending -> processed` mid-conversation, so the correct next action flips
`cancel -> deny`. We run the multi-turn cancel trajectory on the REAL policy and, at the
decision turn, compare the agent's action when order_status is updated via:
  stale / in_place / erratum / field_plus_erratum / full_reprefill (oracle).
Headline: in this long-context real policy, erratum ALONE misses (the stale early field
token still competes) and only field+erratum recovers to the oracle — motivating
field+erratum as the robust default and the diagnostic's field+erratum reference.
Run: MECH_ATTN=sdpa python esys/tau2_editkv.py
"""
import argparse, os, sys, json
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from editkv import EditableContext, Mode
from editkv.diagnostics import needs_erratum
from transformers import AutoModelForCausalLM, AutoTokenizer

TAU2 = "/home/ubuntu/tau2-bench/data/tau2/domains/retail/policy.md"
# tau2 retail policy: "an order can only be cancelled if its status is 'pending'."
# The mutable field is order_status; it changes pending -> processed mid-conversation.
STATUS_OLD = "pending"      # -> cancel allowed
STATUS_NEW = "processed"    # -> cannot cancel -> deny

CONVO = (
    "\n\n# Conversation\n"
    "user: Hi, I'm Yusuf Rossi, zip 19122. I'd like to cancel order #W2378156, ordered by mistake.\n"
    "assistant: tool_call: find_user_id_by_name_zip(first_name=\"Yusuf\", last_name=\"Rossi\", zip=\"19122\")\n"
    "observation: {\"user_id\":\"yusuf_rossi_9620\"}\n"
    "user: Yes please cancel it.\n"
    "assistant: Let me verify the order status against the policy before acting.")
DECISION = ("\n\n# TASK\nPer the policy and the order's CURRENT status, decide the single next "
            "action in one word: cancel (if the order may be cancelled) or deny (if it may not).\n"
            "Decision:")


def build_ctx(model, tok, status):
    policy = open(TAU2).read()
    ctx = EditableContext(model, tok)
    ctx.add_text(policy + "\n\n# Session\nThe order #W2378156 current order_status is: ")
    ctx.add_field("order_status", status)
    ctx.add_text("." + CONVO)
    ctx.prefill()
    return ctx


def first_word(s):
    s = s.strip().lower()
    if "cancel" in s[:10]:
        return "cancel"
    if "deny" in s[:8] or "cannot" in s[:8] or "transfer" in s[:10]:
        return "deny"
    return s.split()[0] if s.split() else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa", trust_remote_code=True).eval()
    # prefill with status=pending (cancel allowed); then status changes to processed.
    ctx = build_ctx(model, tok, STATUS_OLD)
    def g(m):
        try:
            return first_word(ctx.generate("order_status", STATUS_NEW, m, decision_prompt=DECISION, max_new_tokens=12))
        except Exception as e:
            return f"(err:{type(e).__name__})"
    print("real tau2 retail policy:", len(open(TAU2).read()), "chars; field span:", ctx.fields["order_status"].span)
    res = {}
    res["oracle(full_reprefill, status=processed)"] = g(Mode.FULL_REPREFILL)
    res["stale (status still pending)"] = g(Mode.STALE)
    res["in_place -> processed"] = g(Mode.IN_PLACE)
    res["erratum -> processed"] = g(Mode.ERRATUM)
    res["field+erratum -> processed"] = g(Mode.FIELD_PLUS_ERRATUM)
    oracle = res["oracle(full_reprefill, status=processed)"]
    print(f"\n--- decision when order_status changes {STATUS_OLD} -> {STATUS_NEW} (cancel no longer allowed) ---")
    for k, v in res.items():
        tag = " (ORACLE)" if k.startswith("oracle") else (" MATCH" if v == oracle else " MISS")
        print(f"  {k:42s} -> {v}{tag}")
    d = needs_erratum(ctx, "order_status", STATUS_NEW, probe=DECISION)
    print(f"\nDIAGNOSTIC needs_erratum={d.needs_erratum}: in_place='{d.in_place_decision}' erratum='{d.erratum_decision}' | {d.note}")
    json.dump({"model": args.model, "results": res, "oracle": oracle,
               "needs_erratum": d.needs_erratum}, open(os.path.join(os.path.dirname(__file__), "..",
               "results", "tau2_editkv.json"), "w"), indent=2)
    print("TAU2_EDITKV_DONE")


if __name__ == "__main__":
    main()
