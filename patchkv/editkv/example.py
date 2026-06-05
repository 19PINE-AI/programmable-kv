"""editkv example + smoke test. Run: python -m editkv.example [model]"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from editkv import EditableContext, Mode
from editkv.diagnostics import needs_erratum, blast_radius

POLICY = ("You are a retail support agent.\nPOLICY: If account_role is verified_admin, "
          "perform refunds. If account_role is suspended_user, do NOT refund; escalate.\n")
CONVO = "\nCONVERSATION\nuser: Please refund $40 to my order.\nassistant: Let me check the account and policy."
# the decision cue is passed at generate time, so the erratum is injected BEFORE it
DECISION = "\nTASK\nDecide the next action in one word (refund / escalate).\nDecision:"


def build(model, tok):
    ctx = EditableContext(model, tok)
    ctx.add_text(POLICY + "\nSESSION\naccount_role: ")
    ctx.add_field("account_role", "verified_admin")
    ctx.add_text(CONVO)
    ctx.prefill()
    return ctx


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa", trust_remote_code=True).eval()
    ctx = build(model, tok)
    print("field span:", ctx.fields["account_role"].span)
    g = lambda v, m: ctx.generate("account_role", v, m, decision_prompt=DECISION, max_new_tokens=6).strip()
    print("stale  (admin, no edit) :", g("verified_admin", Mode.STALE))
    print("in_place -> suspended   :", g("suspended_user", Mode.IN_PLACE))
    print("erratum  -> suspended   :", g("suspended_user", Mode.ERRATUM))
    print("full_reprefill->suspended:", g("suspended_user", Mode.FULL_REPREFILL))
    d = needs_erratum(ctx, "account_role", "suspended_user", probe=DECISION)
    print(f"\nDIAGNOSTIC: needs_erratum={d.needs_erratum} | in_place='{d.in_place_decision}' "
          f"erratum='{d.erratum_decision}' stale='{d.stale_decision}' | drift={d.logit_drift:.3f}\n  -> {d.note}")
    print("AUTO mode picks         :", g("suspended_user", Mode.AUTO))
    print("EXAMPLE_OK")


if __name__ == "__main__":
    main()
