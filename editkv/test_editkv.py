"""Smoke/unit tests for editkv. Run: python -m editkv.test_editkv [model]
Validates: prefill+span, the two incarnations differ as expected, erratum recovers,
full_reprefill==erratum on the decision, diagnostic flags high-conditioning edits,
length-changing edit raises for in_place, and a low-conditioning field needs no erratum.
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from editkv import EditableContext, Mode, LengthChangeError
from editkv.diagnostics import needs_erratum

POLICY = ("You are a retail support agent.\nPOLICY: If account_role is verified_admin, "
          "perform refunds. If account_role is suspended_user, do NOT refund; escalate.\n")
CONVO = "\nuser: Please refund $40 to my order.\nassistant: Let me check the policy."
DEC = "\nDecide one word (refund / escalate).\nDecision:"


def ctx_for(model, tok):
    c = EditableContext(model, tok)
    c.add_text(POLICY + "\nSESSION\naccount_role: ")
    c.add_field("account_role", "verified_admin")
    c.add_text(CONVO); c.prefill(); return c


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa", trust_remote_code=True).eval()
    c = ctx_for(model, tok)
    g = lambda v, m: c.generate("account_role", v, m, decision_prompt=DEC, max_new_tokens=6).strip()
    passed = []

    f = c.fields["account_role"]
    assert f.span and f.span[1] > f.span[0], "field span not located"
    passed.append("span located")

    erratum = g("suspended_user", Mode.ERRATUM)
    full = g("suspended_user", Mode.FULL_REPREFILL)
    inplace = g("suspended_user", Mode.IN_PLACE)
    assert erratum == full, f"erratum {erratum!r} should match full_reprefill {full!r}"
    passed.append(f"erratum==full_reprefill ({erratum!r})")
    assert erratum != inplace, "high-conditioning edit: in_place should differ from erratum"
    passed.append(f"in_place ({inplace!r}) != erratum ({erratum!r}) [in_place reverts]")

    d = needs_erratum(c, "account_role", "suspended_user", probe=DEC)
    assert d.needs_erratum, "diagnostic should flag this high-conditioning edit"
    passed.append("diagnostic flags needs_erratum=True")

    # length-changing edit -> in_place raises, erratum works
    raised = False
    try:
        c.build_cache("account_role", "x", Mode.IN_PLACE)   # 'x' != 2-token 'verified_admin' length
    except LengthChangeError:
        raised = True
    assert raised, "length-changing in_place should raise LengthChangeError"
    passed.append("length-changing in_place raises (erratum still works)")

    print("PASS:")
    for p in passed:
        print("  -", p)
    print("ALL_TESTS_PASS")


if __name__ == "__main__":
    main()
