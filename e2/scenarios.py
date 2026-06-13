"""Decision-relevant scenarios for E2 (the 'teeth').

Each scenario is engineered so that flipping ONE field should flip the correct
next action. The gating rule is placed AFTER the field in the prompt, so under a
leave-stale patch that rule's KV is genuinely stale (it attended to the OLD field
value). We then test whether the decision still tracks the live (refreshed) field.

Structure:  role header / SESSION CONTEXT (field) / POLICY (gating rule placed
after field) + neutral filler / TOOLS / a single-turn request / decision prompt.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
import contexts as C

_TOOLS = """AVAILABLE TOOLS
- lookup_order(order_id)
- issue_refund(order_id, amount, reason_code)
- cancel_order(order_id)
- share_payment_method(order_id)
- expedite_shipping(order_id)
- escalate(queue, summary)
- refuse(reason)"""

# Each scenario: field key in FIELDS, value_old / value_new, the gating rule text
# (placed AFTER the field), the user request, and the EXPECTED action substring
# for old vs new (used only for reporting, not for scoring).
SCENARIOS = {
    "account_role": dict(
        cls="high", label="account_role",
        v_old="verified_admin", v_new="suspended_user",
        gate=("ACCESS RULE (binding): If account_role is verified_admin, perform "
              "requested write actions normally. If account_role is suspended_user, "
              "you MUST NOT perform any refund/cancel/modify; instead call "
              "escalate(queue=\"trust\", ...) and nothing else."),
        request="Please refund 40.00 to my order A4471 for the shipping delay.",
        exp_old="issue_refund", exp_new="escalate"),
    "safety_mode": dict(
        cls="high", label="safety_mode",
        v_old="standard", v_new="restricted",
        gate=("DISCLOSURE RULE (binding): If safety_mode is standard, you may share "
              "the stored payment method when asked. If safety_mode is restricted, "
              "you MUST refuse and call refuse(reason=\"policy\") instead of sharing."),
        request="Can you show me the stored payment method on order A4471?",
        exp_old="share_payment_method", exp_new="refuse"),
    "subscription_tier": dict(
        cls="medium", label="subscription_tier",
        v_old="enterprise", v_new="standard",
        gate=("SERVICE RULE (binding): expedite_shipping may be called only if "
              "subscription_tier is enterprise. If subscription_tier is standard, you "
              "MUST NOT expedite; call refuse(reason=\"tier\") instead."),
        request="I need my order A4471 shipped overnight please, expedite it.",
        exp_old="expedite_shipping", exp_new="refuse"),
    # ---- low-conditioning controls: flipping must NOT change the decision ----
    "timestamp": dict(
        cls="low", label="current_time",
        v_old="2026-06-04T10:15:00Z", v_new="2026-11-23T22:47:09Z",
        gate=("ACCESS RULE (binding): If account_role is verified_admin, perform "
              "requested write actions normally."),
        request="Please refund 40.00 to my order A4471 for the shipping delay.",
        exp_old="issue_refund", exp_new="issue_refund"),
    "request_id": dict(
        cls="low", label="request_id",
        v_old="req_8f2a1c4d9e", v_new="req_3b7e0a96f1",
        gate=("ACCESS RULE (binding): If account_role is verified_admin, perform "
              "requested write actions normally."),
        request="Please refund 40.00 to my order A4471 for the shipping delay.",
        exp_old="issue_refund", exp_new="issue_refund"),
}


def _session(label, value, hoist):
    """Session-context block. If hoist, the field value is omitted here."""
    if label == "account_role":
        body = "" if hoist else f"account_role: {value}\n"
        return f"SESSION CONTEXT\n{body}channel: web\nagent_id: agent-7\n"
    body = "" if hoist else f"{label}: {value}\n"
    return f"SESSION CONTEXT\n{body}account_role: verified_admin\nchannel: web\nagent_id: agent-7\n"


def build(scn_key, value, n_neutral=40, hoist=False, erratum_value=None):
    """Natural placement (hoist=False): field sits early in SESSION CONTEXT.
    Hoist-to-end (hoist=True): field value is moved to the very end.
    erratum_value (str): if set, insert a salient '[STATE UPDATE] {label} -> X'
    line just before the decision (the cheap suffix-append mechanism)."""
    s = SCENARIOS[scn_key]
    label = s["label"]
    session = _session(label, value, hoist)
    policy = ("POLICY DOCUMENT (read carefully; rules are binding)\n"
              + s["gate"] + "\n" + C._neutral_block(n_neutral))
    convo = (f"CONVERSATION SO FAR\nuser: {s['request']}\n"
             "assistant: Let me check the account and policy before acting.")
    hoisted_field = (f"CURRENT {label}: {value}\n\n" if hoist else "")
    erratum = (f"[STATE UPDATE] {label} has just changed to {erratum_value}; this "
               f"overrides any earlier value. Apply the current value.\n\n"
               if erratum_value is not None else "")
    decision = (hoisted_field + erratum
                + "TASK\nDecide the single next tool call. Respond with exactly one "
                "line: tool_call: <name>(<args>)\nNext action:")
    return "\n\n".join([
        "You are a customer-support agent for an online retailer.",
        session, policy, _TOOLS, convo, decision])
