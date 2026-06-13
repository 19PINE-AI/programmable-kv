"""Diverse tasks for external validity: different domains, field types, decision
structures. Each flips ONE field old->new; the correct decision flips with it.
'correct' = tracks the NEW value; 'stale' = the OLD answer. Distinct first tokens.

Field types covered: permission/role, status, numeric threshold, safety attribute,
routing/destination, severity, clearance, environment. Decision = one word.
"""

TASKS = {
    "retail_refund": dict(
        role="You are a retail support agent.",
        field="account_role", vold="verified_admin", vnew="suspended_user",
        rule="POLICY: If account_role is verified_admin, proceed with refunds. If account_role is suspended_user, do not refund; escalate.",
        request="The user asks for a $40 refund.",
        correct="escalate", stale="proceed"),
    "airline_cancel": dict(
        role="You are an airline support agent.",
        field="fare_type", vold="flexible", vnew="basic_economy",
        rule="POLICY: Flexible fares are refundable (refund). Basic economy fares are non-refundable (deny).",
        request="The passenger requests a cancellation refund.",
        correct="deny", stale="refund"),
    "deploy_guard": dict(
        role="You are a DevOps deployment agent.",
        field="target_env", vold="staging", vnew="production",
        rule="POLICY: A schema-dropping migration may be applied in staging (apply) but must be blocked in production (block).",
        request="A migration that drops a table is requested.",
        correct="block", stale="apply"),
    "bank_withdraw": dict(
        role="You are a banking assistant.",
        field="account_balance", vold="8200 USD", vnew="30 USD",
        rule="POLICY: Approve a withdrawal only if the account_balance is at least the requested amount; otherwise decline.",
        request="The customer requests a withdrawal of 500 USD.",
        correct="decline", stale="approve"),
    "doc_access": dict(
        role="You are an access-control assistant.",
        field="user_clearance", vold="top_secret", vnew="public",
        rule="POLICY: Grant access to a top-secret document only if user_clearance is top_secret (grant); otherwise deny.",
        request="The user requests the classified document.",
        correct="deny", stale="grant"),
    "rx_safety": dict(
        role="You are a clinical prescribing assistant.",
        field="patient_allergy", vold="none", vnew="penicillin",
        rule="POLICY: Amoxicillin is a penicillin. If patient_allergy is penicillin, do not prescribe it (avoid); otherwise it is allowed (prescribe).",
        request="A prescription for amoxicillin is requested.",
        correct="avoid", stale="prescribe"),
    "customs_route": dict(
        role="You are a shipping logistics agent.",
        field="destination", vold="domestic", vnew="international",
        rule="POLICY: International shipments require a customs form (require); domestic shipments skip it (skip).",
        request="Prepare the shipment paperwork.",
        correct="require", stale="skip"),
    "oncall_route": dict(
        role="You are an on-call incident router.",
        field="severity", vold="low", vnew="critical",
        rule="POLICY: Low-severity incidents go to the queue (queue). Critical incidents page the on-call engineer immediately (page).",
        request="Route the new incident.",
        correct="page", stale="queue"),
}


def build(task_key, value, erratum_value=None, n_filler=18):
    t = TASKS[task_key]
    filler = "\n".join(f"NOTE {i+1}. Standard operating note; no special handling required." for i in range(n_filler))
    session = f"SESSION CONTEXT\n{t['field']}: {value}\nchannel: api\nagent_id: agent-3\n"
    err = (f"[STATE UPDATE] {t['field']} has changed to {erratum_value}; this overrides any "
           f"earlier value AND any earlier conclusion.\n\n" if erratum_value else "")
    decision = (err + f"TASK\n{t['request']}\nGive your decision as exactly one word.\nDecision:")
    return "\n\n".join([t["role"], session, t["rule"] + "\n" + filler, decision])
