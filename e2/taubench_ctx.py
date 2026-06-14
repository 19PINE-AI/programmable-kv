"""tau-bench-grounded realistic contexts (Phase B).

Uses the REAL retail policy (wiki.md) as the system prompt (the "rules"), a REAL
order from the tau-bench DB as the status fields surfaced in a tool observation,
and a realistic user request. The mutable field is the ORDER STATUS, which the
policy gates ("an order can only be cancelled if its status is 'pending'"). Note
the structure matches real agents: the gating POLICY precedes the field (so it is
in the causally-exact region), and only the post-observation reasoning is
downstream of the field -- the most favorable case for leave-stale.

Scenarios flip a field and (for high-conditioning ones) the correct action flips.
"""
import os
import json, os

TB = os.environ.get("TAUBENCH_RETAIL", os.path.expanduser("~/tau-bench/tau_bench/envs/retail"))
WIKI = open(os.path.join(TB, "wiki.md")).read()
_ORDERS = json.load(open(os.path.join(TB, "data", "orders.json")))

_TOOLS = """# Tools
- find_user_id_by_email(email)
- get_order_details(order_id)
- get_user_details(user_id)
- cancel_pending_order(order_id, reason)        # reason in {'no longer needed','ordered by mistake'}
- modify_pending_order_items(order_id, item_ids, new_item_ids, payment_method_id)
- return_delivered_order(order_id, item_ids, payment_method_id)
- exchange_delivered_order(order_id, item_ids, new_item_ids, payment_method_id)
- transfer_to_human_agents(summary)
Respond with exactly one line: tool_call: <name>(<args>)"""


def _order_obs(order, status):
    o = dict(order); o = {**o, "status": status}
    items = ", ".join(f'{it["name"]}({it["item_id"]})' for it in o["items"][:3])
    return (f'observation: {{"order_id":"{o["order_id"]}","status":"{status}",'
            f'"items":[{items}],"payment":"{o["payment_history"][0]["payment_method_id"]}"}}')


def _pick_order():
    # a delivered order with >=2 items, so both cancel(pending) and return(delivered) are plausible
    for oid, o in _ORDERS.items():
        if len(o["items"]) >= 2 and "payment_method_id" in o["payment_history"][0]:
            return o
    return next(iter(_ORDERS.values()))


# scenario registry: each flips ONE field; high-conditioning ones flip the action
SCEN = {
    # HIGH: order status gates which action is legal
    "order_status_cancel": dict(
        cls="high", field="order_status", v_old="pending", v_new="delivered",
        request=("I want to cancel my order, it was ordered by mistake."),
        note="pending -> cancel_pending_order; delivered -> cannot cancel (return/deny)"),
    # MEDIUM: payment method type affects refund path wording but action similar
    # LOW: current date (added to header) should not change the action
    "current_date": dict(
        cls="low", field="current_time", v_old="2025-02-10T09:00:00 EST",
        v_new="2025-08-22T17:45:00 EST",
        request=("I want to cancel my order, it was ordered by mistake."),
        note="date change should not affect a cancel decision"),
}


def build(scn_key, value, user_authenticated=True):
    s = SCEN[scn_key]
    order = _pick_order()
    # for the status scenario, value IS the status; for date scenario, status fixed pending
    status = value if s["field"] == "order_status" else "pending"
    header = ""
    if s["field"] == "current_time":
        header = f"Current time: {value}\n"
    sys = f"{header}{WIKI}\n\n{_TOOLS}"
    convo = (
        "# Conversation\n"
        f"user: Hi, {s['request']}\n"
        f"assistant: I can help. Let me pull up the order to check its status.\n"
        f"tool_call: get_order_details(order_id=\"{order['order_id']}\")\n"
        f"{_order_obs(order, status)}\n"
        "assistant: I have the order details. Based on the policy and the order's "
        "current status, the single next tool call is:")
    return sys + "\n\n" + convo


def field_value(scn_key, which):
    s = SCEN[scn_key]
    return s["v_old"] if which == "old" else s["v_new"]
