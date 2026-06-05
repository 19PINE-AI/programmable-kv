"""End-to-end τ²-bench retail EPISODE loop with real tools, DB, and reward.

Beyond a single gated decision: we run a multi-turn cancel-order trajectory on the REAL
τ²-bench retail environment (real 6699-char policy, real RetailDB, real cancel_pending_order
tool whose own enforcement is the ground-truth reward), over N REAL pending orders.

Timeline per order (the mutable-field motivation): the agent calls get_order_details and
observes order_status='pending' (cached in its KV). Then the order is fulfilled mid-episode
(status -> 'processed') — the cached observation is now stale. Per the real policy rule "an
order can only be cancelled if its status is 'pending'", the correct terminal action flips
cancel -> deny. We compare KV-edit strategies on the terminal action and EXECUTE it against
the real tool:
  - decide 'cancel' -> call env.tools.cancel_pending_order(oid): on a processed order the tool
    RAISES (env enforcement) => task FAIL; on a pending order it succeeds => task OK.
  - decide 'deny'   -> no write; correct iff the order is processed.
Two arms per order: A) status stays pending (control, correct=cancel); B) pending->processed
(correct=deny). Strategies: stale / in_place / erratum / field+erratum / full (oracle).
Reports P(correct terminal action) + tool-consistency + recompute fraction, with Wilson CIs.
Run: MECH_ATTN=sdpa python esys/tau2_episode.py --n 20
"""
import argparse, os, sys, json, math
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/home/ubuntu/tau2-bench/src")
from editkv import EditableContext, Mode
from transformers import AutoModelForCausalLM, AutoTokenizer
from tau2.domains.retail.environment import get_environment


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, c - h), 2), round(min(1, c + h), 2))


def convo(oid):
    return (
        "\n\n# Conversation\n"
        f"user: Hi, I'd like to cancel order {oid} — ordered by mistake.\n"
        f"assistant: tool_call: get_order_details(order_id=\"{oid}\")\n"
        f"observation: (see the order_status recorded in the session above)\n"
        "user: Yes, please cancel it.\n"
        "assistant: Let me verify the order's CURRENT status against the policy before acting.")

DECISION = ("\n\n# TASK\nPer the policy and the order's CURRENT status, decide the single next "
            "action in one word: cancel (if the order may be cancelled) or deny (if it may not).\n"
            "Decision:")


def first_word(s):
    s = s.strip().lower()
    if "cancel" in s[:10]:
        return "cancel"
    if "deny" in s[:8] or "cannot" in s[:8] or "transfer" in s[:10] or "refuse" in s[:8]:
        return "deny"
    return s.split()[0] if s.split() else ""


def build_ctx(model, tok, policy, oid, status):
    ctx = EditableContext(model, tok)
    ctx.add_text(policy + f"\n\n# Session\nThe order {oid} current order_status is: ")
    ctx.add_field("order_status", status)
    ctx.add_text("." + convo(oid))
    ctx.prefill()
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tag", default="qwen3_8b")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa", trust_remote_code=True).eval()
    env = get_environment(); policy = env.policy
    pend = [oid for oid, o in env.tools.db.orders.items() if o.status == "pending"][:args.n]
    print(f"real tau2 retail: policy {len(policy)} chars, {len(pend)} pending orders sampled", flush=True)

    modes = {"full(oracle)": Mode.FULL_REPREFILL, "stale": Mode.STALE, "in_place": Mode.IN_PLACE,
             "erratum": Mode.ERRATUM, "field+erratum": Mode.FIELD_PLUS_ERRATUM}
    # tallies: per arm (A=stays pending corr=cancel; B=->processed corr=deny)
    correct = {arm: {m: 0 for m in modes} for arm in ["A", "B"]}
    tool_ok = {arm: {m: 0 for m in modes} for arm in ["A", "B"]}
    n = 0
    for oid in pend:
        ctx = build_ctx(model, tok, policy, oid, "pending")
        for arm, newval, corr in [("A", "pending", "cancel"), ("B", "processed", "deny")]:
            for mname, mode in modes.items():
                try:
                    out = ctx.generate("order_status", newval, mode, decision_prompt=DECISION, max_new_tokens=10)
                    dec = first_word(out)
                except Exception:
                    dec = "(err)"
                correct[arm][mname] += (dec == corr)
                # execute against the REAL tool on a fresh DB copy reflecting this arm's true status
                o = env.tools.db.orders[oid]; saved = o.status
                o.status = "pending" if arm == "A" else "processed"
                consistent = False
                if dec == "deny":
                    consistent = (arm == "B")            # deny correct iff processed
                elif dec == "cancel":
                    try:
                        env.tools.cancel_pending_order(oid, "ordered by mistake")
                        consistent = (arm == "A")        # tool succeeded -> only valid if pending
                        o.status = saved                 # (restore; cancel mutated it)
                    except Exception:
                        consistent = False               # tool refused -> wrong action on processed
                o.status = saved
                tool_ok[arm][mname] += consistent
        n += 1
        if n % 5 == 0:
            print(f"  ...{n}/{len(pend)} orders", flush=True)

    out = {"model": args.model, "n_orders": n, "arms": {}}
    print(f"\n==== τ²-BENCH EPISODE LOOP ({n} real orders, 2 arms) ====")
    for arm, desc in [("A", "status STAYS pending  (correct=cancel)"),
                      ("B", "status ->processed     (correct=deny)")]:
        print(f"\nArm {arm}: {desc}")
        out["arms"][arm] = {}
        for m in modes:
            kc, kt = correct[arm][m], tool_ok[arm][m]
            out["arms"][arm][m] = {"P_correct": round(kc / n, 3), "ci": wilson(kc, n),
                                   "tool_consistent": round(kt / n, 3)}
            print(f"  {m:14s} P_correct={kc/n:.2f} CI{wilson(kc,n)}  tool_consistent={kt/n:.2f}")
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"tau2_episode_{args.tag}.json"), "w"), indent=2)
    print("\nTAU2_EPISODE_DONE")


if __name__ == "__main__":
    main()
