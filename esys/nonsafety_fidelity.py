"""Non-safety FIDELITY test: does a field edit make the output reflect the new value?

Distinct from the safety scenarios. Here the field (current_date) is not branched on
by an interpreting rule; it is *used in a computation at the point of decision*. The
return-window rule ("within 30 days of delivery") is generic and value-independent, so
the rule's cached notes are NOT meaning-stale -- only the date value is. We test whether
the edit makes the agent's decision track the new date.

old date = within window  -> correct action process_return
new date = outside window -> correct action decline_return    (the "new correct")
"stale" outcome = process_return (the old answer, i.e. ignored the change)

poison variant adds a stale prior conclusion computed with the OLD date.
"""
import argparse, json, os, sys, re
import torch
sys.path.insert(0, os.path.dirname(__file__))
from ablation_thinking import (load, chat, clone_cache, prefill, recompute_suffix,
                               extract, sample, wilson)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from align import align_pair, _common_prefix_len
import contexts as C

DELIVERY = "2026-05-20"
V_OLD = "2026-06-05"   # 16 days after delivery -> within 30-day window
V_NEW = "2026-12-01"   # >6 months after -> window closed
CORRECT_NEW = "decline_return"
STALE_ANS = "process_return"

TOOLS = ("AVAILABLE TOOLS\n- process_return(order_id)\n- decline_return(order_id, reason)\n"
         "- lookup_order(order_id)\nRespond with exactly one line: tool_call: <name>(<args>)")
RULE = ("RETURN RULE (binding): An order may be returned only if today's date is within "
        "30 days of its delivery date. If more than 30 days have passed since delivery, the "
        "return window is closed and you MUST call decline_return.")


def build(value, n_neutral=30, erratum_value=None, poison=False):
    session = f"SESSION CONTEXT\ncurrent_date: {value}\nchannel: web\nagent_id: agent-7\n"
    policy = "POLICY DOCUMENT (binding)\n" + RULE + "\n" + C._neutral_block(n_neutral)
    conclusion = (f"\nassistant: I checked the dates and order A4471 (delivered {DELIVERY}) is "
                  "within the 30-day window, so a return is allowed." if poison else
                  "\nassistant: Let me check whether the return window is still open.")
    convo = (f"CONVERSATION SO FAR\nuser: I'd like to return order A4471, delivered on "
             f"{DELIVERY}.{conclusion}")
    err = (f"[STATE UPDATE] current_date has changed to {erratum_value}; this overrides any "
           "earlier value AND any earlier conclusion. Recompute the window.\n\n"
           if erratum_value else "")
    decision = (err + "TASK\nDecide the single next tool call. Respond with exactly one line: "
                "tool_call: <name>(<args>)\nNext action:")
    return "\n\n".join(["You are a customer-support agent for an online retailer.",
                        session, policy, TOOLS, convo, decision])


def build_caches(model, tok, n_neutral, thinking, poison):
    t_old = chat(tok, build(V_OLD, n_neutral, poison=poison), thinking)
    t_new = chat(tok, build(V_NEW, n_neutral, poison=poison), thinking)
    t_err = chat(tok, build(V_OLD, n_neutral, erratum_value=V_NEW, poison=poison), thinking)
    oid = torch.tensor([tok(t_old, add_special_tokens=False)["input_ids"]])
    nid = torch.tensor([tok(t_new, add_special_tokens=False)["input_ids"]])
    eid = torch.tensor([tok(t_err, add_special_tokens=False)["input_ids"]])
    al = align_pair(tok, t_old, t_new); a, b = al["field_span"]; upto = oid.shape[1] - 1
    co = prefill(model, oid); cn = prefill(model, nid)
    fcache = clone_cache(co, upto)
    for i in range(len(fcache.layers)):
        fcache.layers[i].keys[:, :, a:b, :] = cn.layers[i].keys[:, :, a:b, :]
        fcache.layers[i].values[:, :, a:b, :] = cn.layers[i].values[:, :, a:b, :]
    p = _common_prefix_len(oid[0].tolist(), eid[0].tolist()); ue = eid.shape[1] - 1
    ework = recompute_suffix(model, eid, co, p, ue)
    return {"oracle_new": (cn, nid[0, upto], upto), "stale_full": (co, oid[0, upto], upto),
            "field_only": (fcache, nid[0, upto], upto), "erratum": (ework, eid[0, ue], ue)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--n_neutral", type=int, default=30)
    args = ap.parse_args()
    tok, model = load(args.model)
    recs = []
    for thinking in [True, False]:
        budget = 1536 if thinking else 96
        for poison in [False, True]:
            ctx = "poison" if poison else "benign"
            caches = build_caches(model, tok, args.n_neutral, thinking, poison)
            print(f"\n=== think={thinking} ctx={ctx} return-window (new-correct={CORRECT_NEW}) n={args.n} ===", flush=True)
            for mname, (cache, last, upto) in caches.items():
                tools = [sample(model, tok, clone_cache(cache, upto), last, upto,
                                9000 + j, thinking, args.temp, budget) for j in range(args.n)]
                ncorr = sum(t == CORRECT_NEW for t in tools)   # tracks the new value
                nstale = sum(t == STALE_ANS for t in tools)    # ignored the change
                cens = sum(t == "(censored)" for t in tools)
                rec = {"thinking": thinking, "context": ctx, "method": mname, "n": args.n,
                       "P_newcorrect": round(ncorr / args.n, 2), "ci_newcorrect": wilson(ncorr, args.n),
                       "P_stale": round(nstale / args.n, 2), "censored": cens,
                       "dist": {t: tools.count(t) for t in set(tools)}}
                recs.append(rec)
                print(f"  {mname:12s} P_newcorrect={rec['P_newcorrect']:.2f} {rec['ci_newcorrect']} "
                      f"P_stale={rec['P_stale']:.2f} cens={cens}", flush=True)
    json.dump(recs, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"nonsafety_{args.tag}.json"), "w"), indent=2)
    print("\nNONSAFETY_DONE", flush=True)


if __name__ == "__main__":
    main()
