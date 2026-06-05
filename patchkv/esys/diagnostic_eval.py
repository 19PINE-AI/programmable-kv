"""Validate the library's per-case `needs_erratum` diagnostic (precision/recall).

The diagnostic claims to predict, FOR A SPECIFIC EDIT, whether the cheap in_place edit is
sufficient or whether you must escalate to field+erratum. We test it against ground truth on
two classes of edit:
  HIGH-conditioning  (8 diverse gating fields, vold->vnew that flips the decision):
      ground-truth NEEDS erratum = (in_place decision != full-reprefill oracle decision)
  LOW-conditioning   (same contexts with the gating field fixed at its correct value, edit an
      IRRELEVANT request_id field): the decision should NOT change, so in_place suffices and
      ground-truth NEEDS erratum should be False (a specificity / no-over-correction test).
We compare the diagnostic's prediction to ground truth and report a confusion matrix +
precision / recall / accuracy. Run: MECH_ATTN=sdpa python esys/diagnostic_eval.py
"""
import argparse, os, sys, json
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from editkv import EditableContext, Mode, LengthChangeError
from editkv.diagnostics import needs_erratum
from transformers import AutoModelForCausalLM, AutoTokenizer
import diverse_tasks as DT

PROBE = "\nDecision:"
RID_OLD, RID_NEW = "req_8f2a1c4d9e", "req_3b7e0a96f1"


def build_high(model, tok, t):
    ctx = EditableContext(model, tok)
    ctx.add_text(t["role"] + "\n\nSESSION CONTEXT\n" + t["field"] + ": ")
    ctx.add_field(t["field"], t["vold"], label=t["field"])
    ctx.add_text("\n\n" + t["rule"] + "\n\nTASK\n" + t["request"])
    ctx.prefill()
    return ctx


def build_low(model, tok, t):
    # gating field FIXED at its NEW (correct) value; the mutable field is an IRRELEVANT request_id
    ctx = EditableContext(model, tok)
    ctx.add_text(t["role"] + "\n\nSESSION CONTEXT\n" + t["field"] + ": " + t["vnew"] + "\nrequest_id: ")
    ctx.add_field("request_id", RID_OLD, label="request_id")
    ctx.add_text("\n\n" + t["rule"] + "\n\nTASK\n" + t["request"])
    ctx.prefill()
    return ctx


@torch.no_grad()
def decide_logit(model, ctx, field, newval, mode, t):
    """Deterministic forced-choice decision: build the edited cache, forward the last token at
    the decode position, compare the logits of the correct-vs-stale action's first token."""
    cache, last, pos = ctx.build_cache(field, newval, mode, decision_prompt=PROBE)
    out = model(input_ids=torch.tensor([[int(last)]], device=ctx.device), past_key_values=cache,
                cache_position=torch.tensor([pos], device=ctx.device), use_cache=True)
    lg = out.logits[0, -1].float()
    tc = ctx.tok(t["correct"], add_special_tokens=False)["input_ids"][0]
    ts = ctx.tok(t["stale"], add_special_tokens=False)["input_ids"][0]
    return "correct" if lg[tc] >= lg[ts] else "stale"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa", trust_remote_code=True).eval()
    MARGINS = [0.0, 0.1, 0.2, 0.3, 0.5]
    rows = []
    for d, t in DT.TASKS.items():
        for cls, field, newval, ctx in [
                ("high", t["field"], t["vnew"], build_high(model, tok, t)),
                ("low", "request_id", RID_NEW, build_low(model, tok, t))]:
            oracle = decide_logit(model, ctx, field, newval, Mode.FULL_REPREFILL, t)
            try:
                ip = decide_logit(model, ctx, field, newval, Mode.IN_PLACE, t)
            except LengthChangeError:
                ip = "len-change"
            gt = (ip != oracle)        # ground truth: does in_place differ from the oracle?
            preds = {m: needs_erratum(ctx, field, newval, probe=PROBE, margin=m).needs_erratum
                     for m in MARGINS}
            rows.append({"task": d, "class": cls, "oracle": oracle, "in_place": ip,
                         "gt_needs": gt, "preds": preds})
            print(f"  {d:14s} {cls:4s} gt={gt} oracle={oracle} in_place={ip} "
                  f"pred@0.2={preds[0.2]}", flush=True)

    n = len(rows)
    print("\n==== DIAGNOSTIC VALIDATION (needs_erratum), margin sweep ====")
    sweep = {}
    for m in MARGINS:
        tp = sum(r["gt_needs"] and r["preds"][m] for r in rows)
        fp = sum((not r["gt_needs"]) and r["preds"][m] for r in rows)
        fn = sum(r["gt_needs"] and (not r["preds"][m]) for r in rows)
        tn = sum((not r["gt_needs"]) and (not r["preds"][m]) for r in rows)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        sweep[m] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(prec, 3),
                    "recall": round(rec, 3), "accuracy": round((tp + tn) / n, 3)}
        print(f"  margin={m}: P={sweep[m]['precision']} R={sweep[m]['recall']} "
              f"acc={sweep[m]['accuracy']} (TP{tp} FP{fp} FN{fn} TN{tn})")
    json.dump({"model": args.model, "n": n, "sweep": sweep, "rows": rows},
              open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"diagnostic_eval_{args.tag}.json"), "w"), indent=2)
    print("DIAGNOSTIC_EVAL_DONE")


if __name__ == "__main__":
    main()
