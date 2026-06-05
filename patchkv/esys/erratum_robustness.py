"""Erratum robustness trio (Qwen3-8B, non-reasoning, 8 diverse tasks).

(a) PHRASING sensitivity: does the trigger wording matter? Compare templates.
(b) OVER-CORRECTION control: add an erratum for an IRRELEVANT (no-op) field change; the
    (correct) decision must NOT spuriously flip.
(c) MULTI-EDIT: append several stacked erratums; the relevant one must still drive the
    decision (no interference).
"""
import argparse, os, sys, json
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from mech_suite import load, ftok, wilson
import diverse_tasks as DT

PHRASINGS = {
    "override_full": "[STATE UPDATE] {label} has changed to {value}; this overrides any earlier value AND any earlier conclusion. Apply the current value.",
    "bare_value":    "[STATE UPDATE] {label} is now {value}.",
    "minimal":       "{label}: {value}",
    "question":      "Note: has {label} changed? Yes — {label} is now {value}. Re-evaluate.",
}
IRRELEVANT_FIELD = "trace_id"   # not referenced by any task policy
IRRELEVANT_OLD, IRRELEVANT_NEW = "tx_0001", "tx_9999"


def chat(tok, content):
    try:
        return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                       add_generation_prompt=True)


@torch.no_grad()
def decide(model, tok, text, toi):
    ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]]).to("cuda")
    lg = model(input_ids=ids, use_cache=False).logits[0, -1].float()
    return "correct" if lg[toi["correct"]] >= lg[toi["stale"]] else "stale"


def build_with_trigger(key, value, trigger):
    """diverse_tasks.build but the update line uses a custom trigger (insert before TASK)."""
    t = DT.TASKS[key]
    base = DT.build(key, value)   # old value, no erratum
    return base.replace("TASK\n", trigger + "\n\nTASK\n", 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    args = ap.parse_args()
    tok, model = load(args.model)
    out = {"model": args.model}

    # (a) phrasing sensitivity
    print("=== (a) phrasing sensitivity (P_correct over 8 tasks) ===")
    out["phrasing"] = {}
    for pname, ptmpl in PHRASINGS.items():
        kc = 0
        for key in DT.TASKS:
            t = DT.TASKS[key]; toi = {"correct": ftok(tok, t["correct"]), "stale": ftok(tok, t["stale"])}
            trig = ptmpl.format(label=t["field"], value=t["vnew"])
            d = decide(model, tok, chat(tok, build_with_trigger(key, t["vold"], trig)), toi)
            kc += (d == "correct")
        out["phrasing"][pname] = {"P_correct": round(kc / 8, 2), "ci": wilson(kc, 8)}
        print(f"  {pname:14s} P_correct={kc}/8 {out['phrasing'][pname]['ci']}")

    # (b) over-correction: irrelevant erratum must not flip the (stale-correct) decision.
    # field stays at OLD value (decision should be the OLD/'stale' action = correct for old);
    # we add an erratum about an UNRELATED field. Decision must stay 'stale' (unchanged).
    print("\n=== (b) over-correction: irrelevant erratum should NOT change the decision ===")
    nflip = 0
    for key in DT.TASKS:
        t = DT.TASKS[key]; toi = {"correct": ftok(tok, t["correct"]), "stale": ftok(tok, t["stale"])}
        base_dec = decide(model, tok, chat(tok, DT.build(key, t["vold"])), toi)   # no erratum
        irr_trig = PHRASINGS["override_full"].format(label=IRRELEVANT_FIELD, value=IRRELEVANT_NEW)
        irr_dec = decide(model, tok, chat(tok, build_with_trigger(key, t["vold"], irr_trig)), toi)
        flipped = (irr_dec != base_dec)
        nflip += flipped
        print(f"  {key:14s} base={base_dec} +irrelevant_erratum={irr_dec} {'FLIPPED!' if flipped else 'stable'}")
    out["over_correction_flips"] = f"{nflip}/8"

    # (c) multi-edit: relevant erratum stacked AFTER an irrelevant one must still drive decision
    print("\n=== (c) multi-edit: [irrelevant erratum] + [relevant erratum] -> still correct? ===")
    kc = 0
    for key in DT.TASKS:
        t = DT.TASKS[key]; toi = {"correct": ftok(tok, t["correct"]), "stale": ftok(tok, t["stale"])}
        irr = PHRASINGS["override_full"].format(label=IRRELEVANT_FIELD, value=IRRELEVANT_NEW)
        rel = PHRASINGS["override_full"].format(label=t["field"], value=t["vnew"])
        d = decide(model, tok, chat(tok, build_with_trigger(key, t["vold"], irr + "\n\n" + rel)), toi)
        kc += (d == "correct")
        print(f"  {key:14s} stacked-> {d}")
    out["multi_edit"] = {"P_correct": round(kc / 8, 2), "ci": wilson(kc, 8)}
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"erratum_robustness_{args.tag}.json"), "w"), indent=2)
    print(f"\n(a) phrasing: {[ (k,v['P_correct']) for k,v in out['phrasing'].items()]}")
    print(f"(b) over-correction flips: {out['over_correction_flips']} (0 is ideal)")
    print(f"(c) multi-edit P_correct: {out['multi_edit']['P_correct']}")
    print("ROBUSTNESS_DONE")


if __name__ == "__main__":
    main()
