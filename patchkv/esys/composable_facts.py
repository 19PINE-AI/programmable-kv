"""Composable taxonomy — CONTENT (facts/RAG vs rules) x INSERTION POINT (system-area vs end-as-tool-result).

Rules-as-skills were covered (composable_kv.py). Here we transplant FACT chunks (RAG-style passages)
and measure whether QA over them survives, at two insertion points:
  early : [system][PASSAGE][question]                      (retrieved context near the top)
  late  : [system][question][TOOL RESULT: PASSAGE][answer] (returned at the end of the trajectory)
For each: full recompute vs precompiled-and-transplanted passage; metric = the answer appears in a short
greedy continuation. Run: python esys/composable_facts.py --model unsloth/gemma-2-9b-it
"""
import argparse, os, sys, json
import torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from composable_kv import (prefill, cache_slice, cache_concat, precompute_chunk,
                           repositioned_chunk_cache, forward_suffix)

PAD = "\n".join(f"- background note {i}: routine context, no action required." for i in range(20))
PASSAGES = [
 dict(p=("[RETRIEVED DOCUMENT]\nAcme Corp Q3 2025 report: quarterly revenue was 4.2 million dollars; "
         "the chief executive is Jane Doe; the headquarters is in Austin; the company employs 320 people; "
         "the flagship product is named Nimbus.\n" + PAD),
      q="Question: What is the name of Acme Corp's flagship product? Answer with one word.\nAnswer:", ans="nimbus"),
 dict(p=("[RETRIEVED DOCUMENT]\nPatient chart: blood type is O-negative; the prescribed medication is "
         "metformin; the attending physician is Dr. Alvarez; the next appointment is on Tuesday; the "
         "allergy on file is penicillin.\n" + PAD),
      q="Question: Which medication is prescribed for the patient? Answer with one word.\nAnswer:", ans="metformin"),
 dict(p=("[RETRIEVED DOCUMENT]\nFlight AC219 status: departs gate 14 at 6:40 PM; the aircraft is an "
         "Airbus A320; the destination is Denver; the on-time performance is 92 percent; the captain is "
         "Lee.\n" + PAD),
      q="Question: What is the destination city of flight AC219? Answer with one word.\nAnswer:", ans="denver"),
 dict(p=("[RETRIEVED DOCUMENT]\nKB article 7781: to reset the device, hold the power button for 10 "
         "seconds; the warranty lasts 24 months; the support email is help@x.io; the model code is "
         "RX-9; the firmware channel is stable.\n" + PAD),
      q="Question: How many months does the warranty last? Answer with a number.\nAnswer:", ans="24"),
]
SYS = "You are a precise assistant. Use the retrieved document to answer."


def chat_spans(tok, body, chunk):
    full = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = torch.tensor([enc["input_ids"]]); offs = enc["offset_mapping"]
    sc = full.find(chunk); ec = sc + len(chunk)
    a = next(i for i, (lo, hi) in enumerate(offs) if lo <= sc < hi)
    b = next((i for i, (lo, hi) in enumerate(offs) if lo >= ec), len(offs))
    return ids.to("cuda"), a, b


@torch.no_grad()
def answer_text(model, tok, ids, cache_full=None, n=12):
    """Greedy-decode n tokens continuing from ids (using a prebuilt cache for [0..L-1) if given)."""
    L = ids.shape[1]
    if cache_full is None:
        cache = prefill(model, ids[:, :L - 1])
    else:
        cache = cache_slice(cache_full, 0, L - 1)
    cur = int(ids[0, L - 1]); pos = L - 1; out = []
    eos = tok.eos_token_id
    for _ in range(n):
        o = model(input_ids=torch.tensor([[cur]], device="cuda"), past_key_values=cache,
                  cache_position=torch.tensor([pos], device="cuda"), use_cache=True); pos += 1
        cur = int(o.logits[0, -1].argmax()); out.append(cur)
        if cur == eos:
            break
    return tok.decode(out).lower()


@torch.no_grad()
def transplanted_cache(model, ids, a, b):
    """Build [0..L-1) cache where the chunk [a,b) is precompiled-in-isolation, RoPE-repositioned, spliced."""
    L = ids.shape[1]; chunk = precompute_chunk(model, ids[:, a:b]); n = b - a
    cache = prefill(model, ids[:, :a])
    cache = cache_concat(cache, repositioned_chunk_cache(model, chunk, n, a))
    cache = forward_suffix(model, cache, ids[:, b:L - 1], b).past_key_values
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/gemma-2-9b-it"); ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    quant = any(q in args.model.upper() for q in ("FP8", "-INT8", "GPTQ", "AWQ", "W8A", "W4A"))
    kw = dict(device_map="cuda", attn_implementation="eager", trust_remote_code=True)
    if not quant:
        kw["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()

    res = {pos: {"full": 0, "precompiled": 0, "agree": 0} for pos in ["early", "late"]}
    n = 0
    for pas in PASSAGES:
        # early: passage right after system
        body_e = f"{SYS}\n\n{pas['p']}\n\n{pas['q']}"
        ids_e, ae, be = chat_spans(tok, body_e, pas['p'])
        # late: passage returned at the end as a tool result, just before the answer
        body_l = (f"{SYS}\n\nQuestion pending lookup.\n\n[TOOL RESULT]\n{pas['p']}\n\n{pas['q']}")
        ids_l, al, bl = chat_spans(tok, body_l, pas['p'])
        for posname, ids, a, b in [("early", ids_e, ae, be), ("late", ids_l, al, bl)]:
            full_txt = answer_text(model, tok, ids)
            tr = transplanted_cache(model, ids, a, b)
            pre_txt = answer_text(model, tok, ids, cache_full=tr)
            fc = pas['ans'] in full_txt; pc = pas['ans'] in pre_txt
            res[posname]["full"] += fc; res[posname]["precompiled"] += pc; res[posname]["agree"] += (fc == pc)
            print(f"  {posname:5s} | full={'Y' if fc else 'n'}({full_txt[:14]!r}) precompiled={'Y' if pc else 'n'}({pre_txt[:14]!r})", flush=True)
        n += 1
    out = {"model": args.model, "n": n, "results": res}
    print(f"\n=== FACTS/RAG transplant ({args.model}, n={n}) ===")
    for posname in ["early", "late"]:
        r = res[posname]
        print(f"  [{posname}] full_correct={r['full']}/{n} precompiled_correct={r['precompiled']}/{n} agree={r['agree']}/{n}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"composable_facts_{tag}.json"), "w"), indent=2)
    print("FACTS_DONE")


if __name__ == "__main__":
    main()
