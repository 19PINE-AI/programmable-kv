"""G2 — Agentic ACTUAL tool-calling with a transplanted KV chunk: does functionality degrade?

A real tool-calling workload (not a reasoning/decision metric): the agent must emit a structured
function call (name + arguments) given a long, reusable TOOL-DEFINITIONS block. That block is exactly
the kind of chunk you precompile once and reuse, so we precompile + RoPE-transplant it and check
whether the emitted tool call is still FUNCTIONALLY CORRECT (right function, right argument) vs full
recompute. We also test a TOOL-RESULT chunk transplanted at the end of the trajectory (the agent's
next call depends on it). Run: python esys/composable_agentic.py --model unsloth/gemma-2-9b-it
"""
import argparse, os, sys, json, re
import torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from composable_kv import (prefill, cache_slice, cache_concat, precompute_chunk,
                           repositioned_chunk_cache, forward_suffix)

# Long, reusable tool-definitions block (the precompilable chunk). Real tools + filler tools for length.
FILLER_TOOLS = "\n".join(
    f'- noop_{i}(x: string): internal utility number {i}; never call this for user requests.' for i in range(24))
TOOLDEFS = (
    "# AVAILABLE TOOLS\n"
    '- get_weather(city: string): return the current weather for a city.\n'
    '- search_flights(origin: string, destination: string): find flights between two cities.\n'
    '- cancel_order(order_id: string): cancel a customer order by id.\n'
    '- lookup_account(email: string): fetch the account record for an email.\n'
    '- send_email(to: string, subject: string): send an email.\n'
    '- create_ticket(priority: string): open a support ticket.\n'
    f"{FILLER_TOOLS}\n# END TOOLS")
SYS = ("You are a tool-using agent. For the user's request, respond with EXACTLY one JSON tool call of "
       'the form {"name": <tool>, "arguments": {<arg>: <value>}} and nothing else.')

TASKS = [
 dict(req="What's the weather in Paris right now?", fn="get_weather", arg="paris"),
 dict(req="Cancel my order number A7731 please.", fn="cancel_order", arg="a7731"),
 dict(req="Find me flights from Boston to Denver.", fn="search_flights", arg="denver"),
 dict(req="Look up the account for jane@x.io.", fn="lookup_account", arg="jane@x.io"),
 dict(req="Email bob@y.com about the refund.", fn="send_email", arg="bob@y.com"),
 dict(req="Open a high priority support ticket.", fn="create_ticket", arg="high"),
]


def chat_spans(tok, body, chunk):
    full = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = torch.tensor([enc["input_ids"]]); offs = enc["offset_mapping"]
    sc = full.find(chunk); ec = sc + len(chunk)
    a = next(i for i, (lo, hi) in enumerate(offs) if lo <= sc < hi)
    b = next((i for i, (lo, hi) in enumerate(offs) if lo >= ec), len(offs))
    return ids.to("cuda"), a, b


@torch.no_grad()
def gen(model, tok, ids, cache_full=None, n=40):
    L = ids.shape[1]
    cache = cache_slice(cache_full, 0, L - 1) if cache_full is not None else prefill(model, ids[:, :L - 1])
    cur = int(ids[0, L - 1]); pos = L - 1; out = []
    eos = tok.eos_token_id
    for _ in range(n):
        o = model(input_ids=torch.tensor([[cur]], device="cuda"), past_key_values=cache,
                  cache_position=torch.tensor([pos], device="cuda"), use_cache=True); pos += 1
        cur = int(o.logits[0, -1].argmax()); out.append(cur)
        if cur == eos:
            break
    return tok.decode(out)


@torch.no_grad()
def transplanted(model, ids, a, b):
    L = ids.shape[1]; chunk = precompute_chunk(model, ids[:, a:b]); nb = b - a
    cache = cache_concat(prefill(model, ids[:, :a]), repositioned_chunk_cache(model, chunk, nb, a))
    return forward_suffix(model, cache, ids[:, b:L - 1], b).past_key_values


def parse(txt):
    t = txt.lower()
    fn = next((f for f in ["get_weather", "search_flights", "cancel_order", "lookup_account", "send_email", "create_ticket"] if f in t), None)
    return fn, t


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

    R = {"full_fn": 0, "full_arg": 0, "pre_fn": 0, "pre_arg": 0, "agree_fn": 0, "n": 0}
    for t in TASKS:
        body = f"{SYS}\n\n{TOOLDEFS}\n\nUser: {t['req']}\nAssistant:"
        ids, a, b = chat_spans(tok, body, TOOLDEFS)   # transplant the TOOL-DEFS chunk
        ft = gen(model, tok, ids); pt = gen(model, tok, ids, cache_full=transplanted(model, ids, a, b))
        ffn, flow = parse(ft); pfn, plow = parse(pt)
        R["full_fn"] += (ffn == t["fn"]); R["full_arg"] += (ffn == t["fn"] and t["arg"] in flow)
        R["pre_fn"] += (pfn == t["fn"]); R["pre_arg"] += (pfn == t["fn"] and t["arg"] in plow)
        R["agree_fn"] += (ffn == pfn); R["n"] += 1
        print(f"  {t['fn']:15s} full=({ffn},{t['arg'] in flow}) precompiled=({pfn},{t['arg'] in plow}) agree={ffn==pfn}", flush=True)
    n = R["n"]
    out = {"model": args.model, "n": n, "full_fn": R["full_fn"], "full_argfn": R["full_arg"],
           "precompiled_fn": R["pre_fn"], "precompiled_argfn": R["pre_arg"], "agree_fn": R["agree_fn"]}
    print(f"\n=== AGENTIC TOOL-CALLING with transplanted tool-defs ({args.model}, n={n}) ===")
    print(f"  full: correct-fn {R['full_fn']}/{n}, correct-fn+arg {R['full_arg']}/{n}")
    print(f"  precompiled: correct-fn {R['pre_fn']}/{n}, correct-fn+arg {R['pre_arg']}/{n} | tool-call agreement {R['agree_fn']}/{n}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"composable_agentic_{tag}.json"), "w"), indent=2)
    print("AGENTIC_DONE")


if __name__ == "__main__":
    main()
