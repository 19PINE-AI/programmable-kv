"""G2 (rigorous) — Agentic ACTUAL tool-calling with a transplanted KV chunk, N>=100 + bootstrap CIs.

The agent must emit a structured function call (name + argument) given a long, reusable TOOL-DEFINITIONS
block; that block is precompiled once and RoPE-transplanted. We score FUNCTIONAL correctness (right
function + right argument) vs full recompute over >=100 programmatically-varied requests, with bootstrap
95% CIs. Run: python esys/composable_agentic.py --model unsloth/gemma-2-9b-it
"""
import argparse, os, sys, json, random
import torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from composable_kv import (load_lm, prefill, cache_slice, cache_concat, precompute_chunk,
                           repositioned_chunk_cache, forward_suffix)

FILLER_TOOLS = "\n".join(f'- noop_{i}(x: string): internal utility {i}; never call for user requests.' for i in range(24))
TOOLDEFS = (
    "# AVAILABLE TOOLS\n"
    '- get_weather(city: string): current weather for a city.\n'
    '- search_flights(origin: string, destination: string): flights between two cities.\n'
    '- cancel_order(order_id: string): cancel a customer order by id.\n'
    '- lookup_account(email: string): fetch the account record for an email.\n'
    '- send_email(to: string, subject: string): send an email.\n'
    '- create_ticket(priority: string): open a support ticket.\n'
    f"{FILLER_TOOLS}\n# END TOOLS")
SYS = ('You are a tool-using agent. For the user request, respond with EXACTLY one JSON tool call '
       '{"name": <tool>, "arguments": {<arg>: <value>}} and nothing else.')
FNS = ["get_weather", "search_flights", "cancel_order", "lookup_account", "send_email", "create_ticket"]

CITIES = ["Paris", "Denver", "Tokyo", "Cairo", "Lima", "Oslo", "Delhi", "Boston", "Madrid", "Seoul",
          "Lagos", "Dublin", "Quito", "Perth", "Milan", "Austin", "Kyoto", "Bogota", "Accra", "Riga"]
NAMES = ["jane", "bob", "amir", "lena", "carlos", "yuki", "noor", "ivan", "sara", "tom"]
PRIOS = ["low", "medium", "high", "urgent", "critical"]


def gen_tasks(n=108):
    rng = random.Random(0); tasks = []
    while len(tasks) < n:
        fn = FNS[len(tasks) % len(FNS)]
        if fn == "get_weather":
            c = rng.choice(CITIES); tasks.append((f"What's the weather in {c} right now?", fn, c.lower()))
        elif fn == "search_flights":
            a, b = rng.sample(CITIES, 2); tasks.append((f"Find flights from {a} to {b}.", fn, b.lower()))
        elif fn == "cancel_order":
            oid = f"{rng.choice('ABCDEFGH')}{rng.randint(1000,9999)}"; tasks.append((f"Cancel my order {oid} please.", fn, oid.lower()))
        elif fn == "lookup_account":
            e = f"{rng.choice(NAMES)}@{rng.choice(['x.io','acme.com','mail.net'])}"; tasks.append((f"Look up the account for {e}.", fn, e))
        elif fn == "send_email":
            e = f"{rng.choice(NAMES)}@{rng.choice(['x.io','acme.com','mail.net'])}"; tasks.append((f"Email {e} about the update.", fn, e))
        else:
            p = rng.choice(PRIOS); tasks.append((f"Open a {p} priority support ticket.", fn, p))
    return tasks


def boot_ci(xs, B=10000, seed=0):
    n = len(xs)
    if n == 0:
        return [0.0, 0.0]
    r = random.Random(seed); m = sorted(sum(r.choice(xs) for _ in range(n)) / n for _ in range(B))
    return [round(m[int(0.025 * B)], 3), round(m[int(0.975 * B)], 3)]


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
    cur = int(ids[0, L - 1]); pos = L - 1; out = []; eos = tok.eos_token_id
    for _ in range(n):
        o = model(input_ids=torch.tensor([[cur]], device="cuda"), past_key_values=cache,
                  cache_position=torch.tensor([pos], device="cuda"), use_cache=True); pos += 1
        cur = int(o.logits[0, -1].argmax()); out.append(cur)
        if cur == eos:
            break
    return tok.decode(out)


def parse_fn(txt):
    t = txt.lower()
    return next((f for f in FNS if f in t), None), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/gemma-2-9b-it"); ap.add_argument("--tag", default=None)
    ap.add_argument("--n", type=int, default=108)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lm(args.model, attn="eager")

    tasks = gen_tasks(args.n)
    full_ok, pre_ok, agree = [], [], []
    for i, (req, fn, arg) in enumerate(tasks):
        body = f"{SYS}\n\n{TOOLDEFS}\n\nUser: {req}\nAssistant:"
        ids, a, b = chat_spans(tok, body, TOOLDEFS)
        chunk = precompute_chunk(model, ids[:, a:b])
        tr = forward_suffix(model, cache_concat(prefill(model, ids[:, :a]), repositioned_chunk_cache(model, chunk, b - a, a)), ids[:, b:ids.shape[1] - 1], b).past_key_values
        ft = gen(model, tok, ids); pt = gen(model, tok, ids, cache_full=tr)
        ffn, fl = parse_fn(ft); pfn, pl = parse_fn(pt)
        fc = (ffn == fn and arg in fl); pc = (pfn == fn and arg in pl)
        full_ok.append(int(fc)); pre_ok.append(int(pc)); agree.append(int(ffn == pfn))
        if i % 20 == 0:
            print(f"  [{i}/{len(tasks)}] full_acc~{sum(full_ok)/len(full_ok):.2f} pre_acc~{sum(pre_ok)/len(pre_ok):.2f}", flush=True)
    n = len(tasks)
    out = {"model": args.model, "n": n,
           "full_acc": round(sum(full_ok) / n, 3), "full_ci": boot_ci(full_ok),
           "precompiled_acc": round(sum(pre_ok) / n, 3), "precompiled_ci": boot_ci(pre_ok),
           "toolcall_agreement": round(sum(agree) / n, 3), "agreement_ci": boot_ci(agree)}
    print(f"\n=== AGENTIC TOOL-CALLING (transplanted tool-defs), {args.model}, n={n} ===")
    print(f"  full fn+arg acc      = {out['full_acc']} CI{out['full_ci']}")
    print(f"  precompiled fn+arg   = {out['precompiled_acc']} CI{out['precompiled_ci']}")
    print(f"  tool-call agreement  = {out['toolcall_agreement']} CI{out['agreement_ci']}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"composable_agentic_{tag}.json"), "w"), indent=2)
    print("AGENTIC_DONE")


if __name__ == "__main__":
    main()
