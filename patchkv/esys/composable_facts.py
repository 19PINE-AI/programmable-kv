"""G1 (rigorous) — Facts/RAG transplant x insertion point, N>=100 + bootstrap CIs.

Transplant a retrieved FACT passage (vs rules) and answer a fact question over it, at two insertion
points: early (system-area retrieved context) vs late (end-of-trajectory tool result). >=100
programmatically-varied passages; QA-accuracy (answer appears in a short greedy continuation) for full
recompute vs precompiled-transplant, with bootstrap 95% CIs. Run: python esys/composable_facts.py --model ...
"""
import argparse, os, sys, json, random
import torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from composable_kv import (load_lm, prefill, cache_slice, cache_concat, precompute_chunk,
                           repositioned_chunk_cache, forward_suffix)

PAD = "\n".join(f"- background note {i}: routine context, no action required." for i in range(18))
# domains: (template with {a}{b}{c}{d}{e}, [(label, pool)]*5, question templates per slot)
CO = ["Nimbus", "Vertex", "Kestrel", "Lumen", "Orbit", "Quasar", "Falcon", "Cobalt", "Pylon", "Zephyr"]
CITY = ["Austin", "Denver", "Oslo", "Cairo", "Lima", "Tokyo", "Madrid", "Dublin", "Perth", "Riga"]
NAME = ["Alvarez", "Okafor", "Tanaka", "Singh", "Romano", "Larsen", "Mensah", "Petrov", "Khan", "Reyes"]
DRUG = ["metformin", "atenolol", "warfarin", "insulin", "ibuprofen", "lisinopril", "naproxen", "heparin"]
NUM = [str(x) for x in [12, 18, 24, 30, 36, 48, 60, 90]]


DOMAINS = [
 ("Acme report: quarterly revenue was {n0} million dollars; the chief executive is {name}; headquarters in {city}; the flagship product is {co}; the team size is {n1}.",
  [("revenue", NUM, "What was the quarterly revenue in millions?"), ("CEO surname", NAME, "What is the surname of the chief executive?"),
   ("HQ city", CITY, "In which city is the headquarters?"), ("product", CO, "What is the flagship product called?")]),
 ("Patient chart: the prescribed medication is {co}... actually {drug}; the attending physician is Dr. {name}; the city of care is {city}; the dosage is {n0} mg; the review is in {n1} days.",
  [("medication", DRUG, "Which medication is prescribed?"), ("physician", NAME, "What is the physician's surname?"),
   ("city", CITY, "In which city is care provided?"), ("dosage", NUM, "What is the dosage in mg?")]),
 ("Flight log: the destination is {city}; the captain is {name}; the aircraft code is {co}; the gate number is {n0}; the duration is {n1} minutes.",
  [("destination", CITY, "What is the destination city?"), ("captain", NAME, "Who is the captain (surname)?"),
   ("aircraft", CO, "What is the aircraft code?"), ("gate", NUM, "What is the gate number?")]),
 ("Asset record: the owner is {name}; the location is {city}; the model is {co}; the age is {n0} months; the rating is {n1}.",
  [("owner", NAME, "Who is the owner (surname)?"), ("location", CITY, "Where is the asset located?"),
   ("model", CO, "What is the model name?"), ("age", NUM, "How many months old is the asset?")]),
]


def gen_items(n=104):
    rng = random.Random(0); items = []
    while len(items) < n:
        tmpl, slots = DOMAINS[len(items) % len(DOMAINS)]
        co, city, name, drug = rng.choice(CO), rng.choice(CITY), rng.choice(NAME), rng.choice(DRUG)
        n0, n1 = rng.sample(NUM, 2)
        passage = "[RETRIEVED DOCUMENT]\n" + tmpl.format(n0=n0, n1=n1, name=name, city=city, co=co, drug=drug) + "\n" + PAD
        label, pool, q = rng.choice(slots)
        ans = {"revenue": n0, "CEO surname": name, "HQ city": city, "product": co, "team size": n1,
               "medication": drug, "physician": name, "city": city, "dosage": n0, "destination": city,
               "captain": name, "aircraft": co, "gate": n0, "owner": name, "location": city, "model": co, "age": n0}[label]
        items.append((passage, f"Question: {q} Answer with one word.\nAnswer:", ans.lower()))
    return items


def boot_ci(xs, B=10000, seed=0):
    n = len(xs)
    if n == 0:
        return [0.0, 0.0]
    r = random.Random(seed); m = sorted(sum(r.choice(xs) for _ in range(n)) / n for _ in range(B))
    return [round(m[int(0.025 * B)], 3), round(m[int(0.975 * B)], 3)]


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
def answer_text(model, tok, ids, cache_full=None, n=10):
    L = ids.shape[1]
    cache = cache_slice(cache_full, 0, L - 1) if cache_full is not None else prefill(model, ids[:, :L - 1])
    cur = int(ids[0, L - 1]); pos = L - 1; out = []; eos = tok.eos_token_id
    for _ in range(n):
        o = model(input_ids=torch.tensor([[cur]], device="cuda"), past_key_values=cache,
                  cache_position=torch.tensor([pos], device="cuda"), use_cache=True); pos += 1
        cur = int(o.logits[0, -1].argmax()); out.append(cur)
        if cur == eos:
            break
    return tok.decode(out).lower()


@torch.no_grad()
def transplanted(model, ids, a, b):
    L = ids.shape[1]; chunk = precompute_chunk(model, ids[:, a:b])
    cache = cache_concat(prefill(model, ids[:, :a]), repositioned_chunk_cache(model, chunk, b - a, a))
    return forward_suffix(model, cache, ids[:, b:L - 1], b).past_key_values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/gemma-2-9b-it"); ap.add_argument("--tag", default=None)
    ap.add_argument("--n", type=int, default=104)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lm(args.model, attn="sdpa")

    items = gen_items(args.n)
    acc = {p: {"full": [], "pre": [], "agree": []} for p in ["early", "late"]}
    for i, (passage, q, ans) in enumerate(items):
        bodies = {"early": f"{SYS}\n\n{passage}\n\n{q}",
                  "late": f"{SYS}\n\nQuery received; performing lookup.\n\n[TOOL RESULT]\n{passage}\n\n{q}"}
        for pos in ["early", "late"]:
            ids, a, b = chat_spans(tok, bodies[pos], passage)
            fc = ans in answer_text(model, tok, ids)
            pc = ans in answer_text(model, tok, ids, cache_full=transplanted(model, ids, a, b))
            acc[pos]["full"].append(int(fc)); acc[pos]["pre"].append(int(pc)); acc[pos]["agree"].append(int(fc == pc))
        if i % 25 == 0:
            print(f"  [{i}/{len(items)}] early full~{sum(acc['early']['full'])/len(acc['early']['full']):.2f} pre~{sum(acc['early']['pre'])/len(acc['early']['pre']):.2f}", flush=True)
    n = len(items)
    out = {"model": args.model, "n": n, "results": {}}
    print(f"\n=== FACTS/RAG transplant ({args.model}, n={n}) ===")
    for pos in ["early", "late"]:
        fa = sum(acc[pos]["full"]) / n; pa = sum(acc[pos]["pre"]) / n; ag = sum(acc[pos]["agree"]) / n
        out["results"][pos] = {"full_acc": round(fa, 3), "full_ci": boot_ci(acc[pos]["full"]),
                               "precompiled_acc": round(pa, 3), "precompiled_ci": boot_ci(acc[pos]["pre"]),
                               "agreement": round(ag, 3), "agreement_ci": boot_ci(acc[pos]["agree"])}
        print(f"  [{pos:5s}] full={fa:.3f} CI{out['results'][pos]['full_ci']} | precompiled={pa:.3f} "
              f"CI{out['results'][pos]['precompiled_ci']} | agree={ag:.3f} CI{out['results'][pos]['agreement_ci']}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"composable_facts_{tag}.json"), "w"), indent=2)
    print("FACTS_DONE")


if __name__ == "__main__":
    main()
