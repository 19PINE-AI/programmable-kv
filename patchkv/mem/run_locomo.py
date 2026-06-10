"""LoCoMo external validity: transplant REAL conversational memory vs full recompute.

For each LoCoMo conversation we build the multi-session dialogue as the "memory", precompute it
in isolation, RoPE-reposition + splice it before each QA question, and compare to a full
recompute of [sys][conversation][question]. Endpoints per question:
  * correct_full / correct_transplant : gold answer recalled (normalized containment + token-F1)
  * answer_agree : transplant's generated answer == full's (exact, normalized)
This is the MemArt-comparable setting (LoCoMo QA) with our transplant; it tests whether
precompiling real memory preserves question-answering, not just synthetic gated decisions.
Writes results/locomo_<tag>.jsonl.
"""
import os, sys, json, argparse, time, re, string
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "esys"))
from composable_kv import (load_lm, prefill, precompute_chunk, repositioned_chunk_cache,
                           cache_concat, cache_slice, forward_suffix, cos_sin, rotate_half)
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

SYS = ("You are a helpful assistant with access to a long conversation history between two "
       "people. Answer questions about it accurately and concisely.")


@torch.no_grad()
def prefill_lite(model, ids):
    """Build the KV cache WITHOUT materializing all-position logits (logits_to_keep=1).
    At 24k tokens the full logit tensor is 7-15GB of pure waste — we only need the cache."""
    out = model(input_ids=ids.to("cuda"), past_key_values=DynamicCache(), use_cache=True, logits_to_keep=1)
    return out.past_key_values


def norm(s):
    s = s.lower().strip()
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def correct(ans, gold):
    a, g = norm(ans), norm(str(gold))
    if not g:
        return 0
    if g in a or a in g:
        return 1
    # token-F1 >= 0.5
    at, gt = set(a.split()), set(g.split())
    if not at or not gt:
        return 0
    inter = len(at & gt)
    p = inter / len(at); r = inter / len(gt)
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return int(f1 >= 0.5)


def build_memory(conv):
    a = conv.get("speaker_a", "A"); b = conv.get("speaker_b", "B")
    lines = ["# CONVERSATION HISTORY\n"]
    i = 1
    while f"session_{i}" in conv:
        dt = conv.get(f"session_{i}_date_time", "")
        lines.append(f"\n## Session {i} ({dt})")
        for turn in conv[f"session_{i}"]:
            spk = turn.get("speaker", "")
            txt = turn.get("text", "")
            lines.append(f"{spk}: {txt}")
        i += 1
    lines.append("\n# END CONVERSATION\n")
    return "\n".join(lines)


@torch.no_grad()
def gen_answer(model, tok, cache, last_id, pos, max_new=64):
    # operate on `cache` in place (caller discards it); no clone -> half the memory.
    # returns (text, first_token_logits) — the logits are the robust faithfulness probe.
    cur = last_id; p = pos; gen = []; eos = tok.eos_token_id; first_logits = None
    for _ in range(max_new):
        out = model(input_ids=torch.tensor([[cur]], device="cuda"), past_key_values=cache,
                    cache_position=torch.tensor([p], device="cuda"), use_cache=True)
        lg = out.logits[0, -1].float()
        if first_logits is None:
            first_logits = lg.clone()
        cur = int(lg.argmax()); gen.append(cur); p += 1
        if cur == eos or tok.decode(gen[-6:]).count("\n") >= 2:
            break
    return tok.decode(gen, skip_special_tokens=True).strip(), first_logits


def build_ids(tok, mem, question):
    body = (f"{SYS}\n\n{mem}\n\nAnswer the question in as few words as possible (a name, date, "
            f"or short phrase), based only on the conversation.\nQuestion: {question}\nShort answer:")
    try:  # Qwen3 etc.: disable thinking for short factual QA
        full = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        full = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = torch.tensor([enc["input_ids"]]); offs = enc["offset_mapping"]
    sc = full.find(mem); ec = sc + len(mem)
    a = next(i for i, (lo, hi) in enumerate(offs) if lo <= sc < hi)
    b = next((i for i, (lo, hi) in enumerate(offs) if lo >= ec), len(offs))
    return ids, a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--convs", type=int, default=10)
    ap.add_argument("--q_per_conv", type=int, default=20)
    ap.add_argument("--seam", type=int, default=4)
    ap.add_argument("--max_mem_tok", type=int, default=12000)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # flash attention is O(L) memory — essential for the long (up to 24k-token) conversation
    # memory; sdpa falls back to the O(L^2) math backend and OOMs at this length.
    try:
        model = load_lm(args.model, attn="flash_attention_2")
    except Exception as e:
        print(f"flash_attention_2 unavailable ({repr(e)[:80]}); falling back to sdpa", flush=True)
        model = load_lm(args.model, attn="sdpa")
    data = json.load(open(os.path.join(os.path.dirname(__file__), "results", "locomo10.json")))
    path = os.path.join(os.path.dirname(__file__), "results", f"locomo_{tag}.jsonl")
    f = open(path, "w"); t0 = time.time()
    for ci, conv in enumerate(data[:args.convs]):
        mem = build_memory(conv["conversation"])
        memtok = tok(mem, add_special_tokens=False)["input_ids"]
        if len(memtok) > args.max_mem_tok:
            mem = tok.decode(memtok[:args.max_mem_tok])
        qas = [q for q in conv["qa"] if str(q.get("category")) != "5" and q.get("answer")][:args.q_per_conv]
        # precompute the conversation memory chunk ONCE per conversation
        mem_alone = None
        for q in qas:
            ids, a, b = build_ids(tok, mem, q["question"])
            L = ids.shape[1]; nb = b - a
            last = int(ids[0, L - 1])
            # FULL recompute: prefill [:L-1], decode from L-1 (no clone needed)
            fc = prefill_lite(model, ids[:, :L - 1])
            af, fl = gen_answer(model, tok, fc, last, L - 1)
            del fc; torch.cuda.empty_cache()
            # TRANSPLANT: precompute conv in isolation (once/conv), reposition to [a,b), splice, seam-repair
            if mem_alone is None:
                mem_alone = prefill_lite(model, ids[:, a:b])
            pre = prefill_lite(model, ids[:, :a])
            K = min(args.seam, nb)
            if K > 0:
                pre = forward_suffix(model, pre, ids[:, a:a + K], a).past_key_values
                if K < nb:
                    cs, ss = cos_sin(model, list(range(K, nb))); ct, st = cos_sin(model, list(range(a + K, a + nb)))
                    tail = DynamicCache()
                    for i, l in enumerate(mem_alone.layers):
                        kk = l.keys[:, :, K:nb, :].float(); raw = kk * cs - rotate_half(kk) * ss
                        tail.update((raw * ct + rotate_half(raw) * st).to(l.keys.dtype), l.values[:, :, K:nb, :], i)
                    cache = cache_concat(cache_slice(pre, 0, a + K), tail); del tail
                else:
                    cache = cache_slice(pre, 0, a + nb)
            else:
                cache = cache_concat(pre, repositioned_chunk_cache(model, mem_alone, nb, a))
            cache = forward_suffix(model, cache, ids[:, b:L - 1], b).past_key_values
            at, tl = gen_answer(model, tok, cache, last, L - 1)
            del pre, cache; torch.cuda.empty_cache()
            import torch.nn.functional as Fn
            rec = dict(model=args.model, conv=ci, category=str(q.get("category")), L_mem=int(nb), L_total=int(L),
                       gold=str(q["answer"]), ans_full=af, ans_transplant=at,
                       correct_full=correct(af, q["answer"]), correct_transplant=correct(at, q["answer"]),
                       answer_agree=int(norm(af) == norm(at)),
                       ans_cos=float(Fn.cosine_similarity(fl, tl, 0)),
                       ans_top1_agree=int(int(fl.argmax()) == int(tl.argmax())))
            f.write(json.dumps(rec) + "\n"); f.flush()
        del mem_alone; mem_alone = None; torch.cuda.empty_cache()
        print(f"  conv {ci+1}/{min(args.convs,len(data))} done ({time.time()-t0:.0f}s) mem_tok={nb}", flush=True)
    f.close()
    print(f"LOCOMO_DONE {args.model} -> {path}")


if __name__ == "__main__":
    main()
