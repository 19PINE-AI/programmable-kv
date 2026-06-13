"""H5 — Unified edit+compose in a multi-turn AGENT trajectory (the end-to-end target).

A realistic agent turn-loop on ONE evolving KV cache:
  - COMPOSE: a long POLICY skill is precompiled once and spliced in (never re-prefilled).
  - EDIT: as the world changes across turns, the mutable STATE field is updated by an appended erratum
    (no reprefill of the policy or earlier turns).
  - each turn the agent makes a governed decision (tool action) over the composed policy + current state.
We compare the UNIFIED path (compose-once + append-edits, incremental cache) to a FULL-recompute-every-turn
baseline on (a) per-turn decision agreement / correctness and (b) cumulative TTFT.
Run: python esys/editkv_agent.py --model unsloth/Meta-Llama-3.1-8B-Instruct
"""
import argparse, os, sys, json, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from composable_kv import (load_lm, prefill, cache_slice, cache_concat, precompute_chunk,
                           repositioned_chunk_cache, forward_suffix)

FILLER = "\n".join(f"- SOP note {i}: log the interaction and follow standard procedure." for i in range(300))
POLICY = ("# SKILL: ORDER_OPS_POLICY\n"
          "RULE C1: cancel an order ONLY if order_status is pending; otherwise deny.\n"
          "RULE R1: issue a refund ONLY if order_status is delivered; otherwise escalate.\n"
          "RULE V1: for a VIP customer, always escalate disputes to a senior agent.\n"
          f"{FILLER}\nEnd of ORDER_OPS_POLICY.")
SYS = "You are an order-operations agent. Apply ORDER_OPS_POLICY to each request."

# multi-turn trajectory: (state-edit erratum text, request, action_correct vs action_other)
TURNS = [
 (None, "order_status is pending. The user asks to cancel. One word — cancel or deny.\nDecision:", "cancel", "deny"),
 ("[STATE UPDATE] order_status has changed to shipped; overrides any earlier value AND conclusion.",
  "The user asks again to cancel. One word — cancel or deny.\nDecision:", "deny", "cancel"),
 ("[STATE UPDATE] the customer is now flagged VIP; overrides any earlier value AND conclusion.",
  "The user opens a dispute. One word — escalate or ignore.\nDecision:", "escalate", "ignore"),
 ("[STATE UPDATE] order_status has changed to delivered; overrides any earlier value AND conclusion.",
  "The user asks for a refund. One word — refund or escalate.\nDecision:", "refund", "escalate"),
]


def tids(tok, w):
    return tok(w, add_special_tokens=False)["input_ids"][0]


@torch.no_grad()
def decide_logits(model, cache, last_tok, pos):
    o = model(input_ids=torch.tensor([[int(last_tok)]], device="cuda"), past_key_values=cache_slice(cache, 0, pos),
              cache_position=torch.tensor([pos], device="cuda"), use_cache=True)
    return o.logits[0, -1].float()


def chat_wrap(tok, body):
    return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def fwd_safe(model, cache, seg_ids, start):
    """forward_suffix but a no-op (returns cache) if the segment is empty."""
    if seg_ids.shape[1] == 0:
        return cache
    return forward_suffix(model, cache, seg_ids, start).past_key_values


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct"); ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lm(args.model, attn="sdpa")

    # Build the per-turn chat-formatted token sequences (convo grows with the taken actions).
    fids = []; convo = f"{SYS}\n\n{POLICY}"
    for (edit, req, ca, co) in TURNS:
        convo += ("\n\n" + edit if edit else "") + "\n\n" + req
        fids.append(tok(chat_wrap(tok, convo), add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda"))
        convo += " " + ca  # assume the correct action was taken; trajectory continues
    # POLICY span in turn-0's tokens (for compose)
    full0 = chat_wrap(tok, f"{SYS}\n\n{POLICY}\n\n{TURNS[0][1]}")
    enc = tok(full0, add_special_tokens=False, return_offsets_mapping=True); offs = enc["offset_mapping"]
    pa = full0.find(POLICY); pb = pa + len(POLICY)
    a = next(i for i, (lo, hi) in enumerate(offs) if lo <= pa < hi)
    b = next((i for i, (lo, hi) in enumerate(offs) if lo >= pb), len(offs))

    def lcp(x, y):
        n = min(x.shape[1], y.shape[1]); i = 0
        while i < n and int(x[0, i]) == int(y[0, i]):
            i += 1
        return i

    # ---- UNIFIED: compose POLICY once; each turn reuse the longest cached prefix, prefill only the delta ----
    uni_results = []; t_unified = 0.0; cached_ids = None; cache = None
    policy_chunk = precompute_chunk(model, fids[0][:, a:b])
    for ti, (edit, req, ca, co) in enumerate(TURNS):
        ids = fids[ti]; L = ids.shape[1]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        if ti == 0:
            cache = cache_concat(prefill(model, ids[:, :a]), repositioned_chunk_cache(model, policy_chunk, b - a, a))
            cache = fwd_safe(model, cache, ids[:, b:L - 1], b)
        else:
            k = lcp(cached_ids, ids)                       # reuse shared prefix (policy + earlier turns)
            cache = cache_slice(cache, 0, k)
            cache = fwd_safe(model, cache, ids[:, k:L - 1], k)
        lg = decide_logits(model, cache, ids[0, L - 1], L - 1)
        torch.cuda.synchronize(); t_unified += (time.perf_counter() - t0) * 1000
        # commit the (assumed-correct) decision token so the cache matches the next turn's prefix
        cache = fwd_safe(model, cache, ids[:, L - 1:L], L - 1)
        act_id = tok(" " + ca, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
        cache = fwd_safe(model, cache, act_id, L)
        cached_ids = torch.cat([ids, act_id], dim=1)
        uni_results.append("correct" if lg[tids(tok, ca)] >= lg[tids(tok, co)] else "other")

    # ---- FULL: reprefill the whole context from scratch each turn ----
    full_results = []; t_full = 0.0
    for ti, (edit, req, ca, co) in enumerate(TURNS):
        ids = fids[ti]; L = ids.shape[1]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fc = prefill(model, ids[:, :L - 1])
        lg = decide_logits(model, fc, ids[0, L - 1], L - 1)
        torch.cuda.synchronize(); t_full += (time.perf_counter() - t0) * 1000
        full_results.append("correct" if lg[tids(tok, ca)] >= lg[tids(tok, co)] else "other")

    agree = sum(u == f for u, f in zip(uni_results, full_results))
    uni_correct = uni_results.count("correct"); full_correct = full_results.count("correct")
    out = {"model": args.model, "turns": len(TURNS), "unified_decisions": uni_results, "full_decisions": full_results,
           "agreement": f"{agree}/{len(TURNS)}", "unified_correct": f"{uni_correct}/{len(TURNS)}",
           "full_correct": f"{full_correct}/{len(TURNS)}", "t_unified_ms": round(t_unified, 1),
           "t_full_ms": round(t_full, 1), "speedup": round(t_full / t_unified, 2)}
    print(f"=== UNIFIED edit+compose AGENT ({args.model}) ===")
    print(f"  unified: {uni_results}")
    print(f"  full:    {full_results}")
    print(f"  agreement(unified==full)={agree}/{len(TURNS)} | unified_correct={uni_correct}/{len(TURNS)} full_correct={full_correct}/{len(TURNS)}")
    print(f"  cumulative TTFT: unified={t_unified:.0f}ms full={t_full:.0f}ms -> {t_full/t_unified:.2f}x faster")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"editkv_agent_{tag}.json"), "w"), indent=2)
    print("AGENT_DONE")


if __name__ == "__main__":
    main()
