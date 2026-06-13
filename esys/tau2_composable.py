"""H4 — Composable (and unified) KV on the REAL tau2-bench retail policy.

The real retail policy.md (~2k tokens) is exactly the long, reusable chunk an agent re-reads every turn.
We precompile it ONCE and transplant its KV (skip re-prefilling the policy), then make the documented
decision ("an order can be cancelled only if status is pending") on the real policy:
  full       : reprefill [policy][session][convo][decision]               (baseline)
  composed   : precompile+splice the policy, prefill only session+convo+decision
  unified    : composed policy + order_status pending->processed via erratum (compose + edit on one cache)
Metrics: decision correctness (cancel/deny) + composed==full agreement + TTFT saved, over several models.
Run: MECH_ATTN=sdpa python esys/tau2_composable.py --model unsloth/Meta-Llama-3.1-8B-Instruct
"""
import argparse, os, sys, json, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from composable_kv import (load_lm, prefill, cache_slice, cache_concat, precompute_chunk,
                           repositioned_chunk_cache, forward_suffix)

TAU2 = "/home/ubuntu/tau2-bench/data/tau2/domains/retail/policy.md"
CONVO = (
    "\n\n# Conversation\n"
    "user: Hi, I'm Yusuf Rossi, zip 19122. I'd like to cancel order #W2378156, ordered by mistake.\n"
    "assistant: tool_call: find_user_id_by_name_zip(first_name=\"Yusuf\", last_name=\"Rossi\", zip=\"19122\")\n"
    "observation: {\"user_id\":\"yusuf_rossi_9620\"}\nuser: Yes please cancel it.\n"
    "assistant: Let me verify the order status against the policy before acting.")
DECISION = ("\n\n# TASK\nPer the policy and the order's CURRENT status, decide the single next action in "
            "one word: cancel (if the order may be cancelled) or deny (if it may not).\nDecision:")
SESS = "\n\n# Session\nThe order #W2378156 current order_status is: {s}."
ERR = "\n[STATE UPDATE] order_status has changed to processed; this overrides any earlier value AND conclusion.\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct"); ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lm(args.model, attn="sdpa")
    policy = open(TAU2).read()
    tc = tok("cancel", add_special_tokens=False)["input_ids"][0]
    td = tok("deny", add_special_tokens=False)["input_ids"][0]

    def spans(status, update=False):
        body = policy + SESS.format(s=status) + (ERR if update else "") + CONVO + DECISION
        full = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        ids = torch.tensor([enc["input_ids"]]).to("cuda"); offs = enc["offset_mapping"]
        pa = full.find(policy); pb = pa + len(policy)
        a = next(i for i, (lo, hi) in enumerate(offs) if lo <= pa < hi)
        b = next((i for i, (lo, hi) in enumerate(offs) if lo >= pb), len(offs))
        return ids, a, b

    @torch.no_grad()
    def decide(cache, last, pos):
        lg = model(input_ids=torch.tensor([[int(last)]], device="cuda"), past_key_values=cache_slice(cache, 0, pos),
                   cache_position=torch.tensor([pos], device="cuda"), use_cache=True).logits[0, -1].float()
        return "cancel" if lg[tc] >= lg[td] else "deny"

    @torch.no_grad()
    def run(status, update=False, composed=False):
        ids, a, b = spans(status, update); L = ids.shape[1]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        if composed:
            chunk = precompute_chunk(model, ids[:, a:b])
            cache = cache_concat(prefill(model, ids[:, :a]), repositioned_chunk_cache(model, chunk, b - a, a))
            cache = forward_suffix(model, cache, ids[:, b:L - 1], b).past_key_values
        else:
            cache = cache_slice(prefill(model, ids[:, :L - 1]), 0, L - 1)
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1000
        return decide(cache, ids[0, L - 1], L - 1), ms, b - a

    out = {"model": args.model}
    # full vs composed on the real policy, both statuses
    fc_p, _, ptok = run("pending"); cc_p, _, _ = run("pending", composed=True)
    fc_d, ms_full, _ = run("processed"); cc_d, ms_comp, _ = run("processed", composed=True)
    # unified: compose policy + edit pending->processed via erratum (should flip to deny)
    uni, _, _ = run("pending", update=True, composed=True)
    out.update({"policy_tokens": ptok,
                "pending": {"full": fc_p, "composed": cc_p, "correct_is": "cancel"},
                "processed": {"full": fc_d, "composed": cc_d, "correct_is": "deny"},
                "unified_compose+edit(pending->processed)": uni,
                "ttft_full_ms": round(ms_full, 1), "ttft_composed_ms": round(ms_comp, 1),
                "ttft_speedup": round(ms_full / ms_comp, 2)})
    agree = (fc_p == cc_p) + (fc_d == cc_d)
    print(f"=== tau2 REAL-policy composable ({args.model}, policy~{ptok} tok) ===")
    print(f"  pending  : full={fc_p} composed={cc_p} (correct=cancel)")
    print(f"  processed: full={fc_d} composed={cc_d} (correct=deny)")
    print(f"  UNIFIED compose+edit(pending->processed): {uni} (correct=deny)")
    print(f"  composed==full: {agree}/2 | TTFT full={ms_full:.0f}ms composed={ms_comp:.0f}ms -> {ms_full/ms_comp:.2f}x")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"tau2_composable_{tag}.json"), "w"), indent=2)
    print("TAU2_COMPOSABLE_DONE")


if __name__ == "__main__":
    main()
