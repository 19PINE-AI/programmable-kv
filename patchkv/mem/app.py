"""A working agent with editable + composable user-memory in the KV cache.

MemoryAgent maintains the layout  [system][trajectory][MEMORY][current-query]  where:
  * the system prompt is prefilled once,
  * the trajectory grows turn-by-turn (each delta prefilled once, then cached),
  * the user MEMORY is precompiled ONCE in isolation and RoPE-repositioned to float just
    before the current query every turn (O(L_mem) re-rotation, no re-prefill),
  * a memory change is applied by EITHER recompiling the isolated chunk (O(L_mem), once) OR
    appending a salient erratum into the trajectory stream (O(tokens), composes with prefix
    caching) — the two operations from the paper, on the memory substrate.

This is a usable component, not a benchmark stub: `decide()` returns the model's answer and
the measured time-to-first-token. `run_e5.py` benchmarks it against front/end reprefill.
"""
from __future__ import annotations
import os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "esys"))
from transformers.cache_utils import DynamicCache
from composable_kv import (prefill, precompute_chunk, repositioned_chunk_cache, cache_concat,
                           cache_slice, forward_suffix, cos_sin, rotate_half)


def _ids(tok, text, special=False):
    return torch.tensor([tok(text, add_special_tokens=special)["input_ids"]], device="cuda")


class MemoryAgent:
    def __init__(self, model, tok, system: str, memory_md: str, seam: int = 1):
        self.model = model; self.tok = tok
        self.system = system; self.seam = seam   # boundary-repair tokens (E2: 1 suffices, helps Llama)
        # the running [system][trajectory] cache (prefix-cacheable, grows by deltas)
        sys_ids = _ids(tok, system + "\n\n", special=True)
        self.base = prefill(model, sys_ids)
        self.base_len = sys_ids.shape[1]
        self.base_ids = sys_ids            # exact token stream of [sys][traj] (for a matched oracle)
        self.traj_text = ""
        # precompiled memory chunk (isolation positions 0..N-1), reusable across turns
        self.memory_md = memory_md
        self._mem_ids = _ids(tok, memory_md, special=False)
        self.mem_chunk = precompute_chunk(model, self._mem_ids)
        self.mem_len = self._mem_ids.shape[1]

    # ---- trajectory growth (prefix-cached: only the delta is prefilled) ----
    @torch.no_grad()
    def add_turn(self, text: str):
        delta = _ids(self.tok, text + "\n")
        self.base = forward_suffix(self.model, self.base, delta, self.base_len).past_key_values
        self.base_len += delta.shape[1]
        self.base_ids = torch.cat([self.base_ids, delta], dim=1)
        self.traj_text += text + "\n"

    # ---- memory edit: recompile the chunk OR append a salient erratum ----
    @torch.no_grad()
    def update_memory(self, new_md: str, mode: str = "recompile", label: str = "a setting",
                      value: str = "updated"):
        self.memory_md = new_md
        if mode == "recompile":
            self._mem_ids = _ids(self.tok, new_md, special=False)
            self.mem_chunk = precompute_chunk(self.model, self._mem_ids)
            self.mem_len = self._mem_ids.shape[1]
        elif mode == "erratum":
            err = (f"[MEMORY UPDATE] {label} has changed to {value}; this overrides any earlier "
                   f"value AND any earlier conclusion. Apply the current value.")
            self.add_turn(err)   # erratum enters the cached trajectory stream
        else:
            raise ValueError(mode)

    # ---- exact token stream the proposed cache represents (for a matched full-reprefill oracle) ----
    def exact_ids(self, query: str):
        q = _ids(self.tok, "\n" + query)
        return torch.cat([self.base_ids, self._mem_ids, q], dim=1)

    # ---- decision: splice [base][repositioned memory][query], decode, time TTFT ----
    @torch.no_grad()
    def decide(self, query: str, yes="yes", no="no", cot=False, max_new=400):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        base = cache_slice(self.base, 0, self.base_len)
        K = min(self.seam, self.mem_len)
        if K > 0:
            # recompute the first K memory tokens WITH the real preceding context (boundary repair)
            base = forward_suffix(self.model, base, self._mem_ids[:, :K], self.base_len).past_key_values
            from transformers.cache_utils import DynamicCache
            if K < self.mem_len:
                cs, ss = cos_sin(self.model, list(range(K, self.mem_len)))
                ct, st = cos_sin(self.model, list(range(self.base_len + K, self.base_len + self.mem_len)))
                tail = DynamicCache()
                for i, l in enumerate(self.mem_chunk.layers):
                    kk = l.keys[:, :, K:self.mem_len, :].float()
                    raw = kk * cs - rotate_half(kk) * ss
                    tail.update((raw * ct + rotate_half(raw) * st).to(l.keys.dtype), l.values[:, :, K:self.mem_len, :], i)
                cache = cache_concat(cache_slice(base, 0, self.base_len + K), tail)
            else:
                cache = cache_slice(base, 0, self.base_len + self.mem_len)
        else:
            rep = repositioned_chunk_cache(self.model, self.mem_chunk, self.mem_len, self.base_len)
            cache = cache_concat(base, rep)
        pos = self.base_len + self.mem_len
        q_ids = _ids(self.tok, "\n" + query)
        out = forward_suffix(self.model, cache, q_ids, pos)
        cache = out.past_key_values; pos += q_ids.shape[1]
        first = int(out.logits[0, -1].argmax())
        torch.cuda.synchronize(); ttft = (time.perf_counter() - t0) * 1000
        # read the yes/no decision from the first decode logits
        ty = self.tok(yes, add_special_tokens=False)["input_ids"][0]
        tn = self.tok(no, add_special_tokens=False)["input_ids"][0]
        dec = "yes" if out.logits[0, -1, ty] >= out.logits[0, -1, tn] else "no"
        first_logits = out.logits[0, -1].float().clone()
        text = None
        if cot:
            from memkv import generate_from_cache, parse_final
            text = generate_from_cache(self.model, self.tok, cache, first, pos, max_new)
            dec = parse_final(text)
        return dict(decision=dec, ttft_ms=ttft, text=text, first_logits=first_logits)


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, os.path.dirname(__file__))
    from data import make_persona, filler_trajectory
    from composable_kv import load_lm
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen3-4B")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = load_lm(a.model, attn="sdpa")
    p = make_persona(0, 40, 2, gold_yes=True)
    SYS = "You are a careful account-management assistant. Follow the user settings exactly."
    agent = MemoryAgent(model, tok, SYS, p.memory_markdown())
    # a few chit-chat turns
    for t in range(3):
        agent.add_turn(f"User: let's chat about topic {t}.\nAssistant: sure, happy to help.")
    r1 = agent.decide(p.decision_query(True), cot=True)
    print("turn-1 decision:", r1["decision"], "TTFT %.1fms" % r1["ttft_ms"], "gold:", "yes" if p.gold_yes else "no")
    # a tool flips a relevant setting -> memory update via recompile
    flip = p.settings[p.flip_idx]["attr"]
    p2 = p.with_toggle(p.flip_idx, False)
    agent.update_memory(p2.memory_markdown(), mode="recompile")
    r2 = agent.decide(p2.decision_query(True), cot=True)
    print("after edit:", r2["decision"], "TTFT %.1fms" % r2["ttft_ms"], "gold:", "yes" if p2.gold_yes else "no")
    print("APP_SMOKE_OK")
