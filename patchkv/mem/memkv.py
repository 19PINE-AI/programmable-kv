"""Core memory-KV harness: build early/late layouts, full-recompute vs precompiled
transplant, and all edit methods on the memory chunk — reusing the composable_kv
primitives (RoPE reposition, splice, seam-repair) and the editkv erratum idea.

Layouts (memory always precedes the final query/decode so the decode can read it):
  EARLY : [sys][MEMORY][trajectory][query]      (status quo; trajectory pre-digests memory)
  LATE  : [sys][trajectory][MEMORY][query]      (proposed; query reads fresh memory directly)

Caching:
  full       : recompute the whole sequence (oracle at that placement)
  transplant : MEMORY precompiled in isolation (positions 0..N-1), RoPE-repositioned to its
               slot and spliced; only [sys]/[traj]/[query] are (re)prefilled. seam=K repairs
               the first K memory tokens with the real preceding context.

Edit methods (for a memory fact toggle), late layout:
  stale          : reuse old memory KV, no change
  in_place       : recompute only the toggled fact's tokens in the chunk, splice
  erratum        : append a salient [MEMORY UPDATE ...] note after memory, recompute only it
  recompile_chunk: recompute the whole memory chunk in isolation, re-splice
  selective@K    : recompute toggled fact + K highest-deviation downstream chunk tokens
  full_recompute : oracle (recompute everything with the new memory)

decision(): reads the two decision tokens (yes/no) at temperature 0.
"""
from __future__ import annotations
import os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "esys"))
from transformers.cache_utils import DynamicCache
from composable_kv import (load_lm, prefill, precompute_chunk, repositioned_chunk_cache,
                           cache_concat, cache_slice, forward_suffix, reposition, cos_sin, rotate_half)
from transformers import AutoTokenizer

EARLY, LATE = "early", "late"


# ---------------- tokenization of a laid-out prompt with a locatable memory span ----------------
def build_prompt(tok, sys_txt, mem_txt, traj_txt, query_txt, placement):
    """Return (ids[1,L], mem_lo, mem_hi, query_lo) with the memory span located by offsets.
    The whole thing is wrapped in one user message via the chat template."""
    if placement == EARLY:
        body = f"{sys_txt}\n\n{mem_txt}\n\n{traj_txt}\n\n{query_txt}"
    else:
        body = f"{sys_txt}\n\n{traj_txt}\n\n{mem_txt}\n\n{query_txt}"
    full = tok.apply_chat_template([{"role": "user", "content": body}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = torch.tensor([enc["input_ids"]]); offs = enc["offset_mapping"]
    s_char = full.find(mem_txt); e_char = s_char + len(mem_txt)
    q_char = full.rfind(query_txt)
    mem_lo = next(i for i, (lo, hi) in enumerate(offs) if lo <= s_char < hi)
    mem_hi = next((i for i, (lo, hi) in enumerate(offs) if lo >= e_char), len(offs))
    q_lo = next((i for i, (lo, hi) in enumerate(offs) if lo >= q_char), len(offs))
    return ids, mem_lo, mem_hi, q_lo


@torch.no_grad()
def _decision_logits_from_cache(model, cache, last_id, pos):
    """One-token forward at `pos` over a cache of length `pos`, return logits[vocab]."""
    out = model(input_ids=torch.tensor([[last_id]], device="cuda"),
                past_key_values=cache_slice(cache, 0, pos),
                cache_position=torch.tensor([pos], device="cuda"))
    return out.logits[0, -1].float()


def decide(logits, tok, yes="yes", no="no"):
    ty = tok(yes, add_special_tokens=False)["input_ids"][0]
    tn = tok(no, add_special_tokens=False)["input_ids"][0]
    return "yes" if logits[ty] >= logits[tn] else "no"


def gold_margin(logits, tok, gold, yes="yes", no="no"):
    """Signed logit margin toward the gold answer (logit[gold]-logit[other]); >0 favors gold."""
    ty = tok(yes, add_special_tokens=False)["input_ids"][0]
    tn = tok(no, add_special_tokens=False)["input_ids"][0]
    return float((logits[ty] - logits[tn]) if gold == "yes" else (logits[tn] - logits[ty]))


import re as _re
def parse_final(txt):
    m = _re.findall(r"FINAL:\s*(yes|no)", txt, _re.I)
    if m:
        return m[-1].lower()
    m = _re.findall(r"\b(yes|no)\b", txt, _re.I)
    return m[-1].lower() if m else "none"


@torch.no_grad()
def generate_from_cache(model, tok, cache, last_id, pos, max_new=400, stop_think=True):
    """Greedy-decode from a prepared cache of length `pos` (feeding last_id at `pos`).
    Returns the decoded string. Reuses the cache in place (clones to avoid mutation upstream)."""
    cache = cache_slice(cache, 0, pos)
    cur = last_id; p = pos; gen = []
    eos = tok.eos_token_id
    for _ in range(max_new):
        out = model(input_ids=torch.tensor([[cur]], device="cuda"), past_key_values=cache,
                    cache_position=torch.tensor([p], device="cuda"), use_cache=True)
        cur = int(out.logits[0, -1].argmax()); gen.append(cur); p += 1
        if cur == eos:
            break
        tail = tok.decode(gen[-16:])
        if "FINAL:" in tok.decode(gen[-24:]) and ("yes" in tail.lower() or "no" in tail.lower()):
            break
    return tok.decode(gen, skip_special_tokens=True)


@torch.no_grad()
def cot_decision_full(model, tok, ids, gold=None, max_new=400):
    """CoT decision from a full-recompute prefill: prefill [:-1], then greedy-decode."""
    L = ids.shape[1]
    cache = prefill(model, ids[:, :L - 1])
    txt = generate_from_cache(model, tok, cache, int(ids[0, L - 1]), L - 1, max_new)
    return parse_final(txt), txt


# ---------------- full recompute (oracle at a placement) ----------------
@torch.no_grad()
def run_full(model, tok, ids):
    L = ids.shape[1]
    cache = prefill(model, ids[:, :L - 1])
    logits = _decision_logits_from_cache(model, cache, int(ids[0, L - 1]), L - 1)
    return logits


# ---------------- transplant (memory precompiled in isolation, repositioned, spliced) ----------------
@torch.no_grad()
def run_transplant(model, tok, ids, mem_lo, mem_hi, seam=0, mem_alone=None, return_cache=False):
    """Precompile memory in isolation, reposition to [mem_lo..mem_hi), splice, prefill the rest.
    seam: recompute first `seam` memory tokens with the real preceding context (boundary repair).
    mem_alone: optionally a precomputed isolation cache for the memory chunk (reuse across calls).
    return_cache: if True, also return (cache, last_id, pos=L-1) for CoT decoding."""
    L = ids.shape[1]; nb = mem_hi - mem_lo
    if mem_alone is None:
        mem_alone = precompute_chunk(model, ids[:, mem_lo:mem_hi])
    pre = prefill(model, ids[:, :mem_lo])                      # [sys] (early) or [sys][traj] (late)
    if seam > 0:
        pre = forward_suffix(model, pre, ids[:, mem_lo:mem_lo + seam], mem_lo).past_key_values
        k0 = min(seam, nb)
        if k0 < nb:
            cs, ss = cos_sin(model, list(range(k0, nb)))
            ct, st = cos_sin(model, list(range(mem_lo + k0, mem_lo + nb)))
            tail = DynamicCache()
            for i, l in enumerate(mem_alone.layers):
                kk = l.keys[:, :, k0:nb, :].float()
                raw = kk * cs - rotate_half(kk) * ss
                tail.update((raw * ct + rotate_half(raw) * st).to(l.keys.dtype), l.values[:, :, k0:nb, :], i)
            cache = cache_concat(cache_slice(pre, 0, mem_lo + k0), tail)
        else:
            cache = cache_slice(pre, 0, mem_lo + nb)
    else:
        rep = repositioned_chunk_cache(model, mem_alone, nb, mem_lo)
        cache = cache_concat(pre, rep)
    # prefill the suffix after memory up to L-1
    if mem_hi < L - 1:
        cache = forward_suffix(model, cache, ids[:, mem_hi:L - 1], mem_hi).past_key_values
    logits = _decision_logits_from_cache(model, cache, int(ids[0, L - 1]), L - 1)
    if return_cache:
        return logits, mem_alone, cache, int(ids[0, L - 1]), L - 1
    return logits, mem_alone


# ---------------- editing the memory chunk (late layout) ----------------
@torch.no_grad()
def chunk_inplace_edit(model, tok, mem_alone, old_mem_ids, new_mem_ids):
    """Recompute only the changed token span within the isolated memory chunk; return a new
    isolation cache. Requires same length (a value toggle enabled<->disabled is length-stable
    for our vocabulary)."""
    o = old_mem_ids[0].tolist(); n = new_mem_ids[0].tolist()
    if len(o) != len(n):
        raise ValueError("length change")
    # find changed span
    s = 0
    while s < len(o) and o[s] == n[s]:
        s += 1
    e = len(o)
    while e > s and o[e - 1] == n[e - 1]:
        e -= 1
    if s >= e:
        return mem_alone, 0  # no change
    # recompute [s:e] within isolation (attends to prefix 0..s-1)
    prefix = cache_slice(mem_alone, 0, s)
    out = model(input_ids=new_mem_ids[:, s:e].to("cuda"), past_key_values=prefix,
                cache_position=torch.arange(s, e, device="cuda"), use_cache=True)
    new_cache = DynamicCache()
    for i, l in enumerate(mem_alone.layers):
        k = l.keys.clone(); v = l.values.clone()
        k[:, :, s:e, :] = out.past_key_values.layers[i].keys[:, :, s:e, :]
        v[:, :, s:e, :] = out.past_key_values.layers[i].values[:, :, s:e, :]
        new_cache.update(k, v, i)
    return new_cache, (e - s)


@torch.no_grad()
def chunk_selective_edit(model, tok, mem_alone, old_mem_ids, new_mem_ids, K):
    """field+selective@K on the chunk: recompute the changed span PLUS the K downstream
    chunk tokens with highest key-deviation after an in-place refresh (CacheBlend-style on
    the memoized region). Returns new isolation cache and #recomputed tokens."""
    o = old_mem_ids[0].tolist(); n = new_mem_ids[0].tolist()
    s = 0
    while s < len(o) and o[s] == n[s]:
        s += 1
    e = len(o)
    while e > s and o[e - 1] == n[e - 1]:
        e -= 1
    if s >= e:
        return mem_alone, 0
    # oracle isolation cache for deviation ranking (recompute whole chunk once, offline-style)
    oracle = precompute_chunk(model, new_mem_ids)
    # deviation per downstream token = mean over layers of ||k_oracle - k_stale|| at last layer
    dev = torch.zeros(new_mem_ids.shape[1], device="cuda")
    for l_o, l_s in zip(oracle.layers, mem_alone.layers):
        d = (l_o.keys.float() - l_s.keys.float()).norm(dim=-1).mean(dim=1)[0]  # [N]
        dev += d
    dev[s:e] = -1  # the changed span is always recomputed; exclude from top-K of "downstream"
    downstream = torch.arange(new_mem_ids.shape[1], device="cuda") >= e
    dev = torch.where(downstream, dev, torch.full_like(dev, -1))
    topk = torch.topk(dev, min(K, int(downstream.sum().item()))).indices.tolist() if K > 0 else []
    recompute = sorted(set(list(range(s, e)) + topk))
    new_cache = DynamicCache()
    for i, l in enumerate(mem_alone.layers):
        k = l.keys.clone(); v = l.values.clone()
        for idx in recompute:
            k[:, :, idx, :] = oracle.layers[i].keys[:, :, idx, :]
            v[:, :, idx, :] = oracle.layers[i].values[:, :, idx, :]
        new_cache.update(k, v, i)
    return new_cache, len(recompute)


@torch.no_grad()
def run_edit_late(model, tok, sys_txt, mem_old, mem_new, traj_txt, query_txt, method,
                  yes="yes", no="no", erratum_label="setting", erratum_value="enabled", K=8,
                  return_cache=False):
    """Late layout. Build the decision under an edit `method` after memory changed mem_old->mem_new.
    Returns (logits, recompute_tokens). For the transplant path the memory was precompiled."""
    # locate spans using the OLD memory (the cached one); the chunk content is what differs
    ids_old, mlo, mhi, qlo = build_prompt(tok, sys_txt, mem_old, traj_txt, query_txt, LATE)
    L = ids_old.shape[1]
    mem_old_ids = ids_old[:, mlo:mhi]
    # new memory ids (length may match for value toggles)
    ids_new, mlo2, mhi2, qlo2 = build_prompt(tok, sys_txt, mem_new, traj_txt, query_txt, LATE)

    if method == "full_recompute":
        Ln = ids_new.shape[1]
        cache = prefill(model, ids_new[:, :Ln - 1])
        logits = _decision_logits_from_cache(model, cache, int(ids_new[0, Ln - 1]), Ln - 1)
        if return_cache:
            return logits, Ln, cache, int(ids_new[0, Ln - 1]), Ln - 1
        return logits, Ln

    # base: precompiled OLD memory in isolation
    mem_alone = precompute_chunk(model, mem_old_ids)
    pre = prefill(model, ids_old[:, :mlo])     # [sys][traj] (shared; cacheable)

    recompute = 0
    if method == "stale":
        chunk = mem_alone
    elif method == "recompile_chunk":
        mem_new_ids = ids_new[:, mlo2:mhi2]
        chunk = precompute_chunk(model, mem_new_ids); recompute = mem_new_ids.shape[1]
    elif method == "in_place":
        mem_new_ids = ids_new[:, mlo2:mhi2]
        if mem_new_ids.shape[1] != mem_old_ids.shape[1]:
            chunk = precompute_chunk(model, mem_new_ids); recompute = mem_new_ids.shape[1]
        else:
            chunk, recompute = chunk_inplace_edit(model, tok, mem_alone, mem_old_ids, mem_new_ids)
    elif method.startswith("selective"):
        mem_new_ids = ids_new[:, mlo2:mhi2]
        if mem_new_ids.shape[1] != mem_old_ids.shape[1]:
            chunk = precompute_chunk(model, mem_new_ids); recompute = mem_new_ids.shape[1]
        else:
            chunk, recompute = chunk_selective_edit(model, tok, mem_alone, mem_old_ids, mem_new_ids, K)
    elif method == "erratum":
        chunk = mem_alone  # leave memory stale, append an erratum after it
    else:
        raise ValueError(method)

    nb = mhi - mlo
    rep = repositioned_chunk_cache(model, chunk, nb, mlo)
    cache = cache_concat(pre, rep)
    pos = mhi

    if method == "erratum":
        err = (f"\n[MEMORY UPDATE] {erratum_label} has changed to {erratum_value}; this overrides "
               f"any earlier value AND any earlier conclusion. Apply the current value.\n")
        err_ids = tok(err, add_special_tokens=False)["input_ids"]
        cache = forward_suffix(model, cache, torch.tensor([err_ids], device="cuda"), pos).past_key_values
        pos += len(err_ids); recompute += len(err_ids)

    # prefill the query suffix [mhi:L-1] (use the OLD ids' suffix; query text identical)
    suffix = ids_old[:, mhi:L - 1]
    if suffix.shape[1] > 0:
        cache = forward_suffix(model, cache, suffix, pos).past_key_values
        pos += suffix.shape[1]
    logits = _decision_logits_from_cache(model, cache, int(ids_old[0, L - 1]), pos)
    if return_cache:
        return logits, recompute, cache, int(ids_old[0, L - 1]), pos
    return logits, recompute


if __name__ == "__main__":
    import argparse
    from data import make_persona, filler_trajectory
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = load_lm(a.model, attn="sdpa")
    SYS = "You are a careful account-management assistant. Follow the user's settings exactly."
    p = make_persona(0, 24, 4, gold_yes=True)
    traj = filler_trajectory(4, 0)
    mem = p.memory_markdown(); q = p.decision_query(False)
    for placement in (EARLY, LATE):
        ids, mlo, mhi, qlo = build_prompt(tok, SYS, mem, traj, q, placement)
        fl = run_full(model, tok, ids)
        tl, _ = run_transplant(model, tok, ids, mlo, mhi)
        cos = torch.cosine_similarity(fl, tl, 0).item()
        print(f"{placement:5s} L={ids.shape[1]:4d} mem=[{mlo},{mhi}] full={decide(fl,tok)} "
              f"transplant={decide(tl,tok)} cos={cos:.3f} gold={'yes' if p.gold_yes else 'no'}")
    # edit sanity: flip a relevant setting disabled -> gold flips to no
    p_no = p.with_toggle(p.flip_idx, False)
    flip = p.settings[p.flip_idx]["attr"]
    for method in ["stale", "in_place", "erratum", "recompile_chunk", "selective@8", "full_recompute"]:
        K = 8 if "selective" in method else 0
        lg, rc = run_edit_late(model, tok, SYS, mem, p_no.memory_markdown(), traj, q, method,
                               erratum_label=flip, erratum_value="disabled", K=K)
        print(f"edit {method:16s} -> {decide(lg,tok):3s} (gold now=no) recompute_tok={rc}")
    print("MEMKV_SMOKE_OK")
