"""Phase D: faithful PatchKV update mechanism + baselines.

A real in-place update cannot copy oracle KV; it must RECOMPUTE. We provide:

  recompute_field_inplace : overwrite the field span's KV by forwarding the field
     tokens against the (identical) prefix [0,s). Because every token before the
     field is causally independent of it, this is EXACT (== full-new-prefill KV
     for the field span) -- the H2 free lunch.
  recompute_suffix        : truncate the cache to `start` and re-prefill tokens
     [start, end) against it. For a recency residual this rebuilds the tail while
     attending to the (mostly stale, field-refreshed) base -- the realistic,
     approximate refresh.

PatchKV(recent=K) = old cache -> recompute_field_inplace -> recompute_suffix(last K).
Baselines: full_reprefill (ceiling), stale_reuse (floor), hoist_to_end.
All return a DynamicCache of length `upto` ready for decode, plus a recompute-token
count (the cost metric).
"""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e2"))
import capture
from transformers.cache_utils import DynamicCache
from run_e2 import clone_cache


@torch.no_grad()
def recompute_field_inplace(model, new_ids, cache_old, span, capо=None):
    """Return a cache = old with field span [s,e) overwritten by EXACT recomputed KV."""
    s, e = span
    work = clone_cache(cache_old, cache_old.layers[0].keys.shape[2])  # full length copy
    # forward field tokens against prefix [0,s); capture their post-RoPE K/V
    prefix = clone_cache(cache_old, s)
    capture.enable_capture(to_cpu=False)
    cp = torch.arange(s, e, device="cuda")
    model(input_ids=new_ids[:, s:e].to("cuda"), past_key_values=prefix,
          cache_position=cp, use_cache=True)
    capture.disable_capture()
    cap = capture.get_capture()
    # captured K/V are the FULL post-update tensors (prefix[0,s) + field[s,e)),
    # length e; the field tokens we want are positions [s,e).
    for i in range(len(work.layers)):
        kf = cap[i]["k"]; vf = cap[i]["v"]
        if kf.dim() == 3:  # [Hkv,T,d] -> add batch
            kf = kf[None]; vf = vf[None]
        work.layers[i].keys[:, :, s:e, :] = kf[:, :, s:e, :]
        work.layers[i].values[:, :, s:e, :] = vf[:, :, s:e, :]
    return work, (e - s)


@torch.no_grad()
def recompute_suffix(model, new_ids, cache, start, end):
    """Truncate cache to `start`, re-prefill tokens [start,end) against it.
    Returns the extended cache (length end) and token count recomputed."""
    c = clone_cache(cache, start)
    if end > start:
        cp = torch.arange(start, end, device="cuda")
        model(input_ids=new_ids[:, start:end].to("cuda"), past_key_values=c,
              cache_position=cp, use_cache=True)
    return c, (end - start)


@torch.no_grad()
def patchkv_cache(model, new_ids, cache_old, span, recent, upto):
    """Faithful PatchKV: exact field refresh + recompute last-`recent` tokens.
    `upto` is the cache length needed for decode (T-1)."""
    s, e = span
    work, n_field = recompute_field_inplace(model, new_ids, cache_old, span)
    win_start = max(e, upto - recent) if recent > 0 else upto
    work, n_win = recompute_suffix(model, new_ids, work, win_start, upto)
    # ensure length == upto (recompute_suffix truncated to win_start then extended to upto)
    return work, n_field + n_win


@torch.no_grad()
def full_reprefill_cache(model, new_ids, upto):
    c = DynamicCache()
    cp = torch.arange(0, upto, device="cuda")
    model(input_ids=new_ids[:, :upto].to("cuda"), past_key_values=c,
          cache_position=cp, use_cache=True)
    return c, upto


def stale_cache(cache_old, upto):
    return clone_cache(cache_old, upto), 0
