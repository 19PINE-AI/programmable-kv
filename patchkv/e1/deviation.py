"""Deviation metrics for E1.

Given per-layer captured Q/K/V from an OLD forward and a NEW (field-flipped)
forward over length-aligned token sequences, compute:

  * KV deviation  -- cosine distance and relative L2 of K and V, per token, per
    layer (the raw "blast radius" of the cache entries themselves).
  * Attention-output deviation -- the CacheBlend-style metric: holding the query
    at its oracle (NEW) value, how much does a token's attention *output* change
    when downstream K/V is left STALE (= OLD) while only the field span is
    refreshed to NEW. This is the quantity that actually propagates.

Everything is computed per (layer, token); we aggregate over heads with the mean.
"""
import torch
import torch.nn.functional as F


def _repeat_kv(x, n_rep):
    # x: [Hkv, T, d] -> [Hkv*n_rep, T, d]
    if n_rep == 1:
        return x
    Hkv, T, d = x.shape
    return x[:, None, :, :].expand(Hkv, n_rep, T, d).reshape(Hkv * n_rep, T, d)


@torch.no_grad()
def kv_deviation(k_old, k_new, v_old, v_new):
    """Per-(layer-aggregated later) per-token cosine distance + relative L2.

    Inputs are single-layer tensors [Hkv, T, d]. Returns dict of [T] tensors,
    aggregated over kv heads with the mean.
    """
    def per_token(a, b):
        # a,b: [H,T,d]
        cos = F.cosine_similarity(a.float(), b.float(), dim=-1)          # [H,T]
        cos_dist = (1.0 - cos).mean(0)                                   # [T]
        l2 = (a.float() - b.float()).norm(dim=-1)                        # [H,T]
        denom = b.float().norm(dim=-1).clamp_min(1e-8)
        rel_l2 = (l2 / denom).mean(0)                                    # [T]
        return cos_dist, rel_l2

    k_cos, k_rl2 = per_token(k_old, k_new)
    v_cos, v_rl2 = per_token(v_old, v_new)
    return {"k_cos": k_cos, "k_rel_l2": k_rl2, "v_cos": v_cos, "v_rel_l2": v_rl2}


@torch.no_grad()
def attention_output_deviation(q_new, k_new, v_new, k_old, v_old, field_span,
                               scaling, device="cuda", chunk=512):
    """Per-token relative deviation of the attention output under leave-stale.

    q_new,k_new,v_new : NEW-forward captures for one layer. q:[Hq,T,d] k/v:[Hkv,T,d]
    k_old,v_old       : OLD-forward captures for the same layer.
    field_span        : (start, end) token indices that ARE refreshed to NEW.

    Patched cache = OLD everywhere except field span = NEW. Oracle = all NEW.
    Returns [T] tensor: ||o_oracle[t] - o_patched[t]|| / ||o_oracle[t]||,
    mean over query heads.
    """
    q = q_new.to(device).float()
    Hq, T, d = q.shape
    Hkv = k_new.shape[0]
    n_rep = Hq // Hkv

    kN = _repeat_kv(k_new.to(device).float(), n_rep)   # [Hq,T,d]
    vN = _repeat_kv(v_new.to(device).float(), n_rep)
    # build patched K/V = old, with field span overwritten by new
    s, e = field_span
    kP_kv = k_old.to(device).float().clone()
    vP_kv = v_old.to(device).float().clone()
    kP_kv[:, s:e, :] = k_new.to(device).float()[:, s:e, :]
    vP_kv[:, s:e, :] = v_new.to(device).float()[:, s:e, :]
    kP = _repeat_kv(kP_kv, n_rep)
    vP = _repeat_kv(vP_kv, n_rep)

    out_dev = torch.zeros(T, device=device)
    # process query positions in chunks to bound memory
    for cs in range(0, T, chunk):
        ce = min(cs + chunk, T)
        qc = q[:, cs:ce, :]                                  # [Hq,c,d]
        idx = torch.arange(cs, ce, device=device)
        # causal mask: key j allowed if j <= query position
        # scores [Hq,c,T]
        sN = torch.matmul(qc, kN.transpose(1, 2)) * scaling
        sP = torch.matmul(qc, kP.transpose(1, 2)) * scaling
        mask = (torch.arange(T, device=device)[None, :] <= idx[:, None])  # [c,T]
        neg = torch.finfo(sN.dtype).min
        sN = sN.masked_fill(~mask[None], neg)
        sP = sP.masked_fill(~mask[None], neg)
        aN = torch.softmax(sN, dim=-1)
        aP = torch.softmax(sP, dim=-1)
        oN = torch.matmul(aN, vN)                            # [Hq,c,d]
        oP = torch.matmul(aP, vP)
        num = (oN - oP).norm(dim=-1)                         # [Hq,c]
        den = oN.norm(dim=-1).clamp_min(1e-8)
        out_dev[cs:ce] = (num / den).mean(0)
    return out_dev.cpu()
