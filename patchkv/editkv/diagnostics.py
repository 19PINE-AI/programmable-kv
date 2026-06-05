"""editkv.diagnostics — per-case "do I need the erratum?" measurement.

The cheap field-only in-place edit is sufficient for *low-conditioning* fields and for
reasoning models in benign contexts, but reverts to the stale decision for
high-conditioning fields / non-reasoning models / contradictory context. This module
decides, FOR A SPECIFIC EDIT, whether you can use the cheap in-place edit or should pay
the (still cheap) erratum.

Primary test — `needs_erratum`: decode the next decision under the in-place edit and under
the erratum; if they DISAGREE, the field is conditioning the decision through the (stale)
downstream and you need the erratum; if they AGREE, in-place is sufficient. Both caches are
cheap to build, so this is a ~2-short-decode runtime check.

Secondary signal — `blast_radius`: a cheap proxy for how much the edit perturbs the
decision representation (cosine drift of the decode-position logits between stale and
in-place), useful as a fast pre-filter and for logging/telemetry.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from .core import EditableContext, Field, Mode, LengthChangeError


@dataclass
class Diagnosis:
    needs_erratum: bool
    in_place_decision: str
    erratum_decision: str
    stale_decision: str
    agree_in_place_erratum: bool
    in_place_available: bool          # False if the edit changes token length
    logit_drift: float                # cosine drift stale->in_place at decode pos (blast proxy)
    note: str = ""


@torch.no_grad()
def _decode_short(ctx: EditableContext, cache, last, pos, max_new=24, probe: Optional[str] = None):
    # optionally append a decision probe (e.g. the next user turn / a "Decision:" cue)
    if probe:
        pids = ctx.tok(probe, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([[int(last)] + pids], device=ctx.device)
        out = ctx.model(input_ids=ids[:, :-1], past_key_values=cache,
                        cache_position=torch.arange(pos, pos + ids.shape[1] - 1, device=ctx.device),
                        use_cache=True)
        cache = out.past_key_values; last = int(ids[0, -1]); pos = pos + ids.shape[1] - 1
    toks = []; cur = torch.tensor([[int(last)]], device=ctx.device); p = pos
    eos = ctx.tok.eos_token_id
    for _ in range(max_new):
        o = ctx.model(input_ids=cur, past_key_values=cache,
                      cache_position=torch.tensor([p], device=ctx.device), use_cache=True)
        nx = int(o.logits[0, -1].argmax()); toks.append(nx); p += 1
        if nx == eos or "\n" in ctx.tok.decode(toks):
            break
        cur = torch.tensor([[nx]], device=ctx.device)
    return ctx.tok.decode(toks, skip_special_tokens=True).strip()


@torch.no_grad()
def _logit_at(ctx, cache, last, pos):
    o = ctx.model(input_ids=torch.tensor([[int(last)]], device=ctx.device), past_key_values=cache,
                  cache_position=torch.tensor([pos], device=ctx.device), use_cache=True)
    return o.logits[0, -1].float()


@torch.no_grad()
def needs_erratum(ctx: EditableContext, field_name: str, new_value: str,
                  probe: Optional[str] = None, max_new: int = 24) -> Diagnosis:
    f = ctx.fields[field_name]
    Lm1 = ctx._len - 1
    last_ctx = int(ctx._ids[0, Lm1])
    # stale
    stale_dec = _decode_short(ctx, ctx._clone(ctx._cache, Lm1), last_ctx, Lm1, max_new, probe)
    stale_lg = _logit_at(ctx, ctx._clone(ctx._cache, Lm1), last_ctx, Lm1)
    # in-place (may be unavailable for length-changing edits)
    in_place_available = True; ip_dec = ""; drift = float("nan")
    try:
        ipc, iplast, ippos = ctx.build_cache(field_name, new_value, Mode.IN_PLACE)
        ip_dec = _decode_short(ctx, ctx._clone(ipc, ippos), iplast, ippos, max_new, probe)
        ip_lg = _logit_at(ctx, ctx._clone(ipc, ippos), iplast, ippos)
        drift = float(1 - torch.nn.functional.cosine_similarity(stale_lg, ip_lg, dim=0))
    except LengthChangeError:
        in_place_available = False
    # robust reference = FIELD+ERRATUM (refresh the token AND append the override).
    # (Erratum alone can miss in long real-policy contexts, so it is NOT a safe reference.)
    erc, erlast, erpos = ctx.build_cache(field_name, new_value, Mode.FIELD_PLUS_ERRATUM)
    er_dec = _decode_short(ctx, ctx._clone(erc, erpos), erlast, erpos, max_new, probe)

    agree = (ip_dec == er_dec) if in_place_available else False
    needs = (not in_place_available) or (not agree)
    note = ("in-place changes token length -> use field+erratum" if not in_place_available
            else ("in-place matches field+erratum -> in-place sufficient" if agree
                  else "in-place disagrees with field+erratum -> use field+erratum"))
    return Diagnosis(needs_erratum=needs, in_place_decision=ip_dec, erratum_decision=er_dec,
                     stale_decision=stale_dec, agree_in_place_erratum=agree,
                     in_place_available=in_place_available, logit_drift=drift, note=note)


@torch.no_grad()
def blast_radius(ctx: EditableContext, field_name: str, new_value: str) -> float:
    """Cheap fast pre-filter: cosine drift of the decode-position logits between the stale
    cache and the in-place edit. Larger => the field more strongly conditions the decision
    => more likely to need the erratum. (NaN if the edit is length-changing.)"""
    Lm1 = ctx._len - 1
    stale_lg = _logit_at(ctx, ctx._clone(ctx._cache, Lm1), int(ctx._ids[0, Lm1]), Lm1)
    try:
        ipc, iplast, ippos = ctx.build_cache(field_name, new_value, Mode.IN_PLACE)
        ip_lg = _logit_at(ctx, ctx._clone(ipc, ippos), iplast, ippos)
        return float(1 - torch.nn.functional.cosine_similarity(stale_lg, ip_lg, dim=0))
    except LengthChangeError:
        return float("nan")
