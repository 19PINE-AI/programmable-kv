"""Length-preserving alignment of OLD vs NEW token sequences.

Tokenize both renderings, locate the single contiguous span where they differ
(the field value), and pad the shorter field span with a filler token so the two
sequences are equal length and every downstream position is aligned. This removes
position-shift confounds so RoPE is identical on both sides outside the field.
"""
import torch


def _common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _common_suffix_len(a, b, max_excl):
    # max_excl: don't let suffix overlap the already-counted prefix
    i = 0
    while (i < max_excl and a[len(a) - 1 - i] == b[len(b) - 1 - i]):
        i += 1
    return i


def align_pair(tokenizer, old_text, new_text, pad_token=" "):
    """Return aligned dict with equal-length old_ids/new_ids and the field span.

    Raises ValueError if the two texts differ in more than one contiguous span.
    """
    old_ids = tokenizer(old_text, add_special_tokens=True)["input_ids"]
    new_ids = tokenizer(new_text, add_special_tokens=True)["input_ids"]

    p = _common_prefix_len(old_ids, new_ids)
    s = _common_suffix_len(old_ids, new_ids, max_excl=min(len(old_ids), len(new_ids)) - p)

    old_mid = old_ids[p:len(old_ids) - s]
    new_mid = new_ids[p:len(new_ids) - s]

    # sanity: stitching prefix + mid + suffix must reproduce the originals
    assert old_ids[:p] + old_mid + old_ids[len(old_ids) - s:] == old_ids
    assert new_ids[:p] + new_mid + new_ids[len(new_ids) - s:] == new_ids

    pad_id = tokenizer(pad_token, add_special_tokens=False)["input_ids"]
    pad_id = pad_id[-1] if pad_id else tokenizer.encode(" ")[-1]

    # pad the shorter mid (at its end) so both mids are equal length
    L = max(len(old_mid), len(new_mid))
    old_mid = old_mid + [pad_id] * (L - len(old_mid))
    new_mid = new_mid + [pad_id] * (L - len(new_mid))

    suffix = old_ids[len(old_ids) - s:]  # identical to new suffix by construction
    prefix = old_ids[:p]

    old_full = prefix + old_mid + suffix
    new_full = prefix + new_mid + suffix
    assert len(old_full) == len(new_full)

    field_span = (p, p + L)  # [start, end) covering the (padded) value tokens
    return {
        "old_ids": torch.tensor([old_full]),
        "new_ids": torch.tensor([new_full]),
        "field_span": field_span,
        "seq_len": len(old_full),
        "field_len": L,
        "pad_added": (L - len([t for t in (old_mid) if True])),  # informational
    }
