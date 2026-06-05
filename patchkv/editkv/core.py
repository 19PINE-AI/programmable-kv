"""editkv.core — Editable KV cache for mutable fields in agentic contexts.

An `EditableContext` builds a context with declared mutable *fields*, prefills it once
into a KV cache, and lets you change a field later with one of two mechanisms, then
generate the agent's next action:

  * IN_PLACE  — recompute only the changed field's KV (exact: it attends only to the
                unchanged prefix) and overwrite it; leave all downstream KV stale.
                Cheapest (~field tokens). Sufficient for low-conditioning fields and for
                reasoning models in benign contexts; can revert to the stale decision
                otherwise (see editkv.diagnostics).
  * ERRATUM   — leave the whole cache stale and append a short, salient trigger
                ("[STATE UPDATE] <field> -> <new>; overrides any earlier value and
                conclusion"); recompute only the appended span. Robust across models,
                scales, and contradictory context; length-agnostic; ~tens of tokens.

You prefill the CONTEXT (system + conversation, up to where the agent would act). At
generation time you pick an edit mode and (optionally) a decision prompt; the trigger and
prompt are appended at the suffix and only those tokens are computed.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import torch
from transformers.cache_utils import DynamicCache


class Mode(str, Enum):
    STALE = "stale"                    # no edit (baseline)
    IN_PLACE = "in_place"
    ERRATUM = "erratum"
    FIELD_PLUS_ERRATUM = "field_plus_erratum"
    FULL_REPREFILL = "full_reprefill"  # ceiling baseline (recompute everything)
    AUTO = "auto"                      # use the diagnostic to choose in_place vs erratum


class LengthChangeError(ValueError):
    pass


@dataclass
class Field:
    name: str
    value: str
    span: Optional[tuple] = None
    label: Optional[str] = None

    def render(self) -> str:
        return f"{self.label or self.name}: {self.value}"


DEFAULT_TRIGGER = ("[STATE UPDATE] {label} has changed to {value}; this overrides any "
                   "earlier value AND any earlier conclusion. Apply the current value.")


def _common_prefix_len(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class EditableContext:
    def __init__(self, model, tokenizer, device: str = "cuda", trigger_template: str = DEFAULT_TRIGGER):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.trigger_template = trigger_template
        self._segments: List = []
        self.fields = {}
        self._ids = None        # context token ids (1, L)
        self._cache = None      # DynamicCache of length L (full context)
        self._len = None        # L

    # ---- build the context ----
    def add_text(self, text: str) -> "EditableContext":
        self._segments.append(("text", text)); return self

    def add_field(self, name: str, value: str, label: Optional[str] = None) -> Field:
        f = Field(name=name, value=value, label=label)
        self.fields[name] = f
        self._segments.append(("field", f)); return f

    def _render(self) -> str:
        return "".join(seg if k == "text" else seg.render() for k, seg in self._segments)

    def _tok(self, text, special=True):
        return torch.tensor([self.tok(text, add_special_tokens=special)["input_ids"]])

    # ---- prefill ----
    @torch.no_grad()
    def prefill(self) -> "EditableContext":
        ids = self._tok(self._render())
        self._ids = ids
        # locate each field's value token span
        running = ""
        for k, seg in self._segments:
            if k == "text":
                running += seg; continue
            f: Field = seg
            label = f"{f.label or f.name}: "
            running += f.render()
            with_field = self._tok(running)[0].tolist()
            anchored = self._tok(running[:running.rfind(f.render())] + label)[0].tolist()
            f.span = (_common_prefix_len(with_field, anchored), len(with_field))
        self._cache = self.model(input_ids=ids.to(self.device), use_cache=True).past_key_values
        self._len = ids.shape[1]
        return self

    def _clone(self, cache, upto):
        c = DynamicCache()
        for i, l in enumerate(cache.layers):
            c.update(l.keys[:, :, :upto, :].clone(), l.values[:, :, :upto, :].clone(), i)
        return c

    # ---- build the edited, decode-ready cache ----
    @torch.no_grad()
    def _refresh_field_inplace(self, cache, f: Field, new_value: str):
        """Overwrite f.span KV with the exact recomputed KV for `new_value` (length-preserving)."""
        s, e = f.span
        new_render = self._render().replace(f.render(), Field(f.name, new_value, label=f.label).render(), 1)
        new_ids = self._tok(new_render)
        if new_ids.shape[1] != self._len:
            raise LengthChangeError(
                f"field '{f.name}' edit changes token length ({self._len}->{new_ids.shape[1]}); "
                f"use Mode.ERRATUM (length-agnostic).")
        prefix = self._clone(cache, s)
        out = self.model(input_ids=new_ids[:, s:e].to(self.device), past_key_values=prefix,
                         cache_position=torch.arange(s, e, device=self.device), use_cache=True)
        for i in range(len(cache.layers)):
            cache.layers[i].keys[:, :, s:e, :] = out.past_key_values.layers[i].keys[:, :, s:e, :]
            cache.layers[i].values[:, :, s:e, :] = out.past_key_values.layers[i].values[:, :, s:e, :]
        return cache

    @torch.no_grad()
    def build_cache(self, field_name: str, new_value: str, mode: Mode = Mode.ERRATUM,
                    decision_prompt: str = ""):
        """Build a decode-ready cache for the edit. Returns (cache, last_token, decode_pos):
        the cache has length `decode_pos`; feeding `last_token` at `decode_pos` yields the
        next-token logits. Recompute cost = (field tokens if in_place) + (trigger+prompt)."""
        f = self.fields[field_name]
        if mode == Mode.AUTO:
            # Diagnostic decides between the cheap in_place and the robust FIELD_PLUS_ERRATUM.
            # field+erratum (refresh the token AND append the override) is the robust escalation:
            # erratum alone can miss when the field is buried early in a long policy context.
            from .diagnostics import needs_erratum
            mode = Mode.FIELD_PLUS_ERRATUM if needs_erratum(self, field_name, new_value,
                                                  probe=decision_prompt).needs_erratum else Mode.IN_PLACE
        # build the suffix to append (trigger + decision prompt)
        suffix = ""
        if mode in (Mode.ERRATUM, Mode.FIELD_PLUS_ERRATUM):
            suffix += "\n" + self.trigger_template.format(label=f.label or f.name, value=new_value) + "\n"
        suffix += decision_prompt
        suffix_ids = self.tok(suffix, add_special_tokens=False)["input_ids"]

        if mode == Mode.FULL_REPREFILL:
            new_render = self._render().replace(f.render(), Field(f.name, new_value, label=f.label).render(), 1)
            ids = self._tok(new_render + suffix)
            cache = self.model(input_ids=ids[:, :-1].to(self.device), use_cache=True).past_key_values
            return cache, int(ids[0, -1]), ids.shape[1] - 1

        cache = self._clone(self._cache, self._len)              # full stale context (length L)
        if mode in (Mode.IN_PLACE, Mode.FIELD_PLUS_ERRATUM):
            try:
                self._refresh_field_inplace(cache, f, new_value)
            except LengthChangeError:
                if mode == Mode.IN_PLACE:
                    raise
        L = self._len
        if len(suffix_ids) == 0:                                 # no suffix -> decode from context end
            return self._clone(cache, L - 1), int(self._ids[0, L - 1]), L - 1
        if len(suffix_ids) > 1:                                  # forward all but the last suffix token
            self.model(input_ids=torch.tensor([suffix_ids[:-1]], device=self.device), past_key_values=cache,
                       cache_position=torch.arange(L, L + len(suffix_ids) - 1, device=self.device), use_cache=True)
        return cache, int(suffix_ids[-1]), L + len(suffix_ids) - 1

    @torch.no_grad()
    def generate(self, field_name: str, new_value: str, mode: Mode = Mode.ERRATUM,
                 decision_prompt: str = "", max_new_tokens: int = 64, stop_str: Optional[str] = "\n") -> str:
        cache, last, pos = self.build_cache(field_name, new_value, mode, decision_prompt)
        toks = []; cur = last; p = pos; eos = self.tok.eos_token_id
        for _ in range(max_new_tokens):
            out = self.model(input_ids=torch.tensor([[cur]], device=self.device), past_key_values=cache,
                             cache_position=torch.tensor([p], device=self.device), use_cache=True)
            nx = int(out.logits[0, -1].argmax()); toks.append(nx); p += 1
            if nx == eos or (stop_str and stop_str in self.tok.decode(toks)):
                break
            cur = nx
        return self.tok.decode(toks, skip_special_tokens=True)
