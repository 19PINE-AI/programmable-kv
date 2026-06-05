"""editkv — Editable KV cache for mutable fields in agentic contexts.

Two incarnations for changing a field in an already-cached prompt:
  * Mode.IN_PLACE — surgically recompute the field's KV (exact), leave the rest stale.
  * Mode.ERRATUM  — leave the cache stale, append a salient trigger, recompute only it.
  * Mode.AUTO     — use editkv.diagnostics.needs_erratum to choose per-edit.

Quick start:
    from editkv import EditableContext, Mode
    ctx = EditableContext(model, tok)
    ctx.add_text(policy + "\n\nSESSION\n")
    status = ctx.add_field("order_status", "pending")
    ctx.add_text("\n" + conversation + "\nNext action:")
    ctx.prefill()
    print(ctx.generate("order_status", "delivered", mode=Mode.ERRATUM))

    from editkv.diagnostics import needs_erratum
    d = needs_erratum(ctx, "order_status", "delivered")
    print(d.needs_erratum, d.note)
"""
from .core import EditableContext, Field, Mode, LengthChangeError
from .diagnostics import needs_erratum, blast_radius, Diagnosis

__all__ = ["EditableContext", "Field", "Mode", "LengthChangeError",
           "needs_erratum", "blast_radius", "Diagnosis"]
__version__ = "0.1.0"
