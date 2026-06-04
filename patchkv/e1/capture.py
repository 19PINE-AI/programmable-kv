"""Per-layer post-RoPE Q/K/V capture for E1 blast-radius characterization.

We must keep config._attn_implementation == "eager" so the model builds the
correct *additive causal mask* (other implementations may return None and rely on
an is_causal fast path, which a captured eager call would then run WITHOUT a mask
-> silent bidirectional attention). So instead of registering a new attention
name, we monkeypatch the architecture module's `eager_attention_forward` global
(which Qwen2Attention.forward reads at call time) with a wrapper that records the
post-RoPE (query, key, value) and delegates to the original eager function.
"""
import torch

_CAPTURE = {}
_ENABLED = {"on": False, "to_cpu": True}
_ORIG = {}  # module -> original eager fn


def _make_wrapper(orig):
    def capturing(module, query, key, value, attention_mask, scaling,
                  dropout=0.0, **kwargs):
        if _ENABLED["on"]:
            li = getattr(module, "layer_idx", len(_CAPTURE))
            dev = "cpu" if _ENABLED["to_cpu"] else query.device
            _CAPTURE[li] = {
                "q": query.detach().to(dev),
                "k": key.detach().to(dev),
                "v": value.detach().to(dev),
                "scaling": float(scaling),
            }
        return orig(module, query, key, value, attention_mask, scaling,
                    dropout, **kwargs)
    return capturing


def install(model):
    """Patch the eager_attention_forward of the model's architecture module(s)."""
    import importlib
    mod_name = type(model).__module__            # e.g. transformers.models.qwen2.modeling_qwen2
    mod = importlib.import_module(mod_name)
    if not hasattr(mod, "eager_attention_forward"):
        raise RuntimeError(f"{mod_name} has no eager_attention_forward to patch")
    if mod not in _ORIG:
        _ORIG[mod] = mod.eager_attention_forward
        mod.eager_attention_forward = _make_wrapper(_ORIG[mod])
    # make sure layers actually take the eager path
    model.config._attn_implementation = "eager"
    for m in model.modules():
        if hasattr(m, "config"):
            m.config._attn_implementation = "eager"
    return model


def enable_capture(to_cpu=True):
    _CAPTURE.clear()
    _ENABLED["on"] = True
    _ENABLED["to_cpu"] = to_cpu


def disable_capture():
    _ENABLED["on"] = False


def get_capture():
    return dict(_CAPTURE)
