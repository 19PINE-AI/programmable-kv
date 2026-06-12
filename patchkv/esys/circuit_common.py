"""Shared scaffolding for the CIRCUIT-LEVEL mechanism experiments (circ_*).

The mechd_* suite established, with causal KV patching, that the field-conditioned
*conclusion* is memoized onto specific downstream aggregator tokens at prefill and the
decision reads it there. That is a *localization* result (which tokens / layers / a linear
probe). These circuit experiments push to a *component-level* account:

  circ_heads      (Exp1) name the WRITE heads (carry field/trigger -> aggregator note) and
                         READ heads (aggregator note -> decision logit), by direct
                         attribution + causal head patching + attention patterns.
  circ_direction  (Exp2) a causal 1-D conclusion direction on the aggregator residual
                         (difference-of-means + directional patching + orthogonality/DAS).
  circ_sae        (Exp3) a sparse feature on the aggregator that carries the conclusion.
  circ_scrub      (Exp4) causal-scrubbing-style faithfulness of the write->note->read circuit.
  circ_components (Exp5) attn-vs-MLP decomposition of the write at the aggregator.

Task: the POLARITY 2x2 of mechd_common (conclusion = SAFE iff field==trigger). We build a
conclusion-flip PAIR with the field value held BYTE-IDENTICAL and only the rule trigger token
flipped, so every "conclusion" signal we attribute cannot be field content (the field is
constant across the pair). align_pair keeps the two equal length so RoPE matches everywhere
except the single trigger token inside the rule.

All hooks are plain HF forward hooks (no transformer_lens dependency); the model is the same
bf16 HF checkpoint the rest of the paper runs.
"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "e1"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from align import align_pair
from mechd_common import POL, build_pol, conclusion_is_safe

# single-token readout for each polarity action (space-prefixed handled by tokenizer)
ACT_TOK = {"escalate": "escalate", "issue_refund": "issue", "refuse": "refuse",
           "share": "share", "expedite": "expedite"}

SCNS = ["account_role", "safety_mode", "subscription_tier"]
OIDS = ["A4471", "B8820", "C1093", "D5567"]


def load_eager(name):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, device_map="cuda", attn_implementation="eager",
        dtype=torch.bfloat16, trust_remote_code=True).eval()
    return tok, model


def ftok(tok, w):
    return tok(w, add_special_tokens=False)["input_ids"][0]


def decoder_layers(model):
    # llama/qwen/mistral: model.model.layers
    return model.model.layers


def cfg_dims(model):
    c = model.config
    nh = c.num_attention_heads
    hd = getattr(c, "head_dim", None) or (c.hidden_size // nh)
    return nh, hd, c.hidden_size, c.num_hidden_layers


@torch.no_grad()
def prefill(model, ids):
    return model(input_ids=ids.to("cuda"), use_cache=True)


def clone_cache(c, upto):
    d = DynamicCache()
    for i, l in enumerate(c.layers):
        d.update(l.keys[:, :, :upto, :].clone(), l.values[:, :, :upto, :].clone(), i)
    return d


@torch.no_grad()
def decision_logits(model, cache, last, dpos):
    out = model(input_ids=torch.tensor([[last]], device="cuda"), past_key_values=cache,
                cache_position=torch.tensor([dpos], device="cuda"), use_cache=True)
    return out.logits[0, -1].float()


def build_pair(tok, scn, oid, thinking=False):
    """Conclusion-flip pair with the FIELD held identical.

    field value := POL[scn].values[0]; SAFE prompt has trigger==field (conclusion SAFE),
    UNSAFE prompt has trigger==the other value (conclusion UNSAFE). The two prompts differ
    in exactly one token (the trigger word inside the rule).
    Returns aligned tensors + spans + readout token ids.
    """
    s = POL[scn]
    fld = s["values"][0]
    other = s["values"][1]
    t_safe = build_pol(tok, scn, oid, fld, fld, thinking, True)      # conclusion SAFE
    t_unsafe = build_pol(tok, scn, oid, fld, other, thinking, True)  # conclusion UNSAFE
    al = align_pair(tok, t_safe, t_unsafe)
    safe_ids, unsafe_ids = al["old_ids"], al["new_ids"]              # [1,L], equal length
    a, b = al["field_span"]                                          # the differing trigger span
    L = safe_ids.shape[1]
    dpos = L - 1
    last = int(safe_ids[0, dpos])                                    # identical suffix
    toi = {"safe": ftok(tok, ACT_TOK[s["safe"]]), "unsafe": ftok(tok, ACT_TOK[s["unsafe"]])}
    return dict(safe_ids=safe_ids, unsafe_ids=unsafe_ids, trig_span=(a, b), L=L,
                dpos=dpos, last=last, toi=toi, scn=scn, oid=oid)


def conc_score(lg, toi):
    """logit[safe] - logit[unsafe]; high => model concludes SAFE."""
    return float(lg[toi["safe"]] - lg[toi["unsafe"]])


# ----------------------------------------------------------------------------
# aggregator localization (reuse the KV-patching recovery, restricted to the
# post-trigger downstream region) -- grounds "aggregator" causally, not by heuristic.
# clean = SAFE cache, corrupt = UNSAFE cache; patch SAFE KV into UNSAFE at a position set.
# ----------------------------------------------------------------------------
@torch.no_grad()
def kv_patch_score(model, c_corrupt, c_clean, positions, last, dpos, toi):
    w = clone_cache(c_corrupt, dpos)
    pos = torch.tensor(positions, device=w.layers[0].keys.device)
    for i in range(len(w.layers)):
        w.layers[i].keys[:, :, pos, :] = c_clean.layers[i].keys[:, :, pos, :]
        w.layers[i].values[:, :, pos, :] = c_clean.layers[i].values[:, :, pos, :]
    return conc_score(decision_logits(model, w, last, dpos), toi)


@torch.no_grad()
def find_aggregators(model, P, topn=8):
    """Rank post-trigger downstream positions by single-position KV-patch recovery.
    Returns (ranked_positions, recovery_dict, s_unsafe, s_safe). Recovery is toward SAFE."""
    co = prefill(model, P["unsafe_ids"]).past_key_values   # corrupt (UNSAFE conclusion)
    cn = prefill(model, P["safe_ids"]).past_key_values     # clean   (SAFE conclusion)
    last, dpos, toi = P["last"], P["dpos"], P["toi"]
    s_un = conc_score(decision_logits(model, clone_cache(co, dpos), last, dpos), toi)
    s_sa = conc_score(decision_logits(model, clone_cache(cn, dpos), last, dpos), toi)
    denom = s_sa - s_un
    b = P["trig_span"][1]
    down = list(range(b, dpos))
    rec = {}
    for p in down:
        sc = kv_patch_score(model, co, cn, [p], last, dpos, toi)
        rec[p] = (sc - s_un) / denom if abs(denom) > 1e-6 else 0.0
    ranked = sorted(down, key=lambda p: rec[p], reverse=True)
    return ranked[:topn], rec, s_un, s_sa, denom


# ----------------------------------------------------------------------------
# Hook machinery: capture per-head o_proj inputs (per-head context vectors) and
# per-layer attn-block / mlp-block outputs at all positions. Optionally PATCH a
# set of (layer, head) o_proj-input slices at a given position to injected values.
# ----------------------------------------------------------------------------
class Capture:
    """Capture attn o_proj INPUT (per-head context [B,S,H*Dh]), attn-block out, mlp out."""
    def __init__(self, model):
        self.model = model
        self.h = []
        self.attn_in = {}
        self.attn_out = {}
        self.mlp_out = {}

    def __enter__(self):
        for li, layer in enumerate(decoder_layers(self.model)):
            op = layer.self_attn.o_proj
            self.h.append(op.register_forward_pre_hook(self._cap_in(li)))
            self.h.append(layer.self_attn.register_forward_hook(self._cap_attn(li)))
            self.h.append(layer.mlp.register_forward_hook(self._cap_mlp(li)))
        return self

    def _cap_in(self, li):
        def hook(mod, args):
            self.attn_in[li] = args[0].detach()
        return hook

    def _cap_attn(self, li):
        def hook(mod, args, out):
            self.attn_out[li] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    def _cap_mlp(self, li):
        def hook(mod, args, out):
            self.mlp_out[li] = out.detach()
        return hook

    def __exit__(self, *a):
        for x in self.h:
            x.remove()
        self.h = []


class HeadPatch:
    """Forward-pre-hook on o_proj that overwrites per-head context slices at one position.

    patch: dict layer_idx -> list of (head, replacement_vector[Dh]).  pos: absolute position.
    """
    def __init__(self, model, patch, pos, hd):
        self.model = model
        self.patch = patch
        self.pos = pos
        self.hd = hd
        self.h = []

    def __enter__(self):
        for li, layer in enumerate(decoder_layers(self.model)):
            if li in self.patch:
                self.h.append(layer.self_attn.o_proj.register_forward_pre_hook(self._mk(li)))
        return self

    def _mk(self, li):
        reps = self.patch[li]
        pos, hd = self.pos, self.hd

        def hook(mod, args):
            x = args[0].clone()
            if x.shape[1] > pos:                       # only during prefill (S>pos)
                for (head, vec) in reps:
                    x[0, pos, head * hd:(head + 1) * hd] = vec.to(x.dtype)
            return (x,) + args[1:]
        return hook

    def __exit__(self, *a):
        for x in self.h:
            x.remove()
        self.h = []


def head_contribs(attn_in_layer, o_proj_weight, nh, hd, pos):
    """Per-head contribution to the residual at `pos`: [nh, hidden].
    contribution_h = context[pos, h_slice] @ W_o[:, h_slice].T  (Llama o_proj has no bias)."""
    ctx = attn_in_layer[0, pos]                         # [H*Dh]
    W = o_proj_weight                                   # [hidden, H*Dh]
    out = torch.empty(nh, W.shape[0], device=W.device, dtype=torch.float32)
    for h in range(nh):
        sl = slice(h * hd, (h + 1) * hd)
        out[h] = (W[:, sl].float() @ ctx[sl].float())
    return out


def logit_dir(model, toi):
    """Unembedding conclusion direction e_safe - e_unsafe in residual space (pre-final-norm
    contributions get projected through final RMSNorm scale at read time; for attribution we
    use the raw unembed row difference, the standard direct-logit-attribution direction)."""
    W_U = model.lm_head.weight                          # [vocab, hidden]
    return (W_U[toi["safe"]].float() - W_U[toi["unsafe"]].float())


def final_norm(model):
    return model.model.norm
