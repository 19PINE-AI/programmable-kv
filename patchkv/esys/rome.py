"""A faithful, self-contained ROME (Rank-One Model Editing) for Llama/Mistral-style MLPs.

ROME (Meng et al., 2022) inserts a new factual association by a rank-one update to one
MLP down-projection W_down at a mid layer L. Given a key k* (the down_proj input at the
subject's last token) and an optimized value v* (the target down_proj output that makes the
model emit the desired object), and the uncentered key covariance C = E[k k^T] estimated
from text, the closed-form update is

    W' = W + (v* - W k*) (C^{-1} k*)^T / ((C^{-1} k*) . k*)

so that W' k* = v* while perturbing other directions minimally (in the C metric).

We target the SwiGLU MLP: down_proj input = act_fn(gate_proj(x)) * up_proj(x). This module
implements estimate_cov / compute_k_star / compute_v_star / apply_rome / restore, and a
self-test (`python esys/rome.py`) that edits "The Eiffel Tower is in the city of" Paris->Rome
to verify the implementation is correct before it is used as a baseline.
"""
import os, sys, torch
import torch.nn.functional as F


def _layer_mlp(model, L):
    return model.model.layers[L].mlp


def _down(model, L):
    return _layer_mlp(model, L).down_proj


@torch.no_grad()
def estimate_cov(model, tok, L, texts, max_tokens=50000, device="cuda"):
    """C = sum k k^T over down_proj inputs at layer L (uncentered 2nd moment)."""
    mlp = _layer_mlp(model, L)
    feats = {}
    def hook(mod, inp, out):
        feats["k"] = inp[0].detach()             # (1, T, d_intermediate) input to down_proj
    h = mlp.down_proj.register_forward_hook(hook)
    d = mlp.down_proj.in_features
    C = torch.zeros(d, d, dtype=torch.float64, device=device)
    n = 0
    try:
        for t in texts:
            if n >= max_tokens:
                break
            ids = torch.tensor([tok(t, add_special_tokens=True)["input_ids"][:512]]).to(device)
            if ids.shape[1] < 2:
                continue
            model(input_ids=ids, use_cache=False)
            k = feats["k"][0].double()           # (T, d)
            C += k.T @ k
            n += k.shape[0]
    finally:
        h.remove()
    C /= max(1, n)
    return C, n


@torch.no_grad()
def compute_k_star(model, tok, L, prompts, device="cuda"):
    """k* = mean down_proj-input at the LAST token over several context prompts."""
    mlp = _layer_mlp(model, L)
    feats = {}
    h = mlp.down_proj.register_forward_hook(lambda m, i, o: feats.__setitem__("k", i[0].detach()))
    ks = []
    try:
        for p in prompts:
            ids = torch.tensor([tok(p, add_special_tokens=True)["input_ids"]]).to(device)
            model(input_ids=ids, use_cache=False)
            ks.append(feats["k"][0, -1].double())
    finally:
        h.remove()
    return torch.stack(ks).mean(0)


def compute_v_star(model, tok, L, edit_prompt, target_token_id, n_steps=25, lr=0.5,
                   kl_weight=0.0625, device="cuda"):
    """Optimize a delta added to the layer-L MLP output at the last position so the model's
    next-token logits favor `target_token_id`. Returns v* = (down_proj output at last pos) + delta."""
    mlp = _layer_mlp(model, L)
    ids = torch.tensor([tok(edit_prompt, add_special_tokens=True)["input_ids"]]).to(device)
    last = ids.shape[1] - 1
    # capture the clean down_proj output (the value we will shift)
    clean = {}
    h0 = mlp.down_proj.register_forward_hook(lambda m, i, o: clean.__setitem__("v", o.detach()[0, last].clone()))
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    h0.remove()
    delta = torch.zeros_like(clean["v"], requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    def add_hook(m, i, o):
        o = o.clone(); o[0, last] = o[0, last] + delta.to(o.dtype); return o
    for _ in range(n_steps):
        opt.zero_grad()
        h = mlp.down_proj.register_forward_hook(add_hook)
        try:
            logits = model(input_ids=ids, use_cache=False).logits[0, last].float()
        finally:
            h.remove()
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_token_id], device=device))
        loss = loss + kl_weight * delta.float().pow(2).mean()   # stay-close regularizer
        loss.backward()
        opt.step()
    return (clean["v"].double() + delta.detach().double())


@torch.no_grad()
def apply_rome(model, L, k_star, v_star, C, lam=1e4, device="cuda"):
    """Rank-one edit of W_down at layer L. Returns the original weight (for restore)."""
    W = _down(model, L).weight                     # (d_out, d_in)
    Wd = W.data.double()
    d = C.shape[0]
    Creg = C + (lam / d) * torch.eye(d, dtype=torch.float64, device=device) * C.diag().mean()
    Cinv_k = torch.linalg.solve(Creg, k_star)      # C^{-1} k*
    denom = (Cinv_k @ k_star).clamp_min(1e-6)
    residual = v_star - Wd @ k_star                # (d_out,)
    update = torch.outer(residual, Cinv_k) / denom # (d_out, d_in)
    orig = W.data.clone()
    W.data = (Wd + update).to(W.dtype)
    return orig


@torch.no_grad()
def restore(model, L, orig):
    _down(model, L).weight.data.copy_(orig)


# ---------------- self-test: canonical factual edit ----------------
def _self_test():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    name = os.environ.get("ROME_TEST_MODEL", "unsloth/Meta-Llama-3.1-8B-Instruct")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, device_map="cuda", dtype=torch.bfloat16).eval()
    L = int(os.environ.get("ROME_LAYER", model.config.num_hidden_layers // 5))
    prompt = "The Eiffel Tower is located in the city of"
    tgt = " Rome"
    tgt_id = tok(tgt, add_special_tokens=False)["input_ids"][0]
    def top(p):
        ids = torch.tensor([tok(p, add_special_tokens=True)["input_ids"]]).cuda()
        with torch.no_grad():
            lg = model(ids).logits[0, -1]
        return tok.decode([lg.argmax()])
    print("before:", repr(top(prompt)))
    corpus = [f"This is sentence number {i} about various unrelated everyday topics and places." for i in range(200)]
    C, n = estimate_cov(model, tok, L, corpus, max_tokens=20000)
    print(f"cov over {n} tokens, layer {L}")
    k = compute_k_star(model, tok, L, [prompt, "The Eiffel Tower is in", "Where is the Eiffel Tower? It is in"])
    v = compute_v_star(model, tok, L, prompt, tgt_id)
    apply_rome(model, L, k, v, C)
    print("after: ", repr(top(prompt)))
    print("locality (should be unchanged): 'The Colosseum is in the city of' ->", repr(top("The Colosseum is in the city of")))


if __name__ == "__main__":
    _self_test()
