"""H3 — Multimodal image transplant with M-RoPE POSITION SHIFT (temporal re-rotation).

Encode an image once at position A (short prefix); transplant its cached KV into a LONGER context where
the image sits at a different position B, re-rotating the image keys with old->new 3D position-ids
(only the temporal mrope-section changes; h,w cancel). VQA agreement vs full re-encode at B. This is the
general 'encode once, reuse anywhere' claim for images. Run: python esys/composable_vision_shift.py
"""
import argparse, os, sys, json, random
import torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoProcessor
from transformers.cache_utils import DynamicCache
from composable_vision import load_vlm, cache_slice, cache_concat, COLORS


def make_img(digit, rgb, size=672):
    im = Image.new("RGB", (size, size), rgb); d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=int(size * .6))
    except Exception:
        f = ImageFont.load_default()
    d.text((size * .33, size * .12), str(digit), fill=(255, 255, 255), font=f)
    return im


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], -1)


@torch.no_grad()
def mrope_cossin(rot, position_ids, mrope_section):
    """Assembled M-RoPE cos/sin for given 3D position_ids [3,1,L] -> [1,L,D] (fp32)."""
    dummy = torch.zeros(1, position_ids.shape[-1], 1, dtype=torch.float32, device="cuda")
    cos, sin = rot(dummy, position_ids)            # [3,1,L,D]
    ms = [s * 2 for s in mrope_section]
    cos = torch.cat([c[i % 3] for i, c in enumerate(cos.float().split(ms, dim=-1))], dim=-1)  # [1,L,D]
    sin = torch.cat([s[i % 3] for i, s in enumerate(sin.float().split(ms, dim=-1))], dim=-1)
    return cos[:, None], sin[:, None]              # [1,1,L,D]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct"); ap.add_argument("--tag", default=None)
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = load_vlm(args.model)
    rot = model.model.rotary_emb if hasattr(model.model, "rotary_emb") else model.model.language_model.rotary_emb
    ms = model.config.rope_scaling["mrope_section"]
    img_tok = getattr(model.config, "image_token_id", None) or getattr(model.config, "image_token_index", None)

    def build(img, q, pad=""):
        content = ([{"type": "text", "text": pad}] if pad else []) + [{"type": "image"}, {"type": "text", "text": q}]
        text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
        return proc(text=[text], images=[img], return_tensors="pt").to("cuda")

    def span(ids):
        m = (ids[0] == img_tok).nonzero().squeeze(-1); return int(m[0]), int(m[-1]) + 1

    def gen_from(cache, ids, L, n=4):
        try:
            pids, _ = model.model.get_rope_index(ids, None, attention_mask=torch.ones_like(ids))
        except Exception:
            pids = None
        out = []; past = cache_slice(cache, 0, L - 1); cur = int(ids[0, L - 1]); pos = L - 1
        for _ in range(n):
            kw = dict(input_ids=torch.tensor([[cur]], device="cuda"), past_key_values=past,
                      cache_position=torch.tensor([pos], device="cuda"), use_cache=True)
            o = model.model(**kw); past = o.past_key_values; pos += 1
            lg = model.lm_head(o.last_hidden_state); cur = int(lg[0, -1].argmax()); out.append(cur)
        return proc.tokenizer.decode(out, skip_special_tokens=True).lower()

    rng = random.Random(0); shifted, full_at_b = [], []
    PAD = "Context preface. " * 40  # makes the image land at a later position in context B
    for i in range(args.n):
        dg = rng.randint(0, 9); cn, rgb = rng.choice(COLORS); ask_c = (i % 2 == 0)
        img = make_img(dg, rgb); q = ("What is the background colour? One word." if ask_c else "What digit is shown? One word.")
        ans = (cn if ask_c else str(dg))
        # context A (short): encode image, cache its KV + its position_ids
        iA = build(img, q); idsA = iA["input_ids"]; aA, bA = span(idsA)
        cacheA = model(**iA, use_cache=True).past_key_values
        pidsA, _ = model.model.get_rope_index(idsA, iA.get("image_grid_thw"), attention_mask=torch.ones_like(idsA))
        # context B (padded): image at a later position; FULL re-encode for reference
        iB = build(img, q, pad=PAD); idsB = iB["input_ids"]; LB = idsB.shape[1]; aB, bB = span(idsB)
        pidsB, _ = model.model.get_rope_index(idsB, iB.get("image_grid_thw"), attention_mask=torch.ones_like(idsB))
        full_b = model.generate(**iB, max_new_tokens=4, do_sample=False)
        full_txt = proc.batch_decode(full_b[:, LB:], skip_special_tokens=True)[0].lower()
        # SHIFTED transplant: re-rotate cached image keys from pidsA[span] to pidsB[span]
        cs_o, ss_o = mrope_cossin(rot, pidsA[:, :, aA:bA], ms); cs_n, ss_n = mrope_cossin(rot, pidsB[:, :, aB:bB], ms)
        img_kv = DynamicCache()
        for li, l in enumerate(cacheA.layers):
            k = l.keys[:, :, aA:bA, :].float()
            raw = k * cs_o - rotate_half(k) * ss_o
            kk = (raw * cs_n + rotate_half(raw) * ss_n).to(l.keys.dtype)
            img_kv.update(kk, l.values[:, :, aA:bA, :], li)
        # build B cache: prefill text[0,aB) + shifted image KV + text[bB,L)
        def fwd(seg, past, start):
            return model.model(input_ids=seg, past_key_values=past, use_cache=True,
                               position_ids=pidsB[:, :, start:start + seg.shape[1]],
                               cache_position=torch.arange(start, start + seg.shape[1], device="cuda"))
        pre = fwd(idsB[:, :aB], DynamicCache(), 0).past_key_values
        spliced = cache_concat(pre, img_kv)
        o = fwd(idsB[:, bB:LB], spliced, bB)
        sh_txt = gen_from(o.past_key_values, idsB, LB)
        shifted.append(int(ans in sh_txt)); full_at_b.append(int(ans in full_txt))
        if i < 3 or i % 8 == 0:
            print(f"  [{i}] ans={ans} full@B={full_txt[:8]!r}({ans in full_txt}) shifted={sh_txt[:8]!r}({ans in sh_txt}) | posA_img={aA} posB_img={aB}", flush=True)
    n = args.n; agree = sum(s == f for s, f in zip(shifted, full_at_b))
    out = {"model": args.model, "n": n, "full_at_B_acc": round(sum(full_at_b) / n, 3),
           "shifted_transplant_acc": round(sum(shifted) / n, 3), "agreement": round(agree / n, 3)}
    print(f"\n=== M-RoPE POSITION-SHIFT image transplant ({args.model}, n={n}) ===")
    print(f"  full re-encode @B acc={out['full_at_B_acc']} | shifted-transplant acc={out['shifted_transplant_acc']} | agreement={out['agreement']}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"vision_shift_{tag}.json"), "w"), indent=2)
    print("VISION_SHIFT_DONE")


if __name__ == "__main__":
    main()
