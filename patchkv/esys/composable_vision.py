"""G6 — Composable KV for IMAGES: pre-encode an image once, splice its cached KV into a trajectory,
skip the vision-tower + image-token prefill. Feasibility: does VQA survive the transplant?

Use case: an agent reads an image; today that costs a full prefill (vision tower + prefill of the
>1k image soft-tokens). We cache the image's LM KV once and SPLICE it in, re-running only the text.
We test SAME-position reuse (no reposition) first — the core "skip the prefill" claim — measuring VQA
agreement vs full re-encode and the image-token prefill saved. M-RoPE note: image position is (t,h,w);
moving the image shifts only the temporal t (h,w intrinsic), so a position shift re-rotates only the
temporal mrope section. Run: python esys/composable_vision.py --model Qwen/Qwen2.5-VL-3B-Instruct
"""
import argparse, os, sys, json, random
import torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoProcessor
from transformers.cache_utils import DynamicCache

COLORS = [("red", (200, 30, 30)), ("green", (30, 160, 30)), ("blue", (40, 60, 200)),
          ("yellow", (220, 200, 40)), ("purple", (130, 40, 160)), ("orange", (230, 130, 30))]


def make_img(digit, rgb, size=336):
    im = Image.new("RGB", (size, size), rgb)
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=int(size * 0.6))
    except Exception:
        f = ImageFont.load_default()
    d.text((size * 0.32, size * 0.12), str(digit), fill=(255, 255, 255), font=f)
    return im


def cache_slice(c, lo, hi):
    d = DynamicCache()
    for i, l in enumerate(c.layers):
        d.update(l.keys[:, :, lo:hi, :].clone(), l.values[:, :, lo:hi, :].clone(), i)
    return d


def cache_concat(*cs):
    d = DynamicCache()
    for i in range(len(cs[0].layers)):
        d.update(torch.cat([c.layers[i].keys for c in cs], 2), torch.cat([c.layers[i].values for c in cs], 2), i)
    return d


def load_vlm(name):
    import transformers as T
    dtype = torch.bfloat16
    for cls in ["Qwen3VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration", "Qwen2VLForConditionalGeneration",
                "Gemma3ForConditionalGeneration", "AutoModelForImageTextToText"]:
        if hasattr(T, cls) and (cls.split("For")[0].lower().replace("_", "") in name.lower().replace("-", "").replace(".", "") or cls == "AutoModelForImageTextToText"):
            try:
                return getattr(T, cls).from_pretrained(name, dtype=dtype, device_map="cuda", trust_remote_code=True).eval()
            except Exception as e:
                print(f"[load] {cls} failed: {str(e)[:60]}")
    return T.AutoModelForImageTextToText.from_pretrained(name, dtype=dtype, device_map="cuda", trust_remote_code=True).eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct"); ap.add_argument("--tag", default=None)
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = load_vlm(args.model)
    cfg = model.config
    img_tok = getattr(cfg, "image_token_id", None) or getattr(cfg, "image_token_index", None)
    print(f"loaded {args.model}; image_token_id={img_tok}")

    def build(img, q):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return proc(text=[text], images=[img], return_tensors="pt").to("cuda")

    def img_span(input_ids):
        m = (input_ids[0] == img_tok).nonzero().squeeze(-1)
        return int(m[0]), int(m[-1]) + 1

    rng = random.Random(0)
    ag_full, ag_pre, agree = [], [], []
    for i in range(args.n):
        digit = rng.randint(0, 9); cname, rgb = rng.choice(COLORS)
        ask_color = (i % 2 == 0)
        img = make_img(digit, rgb)
        q = "What color is the background? Answer one word." if ask_color else "What digit is shown? Answer one word."
        ans = cname if ask_color else str(digit)
        inp = build(img, q); ids = inp["input_ids"]; L = ids.shape[1]; a, b = img_span(ids)
        # FULL forward -> answer
        gen = model.generate(**inp, max_new_tokens=4, do_sample=False)
        full_txt = proc.batch_decode(gen[:, L:], skip_special_tokens=True)[0].lower()
        # cache the FULL kv once (this contains the image-token KV at [a,b))
        out = model(**inp, use_cache=True)
        full_cache = out.past_key_values
        # PRECOMPILED reuse: prefill text [0,a), splice cached image KV [a,b), prefill text [b,L) -- NO pixel_values
        pos = None
        try:
            # text-only forwards need M-RoPE position ids that match the image layout
            pids, _ = model.get_rope_index(ids, inp.get("image_grid_thw"), attention_mask=torch.ones_like(ids))
        except Exception:
            pids = None
        def fwd(seg_ids, past, start):
            n = seg_ids.shape[1]
            kw = dict(input_ids=seg_ids, past_key_values=past, use_cache=True,
                      cache_position=torch.arange(start, start + n, device="cuda"))
            if pids is not None:
                kw["position_ids"] = pids[:, :, start:start + n]
            return model.model(**kw) if hasattr(model, "model") else model(**kw)
        pre = fwd(ids[:, :a], DynamicCache(), 0).past_key_values
        spliced = cache_concat(pre, cache_slice(full_cache, a, b))
        fin = fwd(ids[:, b:L], spliced, b)
        lg = fin.logits if hasattr(fin, "logits") else model.lm_head(fin.last_hidden_state)
        nxt = int(lg[0, -1].argmax())
        # greedy 3 more
        dec = [nxt]; past = fin.past_key_values; pos2 = L
        for _ in range(3):
            o = fwd(torch.tensor([[dec[-1]]], device="cuda"), past, pos2); pos2 += 1
            past = o.past_key_values
            lg = o.logits if hasattr(o, "logits") else model.lm_head(o.last_hidden_state)
            dec.append(int(lg[0, -1].argmax()))
        pre_txt = proc.tokenizer.decode(dec, skip_special_tokens=True).lower()
        fc = ans in full_txt; pc = ans in pre_txt
        ag_full.append(int(fc)); ag_pre.append(int(pc)); agree.append(int(fc == pc))
        if i < 4 or i % 8 == 0:
            print(f"  [{i}] ans={ans} full={full_txt[:8]!r}({fc}) precompiled={pre_txt[:8]!r}({pc}) | img_tokens={b-a}", flush=True)
    n = args.n
    out = {"model": args.model, "n": n, "img_tokens": b - a,
           "full_acc": round(sum(ag_full) / n, 3), "precompiled_acc": round(sum(ag_pre) / n, 3),
           "agreement": round(sum(agree) / n, 3)}
    print(f"\n=== IMAGE KV TRANSPLANT ({args.model}, n={n}, ~{b-a} img tokens) ===")
    print(f"  full VQA acc={out['full_acc']} | precompiled(spliced image KV) acc={out['precompiled_acc']} | agreement={out['agreement']}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"composable_vision_{tag}.json"), "w"), indent=2)
    print("VISION_DONE")


if __name__ == "__main__":
    main()
