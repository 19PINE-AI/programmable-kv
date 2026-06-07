"""G6 — Composable KV for IMAGES across DIVERSE tasks (perception / visual-reasoning / agentic), N>=100.

Cache an image's LM KV once, splice it into the trajectory (skip the vision tower + image-token
prefill), re-run only text. Feasibility across diverse tasks with bootstrap 95% CIs, per category:
  perception : read a digit / name the colour
  reasoning  : count shapes, identify a shape, spatial (left/right colour), size comparison
  agentic    : the image governs a TOOL decision (status light -> halt/proceed; gauge fill -> scale)
We report full-VQA vs precompiled(spliced image KV) accuracy + agreement, per category, per model.
M-RoPE handled via get_rope_index; same-position reuse needs no temporal re-rotation.
Run: python esys/composable_vision.py --model Qwen/Qwen2.5-VL-7B-Instruct --n 120
"""
import argparse, os, sys, json, random
import torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoProcessor
from transformers.cache_utils import DynamicCache

COLORS = [("red", (200, 30, 30)), ("green", (30, 160, 30)), ("blue", (40, 60, 200)),
          ("yellow", (220, 200, 40)), ("purple", (130, 40, 160)), ("orange", (230, 130, 30))]
SHAPES = ["circle", "square", "triangle"]


def _font(sz):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=sz)
    except Exception:
        return ImageFont.load_default()


def _shape(d, kind, box, fill):
    x0, y0, x1, y1 = box
    if kind == "circle":
        d.ellipse(box, fill=fill)
    elif kind == "square":
        d.rectangle(box, fill=fill)
    else:
        d.polygon([(x0, y1), ((x0 + x1) / 2, y0), (x1, y1)], fill=fill)


def gen_tasks(n, size=1008):
    rng = random.Random(0); tasks = []
    cats = ["perc_digit", "perc_color", "reason_count", "reason_shape", "reason_spatial", "reason_size",
            "agentic_status", "agentic_gauge"]
    while len(tasks) < n:
        cat = cats[len(tasks) % len(cats)]
        im = Image.new("RGB", (size, size), (245, 245, 245)); d = ImageDraw.Draw(im)
        if cat == "perc_digit":
            dg = rng.randint(0, 9); cn, rgb = rng.choice(COLORS)
            im.paste(Image.new("RGB", (size, size), rgb)); d = ImageDraw.Draw(im)
            d.text((size * .33, size * .12), str(dg), fill=(255, 255, 255), font=_font(int(size * .6)))
            q, ans, c = "What digit is shown? Answer one word.", str(dg), "perception"
        elif cat == "perc_color":
            dg = rng.randint(0, 9); cn, rgb = rng.choice(COLORS)
            im.paste(Image.new("RGB", (size, size), rgb)); d = ImageDraw.Draw(im)
            d.text((size * .33, size * .12), str(dg), fill=(255, 255, 255), font=_font(int(size * .6)))
            q, ans, c = "What is the background colour? Answer one word.", cn, "perception"
        elif cat == "reason_count":
            k = rng.randint(2, 5); cn, rgb = rng.choice(COLORS)
            for _ in range(k):
                x = rng.randint(40, size - 200); y = rng.randint(40, size - 200)
                _shape(d, "circle", (x, y, x + 150, y + 150), rgb)
            q, ans, c = "How many circles are in the image? Answer with a number.", str(k), "reasoning"
        elif cat == "reason_shape":
            kind = rng.choice(SHAPES); cn, rgb = rng.choice(COLORS)
            _shape(d, kind, (size * .25, size * .25, size * .75, size * .75), rgb)
            q, ans, c = "What shape is shown (circle, square, or triangle)? Answer one word.", kind, "reasoning"
        elif cat == "reason_spatial":
            (cl, rl), (cr, rr) = rng.sample(COLORS, 2)
            _shape(d, "circle", (size * .08, size * .35, size * .38, size * .65), rl)
            _shape(d, "square", (size * .62, size * .35, size * .92, size * .65), rr)
            q, ans, c = "What colour is the shape on the LEFT? Answer one word.", cl, "reasoning"
        elif cat == "reason_size":
            cn, rgb = rng.choice(COLORS); big_left = rng.random() < 0.5
            rb, rs = (int(size * .34), int(size * .16))
            if big_left:
                _shape(d, "circle", (size * .05, size * .3, size * .05 + 2 * rb, size * .3 + 2 * rb), rgb)
                _shape(d, "circle", (size * .7, size * .4, size * .7 + 2 * rs, size * .4 + 2 * rs), rgb)
            else:
                _shape(d, "circle", (size * .1, size * .4, size * .1 + 2 * rs, size * .4 + 2 * rs), rgb)
                _shape(d, "circle", (size * .55, size * .3, size * .55 + 2 * rb, size * .3 + 2 * rb), rgb)
            q, ans, c = "Which circle is bigger, the one on the left or the right? Answer left or right.", ("left" if big_left else "right"), "reasoning"
        elif cat == "agentic_status":
            red = rng.random() < 0.5; rgb = (200, 30, 30) if red else (30, 160, 30)
            d.rectangle((size * .2, size * .2, size * .8, size * .8), fill=rgb)
            q = ("A monitoring panel shows a status light. POLICY: if the light is red, call the tool "
                 "halt; if it is green, call the tool proceed. Which tool do you call? Answer one word.")
            ans, c = ("halt" if red else "proceed"), "agentic"
        else:  # agentic_gauge
            fill = rng.choice([20, 35, 65, 80]); d.rectangle((size * .1, size * .45, size * .9, size * .55), outline=(0, 0, 0), width=6)
            d.rectangle((size * .1, size * .45, size * .1 + (size * .8) * fill / 100, size * .55), fill=(40, 60, 200))
            q = (f"A load gauge is shown. POLICY: if the bar is more than half full, call scale_up; "
                 f"otherwise call scale_down. Which tool do you call? Answer one word.")
            ans, c = ("scale_up" if fill > 50 else "scale_down"), "agentic"
        tasks.append((im, q, ans.lower(), c))
    return tasks


def boot_ci(xs, B=10000, seed=0):
    if not xs:
        return [0.0, 0.0]
    r = random.Random(seed); m = sorted(sum(r.choice(xs) for _ in range(len(xs))) / len(xs) for _ in range(B))
    return [round(m[int(.025 * B)], 3), round(m[int(.975 * B)], 3)]


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
    key = name.lower().replace("-", "").replace(".", "").replace("_", "")
    for cls in ["Qwen3VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration", "Qwen2VLForConditionalGeneration",
                "Gemma3ForConditionalGeneration"]:
        tag = cls.split("For")[0].lower().replace("_", "")
        if hasattr(T, cls) and tag in key:
            return getattr(T, cls).from_pretrained(name, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
    return T.AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct"); ap.add_argument("--tag", default=None)
    ap.add_argument("--n", type=int, default=120); ap.add_argument("--size", type=int, default=1008)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = load_vlm(args.model)
    img_tok = getattr(model.config, "image_token_id", None) or getattr(model.config, "image_token_index", None)
    print(f"loaded {args.model}; image_token_id={img_tok}", flush=True)

    def build(img, q):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return proc(text=[text], images=[img], return_tensors="pt").to("cuda")

    cats = {}
    tasks = gen_tasks(args.n, args.size); ntok = 0
    for i, (img, q, ans, c) in enumerate(tasks):
        inp = build(img, q); ids = inp["input_ids"]; L = ids.shape[1]
        m = (ids[0] == img_tok).nonzero().squeeze(-1); a, b = int(m[0]), int(m[-1]) + 1; ntok = b - a
        gen = model.generate(**inp, max_new_tokens=4, do_sample=False)
        full_txt = proc.batch_decode(gen[:, L:], skip_special_tokens=True)[0].lower()
        full_cache = model(**inp, use_cache=True).past_key_values
        try:
            pids, _ = model.get_rope_index(ids, inp.get("image_grid_thw"), attention_mask=torch.ones_like(ids))
        except Exception:
            pids = None
        def fwd(seg, past, start):
            kw = dict(input_ids=seg, past_key_values=past, use_cache=True,
                      cache_position=torch.arange(start, start + seg.shape[1], device="cuda"))
            if pids is not None:
                kw["position_ids"] = pids[:, :, start:start + seg.shape[1]]
            return model.model(**kw) if hasattr(model, "model") else model(**kw)
        pre = fwd(ids[:, :a], DynamicCache(), 0).past_key_values
        spliced = cache_concat(pre, cache_slice(full_cache, a, b))
        o = fwd(ids[:, b:L], spliced, b); past = o.past_key_values; pos = L
        dec = []
        for _ in range(4):
            lg = o.logits if hasattr(o, "logits") else model.lm_head(o.last_hidden_state)
            nx = int(lg[0, -1].argmax()); dec.append(nx)
            o = fwd(torch.tensor([[nx]], device="cuda"), past, pos); past = o.past_key_values; pos += 1
        pre_txt = proc.tokenizer.decode(dec, skip_special_tokens=True).lower()
        fc = ans in full_txt; pc = ans in pre_txt
        cats.setdefault(c, {"full": [], "pre": [], "agree": []})
        cats[c]["full"].append(int(fc)); cats[c]["pre"].append(int(pc)); cats[c]["agree"].append(int(fc == pc))
        if i % 24 == 0:
            print(f"  [{i}/{len(tasks)}] {c} ans={ans} full={fc} pre={pc} img_tok={ntok}", flush=True)
    allf = [v for c in cats.values() for v in c["full"]]; allp = [v for c in cats.values() for v in c["pre"]]; alla = [v for c in cats.values() for v in c["agree"]]
    out = {"model": args.model, "n": len(tasks), "img_tokens": ntok, "by_category": {}}
    print(f"\n=== IMAGE KV TRANSPLANT — DIVERSE ({args.model}, n={len(tasks)}, ~{ntok} img tok) ===")
    for c, v in sorted(cats.items()):
        nf = sum(v["full"]) / len(v["full"]); npr = sum(v["pre"]) / len(v["pre"]); ng = sum(v["agree"]) / len(v["agree"])
        out["by_category"][c] = {"n": len(v["full"]), "full": round(nf, 3), "precompiled": round(npr, 3),
                                 "precompiled_ci": boot_ci(v["pre"]), "agreement": round(ng, 3)}
        print(f"  {c:9s} n={len(v['full']):>3} full={nf:.2f} precompiled={npr:.2f} CI{boot_ci(v['pre'])} agree={ng:.2f}")
    out["overall"] = {"full": round(sum(allf) / len(allf), 3), "precompiled": round(sum(allp) / len(allp), 3),
                      "precompiled_ci": boot_ci(allp), "agreement": round(sum(alla) / len(alla), 3), "agreement_ci": boot_ci(alla)}
    print(f"  OVERALL full={out['overall']['full']} precompiled={out['overall']['precompiled']} "
          f"CI{out['overall']['precompiled_ci']} agreement={out['overall']['agreement']} CI{out['overall']['agreement_ci']}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"composable_vision_{tag}.json"), "w"), indent=2)
    print("VISION_DONE")


if __name__ == "__main__":
    main()
