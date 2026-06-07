"""H1 — Multimodal TTFT savings: reusing a cached image KV vs re-encoding (vision tower + image prefill).

Full path per turn: run the vision tower + prefill the image's soft-tokens + text -> first-token logits.
Precompiled path: image KV cached once (offline); per turn only prefill text + splice the image KV.
We time both (median, CUDA-synced) across image sizes (=> token counts) and report the speedup —
the practical win of image-KV reuse. Run: python esys/vision_ttft.py --model Qwen/Qwen2.5-VL-7B-Instruct
"""
import argparse, os, sys, json, time
import torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoProcessor
from transformers.cache_utils import DynamicCache
from composable_vision import load_vlm, cache_slice, cache_concat


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct"); ap.add_argument("--tag", default=None)
    ap.add_argument("--sizes", default="448,672,1008,1344"); ap.add_argument("--trials", type=int, default=8)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = load_vlm(args.model)
    img_tok = getattr(model.config, "image_token_id", None) or getattr(model.config, "image_token_index", None)
    q = "Describe the image and decide the next action."

    def build(img):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return proc(text=[text], images=[img], return_tensors="pt").to("cuda")

    def med(f, trials):
        ts = []
        for _ in range(trials):
            torch.cuda.synchronize(); t0 = time.perf_counter(); f(); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        return sorted(ts)[len(ts) // 2]

    out = {"model": args.model, "by_size": {}}
    print(f"=== MULTIMODAL TTFT: reuse cached image KV vs re-encode ({args.model}) ===")
    print(f"  {'img_px':>6} {'img_tok':>7} {'full_ms':>9} {'reuse_ms':>9} {'speedup':>8}")
    for px in [int(x) for x in args.sizes.split(",")]:
        img = Image.new("RGB", (px, px), (60, 120, 200))
        inp = build(img); ids = inp["input_ids"]; L = ids.shape[1]
        m = (ids[0] == img_tok).nonzero().squeeze(-1); a, b = int(m[0]), int(m[-1]) + 1
        try:
            pids, _ = model.get_rope_index(ids, inp.get("image_grid_thw"), attention_mask=torch.ones_like(ids))
        except Exception:
            pids = None
        # cache the image KV once (offline; excluded from per-turn timing)
        full_cache = model(**inp, use_cache=True).past_key_values
        img_kv = cache_slice(full_cache, a, b)

        def full():  # vision tower + full prefill (everything recomputed)
            model(**inp, use_cache=True)

        def reuse():  # only text prefill + splice cached image KV (no vision tower, no image prefill)
            def fwd(seg, past, start):
                kw = dict(input_ids=seg, past_key_values=past, use_cache=True,
                          cache_position=torch.arange(start, start + seg.shape[1], device="cuda"))
                if pids is not None:
                    kw["position_ids"] = pids[:, :, start:start + seg.shape[1]]
                return (model.model(**kw) if hasattr(model, "model") else model(**kw))
            pre = fwd(ids[:, :a], DynamicCache(), 0).past_key_values
            spliced = cache_concat(pre, img_kv)
            fwd(ids[:, b:L], spliced, b)

        # warmup
        full(); reuse()
        f_ms = med(full, args.trials); r_ms = med(reuse, args.trials)
        out["by_size"][px] = {"img_tokens": b - a, "full_ms": round(f_ms, 1), "reuse_ms": round(r_ms, 1),
                              "speedup": round(f_ms / r_ms, 2)}
        print(f"  {px:>6} {b-a:>7} {f_ms:>9.1f} {r_ms:>9.1f} {f_ms/r_ms:>7.2f}x", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"vision_ttft_{tag}.json"), "w"), indent=2)
    print("VISION_TTFT_DONE")


if __name__ == "__main__":
    main()
