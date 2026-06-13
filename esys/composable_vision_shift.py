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
from composable_vision import load_vlm, cache_slice, cache_concat, gen_tasks, boot_ci


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], -1)


@torch.no_grad()
def mrope_cossin(rot, position_ids, mrope_section):
    """Assembled M-RoPE cos/sin for 3D position_ids [3,1,L] -> [1,1,L,D] (fp32).
    Handles BOTH layouts: Qwen2.5-VL sectioned (rotary returns 3-component [3,1,L,D], we assemble the
    contiguous t/h/w sections) and Qwen3-VL interleaved (rotary returns already-assembled [1,L,D])."""
    dummy = torch.zeros(1, position_ids.shape[-1], 1, dtype=torch.float32, device="cuda")
    cos, sin = rot(dummy, position_ids)
    if cos.ndim == 4 and cos.shape[0] == 3:        # sectioned (Qwen2.5-VL): assemble t/h/w sections
        ms = [s * 2 for s in mrope_section]
        cos = torch.cat([c[i % 3] for i, c in enumerate(cos.float().split(ms, dim=-1))], dim=-1)
        sin = torch.cat([s[i % 3] for i, s in enumerate(sin.float().split(ms, dim=-1))], dim=-1)
    else:                                          # interleaved (Qwen3-VL): rotary already assembled it
        cos = cos.float(); sin = sin.float()
    return cos[:, None], sin[:, None]              # [1,1,L,D]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct"); ap.add_argument("--tag", default=None)
    ap.add_argument("--n", type=int, default=120); ap.add_argument("--size", type=int, default=672)
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = load_vlm(args.model)
    # text rotary: model.model.rotary_emb (Qwen2.5-VL) or model.model.language_model.rotary_emb (Qwen3-VL)
    rot = getattr(model.model, "rotary_emb", None) or model.model.language_model.rotary_emb
    # mrope_section: top-level rope_scaling (Qwen2.5-VL) or text_config.rope_scaling (Qwen3-VL)
    rs = getattr(model.config, "rope_scaling", None) or getattr(getattr(model.config, "text_config", None), "rope_scaling", None)
    ms = rs["mrope_section"]
    print(f"mrope_section={ms} interleaved={rs.get('mrope_interleaved', False)}", flush=True)
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

    cats = {}; PAD = "Context preface. " * 40  # makes the image land at a later position in context B
    tasks = gen_tasks(args.n, args.size)
    for i, (img, q, ans, c) in enumerate(tasks):
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
        fc = int(ans in full_txt); sc = int(ans in sh_txt)
        cats.setdefault(c, {"full": [], "sh": [], "agree": []})
        cats[c]["full"].append(fc); cats[c]["sh"].append(sc); cats[c]["agree"].append(int(fc == sc))
        if i % 24 == 0:
            print(f"  [{i}/{len(tasks)}] {c} ans={ans} full@B={fc} shifted={sc} | posA_img={aA} posB_img={aB} (Δ={aB-aA})", flush=True)
    allf = [v for d in cats.values() for v in d["full"]]; alls = [v for d in cats.values() for v in d["sh"]]; alla = [v for d in cats.values() for v in d["agree"]]
    out = {"model": args.model, "n": len(tasks), "by_category": {}}
    print(f"\n=== M-RoPE POSITION-SHIFT image transplant — DIVERSE ({args.model}, n={len(tasks)}, {len(cats)} categories) ===")
    for c, v in sorted(cats.items()):
        ag = sum(v["agree"]) / len(v["agree"])
        out["by_category"][c] = {"n": len(v["full"]), "full": round(sum(v["full"]) / len(v["full"]), 3),
                                 "shifted": round(sum(v["sh"]) / len(v["sh"]), 3), "agreement": round(ag, 3)}
        print(f"  {c:9s} n={len(v['full']):>3} full@B={out['by_category'][c]['full']:.2f} shifted={out['by_category'][c]['shifted']:.2f} agree={ag:.2f}")
    out["overall"] = {"full_at_B": round(sum(allf) / len(allf), 3), "shifted": round(sum(alls) / len(alls), 3),
                      "agreement": round(sum(alla) / len(alla), 3), "agreement_ci": boot_ci(alla), "shifted_ci": boot_ci(alls)}
    print(f"  OVERALL full@B={out['overall']['full_at_B']} shifted={out['overall']['shifted']} CI{out['overall']['shifted_ci']} "
          f"agreement={out['overall']['agreement']} CI{out['overall']['agreement_ci']}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", f"vision_shift_{tag}.json"), "w"), indent=2)
    print("VISION_SHIFT_DONE")


if __name__ == "__main__":
    main()
