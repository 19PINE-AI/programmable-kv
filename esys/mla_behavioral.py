"""MLA architecture check: does the editkv erratum work on a Multi-head Latent Attention model?

The in_place KV edit is MLA-sensitive (MLA compresses K/V into a shared latent c_KV, so editing a
field's KV needs MLA-aware handling) and the HF custom-modeling path for DeepSeek-V2 is incompatible
with transformers 4.57 (the removed get_usable_length cache API — same blocker as Phi). BUT the
erratum is architecture-agnostic (append-only text) and vLLM ships a *native* MLA implementation,
so we verify editkv's erratum behaviorally on DeepSeek-V2-Lite (MLA) through vLLM: does keeping the
stale field + appending the erratum flip the decision to the oracle answer?

Run: python esys/mla_behavioral.py
"""
import os, json, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import vllm.platforms as _P
_P.builtin_platform_plugins["cuda"] = lambda: "vllm.platforms.cuda.CudaPlatform"
_P._current_platform = None
from vllm import LLM, SamplingParams

POLICY = ("You are a retail support agent. Binding policy: an order may be CANCELLED only if its "
          "order_status is exactly 'pending'. If the status is 'processed' or 'delivered', the order "
          "CANNOT be cancelled and you must DENY the request.")
ERR = ("\n\n[STATE UPDATE] The order_status has just changed to 'processed'. This overrides any earlier "
       "value AND any earlier conclusion. Apply the current value.")


def user_msg(status, erratum=False):
    s = (f"{POLICY}\n\nThe order #W123 current order_status is: {status}." +
         (ERR if erratum else "") +
         "\n\nThe customer asks to cancel order #W123. Per the policy and the order's CURRENT status, "
         "answer with exactly one word — 'cancel' or 'deny'.")
    return s


def first_word(t):
    t = t.strip().lower()
    return "cancel" if "cancel" in t[:20] else ("deny" if ("deny" in t[:20] or "cannot" in t[:20]) else t.split()[:1])


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek-ai/DeepSeek-V2-Lite-Chat"
    llm = LLM(model=model, gpu_memory_utilization=0.42, max_model_len=4096, dtype="bfloat16",
              trust_remote_code=True, enforce_eager=True)
    sp = SamplingParams(max_tokens=8, temperature=0.0)
    cases = {"stale (pending)": user_msg("pending"), "oracle (processed)": user_msg("processed"),
             "erratum (pending + update->processed)": user_msg("pending", erratum=True)}
    outs = llm.chat([[{"role": "user", "content": c}] for c in cases.values()], sp, use_tqdm=False)
    res = {}
    print(f"=== MLA behavioral erratum check on {model} (via vLLM native MLA, chat template) ===")
    for (label, _), o in zip(cases.items(), outs):
        d = first_word(o.outputs[0].text)
        res[label] = d
        print(f"  {label:42s} -> {d}  (raw: {o.outputs[0].text.strip()[:30]!r})")
    ok = (res["oracle (processed)"] == "deny" and res["stale (pending)"] == "cancel"
          and res["erratum (pending + update->processed)"] == res["oracle (processed)"])
    print(f"  erratum matches oracle (MLA): {ok}")
    json.dump({"model": model, "results": res, "erratum_recovers": ok},
              open(os.path.join(os.path.dirname(__file__), "..", "results", "mla_behavioral.json"), "w"), indent=2)
    print("MLA_BEHAVIORAL_DONE")


if __name__ == "__main__":
    main()
