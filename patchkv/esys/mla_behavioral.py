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

POLICY = ("You are a retail support agent. POLICY (binding): an order can be cancelled ONLY IF its "
          "order_status is 'pending'; if it is 'processed' or 'delivered' it CANNOT be cancelled and "
          "you must deny.\n\n")
ERR = "[STATE UPDATE] order_status has changed to processed; this overrides any earlier value AND conclusion.\n"
TASK = ("\nuser: Please cancel my order, ordered by mistake.\nassistant: Let me check the policy and the "
        "current order status.\nDecide one word — cancel or deny.\nDecision:")


def prompt(status, erratum=False):
    s = POLICY + f"# Session\norder_status: {status}\n"
    if erratum:
        s += ERR
    return s + TASK


def first_word(t):
    t = t.strip().lower()
    return "cancel" if "cancel" in t[:12] else ("deny" if ("deny" in t[:12] or "cannot" in t[:12]) else t.split()[:1])


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek-ai/DeepSeek-V2-Lite-Chat"
    llm = LLM(model=model, gpu_memory_utilization=0.55, max_model_len=4096, dtype="bfloat16",
              trust_remote_code=True, enforce_eager=True)
    sp = SamplingParams(max_tokens=6, temperature=0.0)
    prompts = {"stale (pending)": prompt("pending"), "oracle (processed)": prompt("processed"),
               "erratum (pending + update->processed)": prompt("pending", erratum=True)}
    outs = llm.generate(list(prompts.values()), sp, use_tqdm=False)
    res = {}
    print(f"=== MLA behavioral erratum check on {model} (via vLLM native MLA) ===")
    for (label, _), o in zip(prompts.items(), outs):
        d = first_word(o.outputs[0].text)
        res[label] = d
        print(f"  {label:42s} -> {d}")
    ok = (res["oracle (processed)"] == "deny" and res["stale (pending)"] == "cancel"
          and res["erratum (pending + update->processed)"] == res["oracle (processed)"])
    print(f"  erratum matches oracle (MLA): {ok}")
    json.dump({"model": model, "results": res, "erratum_recovers": ok},
              open(os.path.join(os.path.dirname(__file__), "..", "results", "mla_behavioral.json"), "w"), indent=2)
    print("MLA_BEHAVIORAL_DONE")


if __name__ == "__main__":
    main()
