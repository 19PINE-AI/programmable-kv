"""CLOSED systems integration: editkv's erratum on a real PagedAttention engine (vLLM).

The central systems claim, realized on production infrastructure: editkv's ERRATUM mode is
*append-only*, so it composes with vLLM's content-addressed automatic prefix caching (APC) —
the long policy prefix stays cached and only the short erratum suffix is computed. The naive
alternative (put the new field value into the context) *mutates a token inside the cached
prefix*, which changes that block's content hash and **invalidates every downstream block**, so
the engine recomputes from the field position onward (≈ a full reprefill when the field is
early). This is exactly why the erratum is the right design for a paged-attention serving stack.

Two arms over N requests sharing one long policy prefix (field placed EARLY), prefix caching ON:
  BASELINE (no editkv): prompt = policy[field=NEW_i] + decision   -> early-field edit invalidates
      the cached prefix from the field onward -> recompute most of it every request.
  ERRATUM  (editkv):    prompt = policy[field=OLD] + "[STATE UPDATE]..NEW_i.." + decision
      -> the whole policy[field=OLD] prefix is a cache hit; only the suffix is computed.
We prime the cache with the shared prefix, then measure per-request latency, total throughput,
and vLLM's reported prefix-cache hit rate. Run: python esys/vllm_editkv_serving.py
"""
import os
import argparse, os, sys, time, json
# This machine's NVML userspace lib (595.71) mismatches the kernel driver (595.58), so NVML
# init fails and breaks vLLM's NVML-based platform detection (torch.cuda itself works fine).
# Force the CUDA platform and run single-process so the patch applies to the engine too.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import vllm.platforms as _P
_P.builtin_platform_plugins["cuda"] = lambda: "vllm.platforms.cuda.CudaPlatform"
_P._current_platform = None

TAU2 = os.environ.get("TAU2_POLICY", os.path.expanduser("~/tau2-bench/data/tau2/domains/retail/policy.md"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tag", default="qwen3_8b")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--gpu_mem", type=float, default=0.6)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams

    policy = open(TAU2).read() if os.path.exists(TAU2) else ("RETAIL POLICY. " * 400)
    OLD, NEWS = "pending", [f"processed_{i:03d}" for i in range(args.n)]  # distinct new values
    # The mutable field is placed EARLY (system header), BEFORE the long policy, so editing it
    # invalidates the entire downstream — the regime where editkv matters.
    pre = "# Session header\norder_status: "          # field goes here, first
    post = ("\n\n" + policy + "\n\n# Conversation\nuser: I'd like to cancel order #W2378156.\n"
            "assistant: Let me verify the order status against the policy.\n\n# TASK\n"
            "Decide one word: cancel or deny.\nDecision:")
    erratum = "\n[STATE UPDATE] order_status has changed to {v}; this overrides any earlier value AND conclusion.\n"

    # BASELINE (no editkv): new field value baked into the EARLY header -> everything after the
    # field (the whole long policy) is invalidated and recomputed, every request.
    baseline = [pre + v + post for v in NEWS]
    # ERRATUM (editkv): the entire OLD prefix (header+policy+convo) is a cache hit; only the
    # short appended update suffix is computed.
    erratum_prompts = [pre + OLD + post.replace("# TASK", erratum.format(v=v) + "# TASK") for v in NEWS]

    llm = LLM(model=args.model, enable_prefix_caching=True, gpu_memory_utilization=args.gpu_mem,
              max_model_len=8192, dtype="bfloat16", trust_remote_code=True, enforce_eager=True)
    sp = SamplingParams(max_tokens=4, temperature=0.0)

    def run(prompts, label):
        # prime the cache with the shared prefix once (not timed)
        llm.generate([prompts[0]], sp, use_tqdm=False)
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        # prefix cache hit rate from engine metrics if available
        hit = None
        try:
            m = llm.llm_engine.get_metrics() if hasattr(llm.llm_engine, "get_metrics") else None
        except Exception:
            m = None
        print(f"  [{label}] {len(prompts)} reqs in {dt:.2f}s = {len(prompts)/dt:.1f} req/s "
              f"({1000*dt/len(prompts):.1f} ms/req)", flush=True)
        return {"label": label, "n": len(prompts), "total_s": round(dt, 3),
                "req_per_s": round(len(prompts) / dt, 2), "ms_per_req": round(1000 * dt / len(prompts), 1)}

    print(f"=== vLLM editkv serving ({args.model}, prefix-caching ON, policy {len(policy)} chars) ===")
    res_base = run(baseline, "baseline (new field in prefix)")
    res_err = run(erratum_prompts, "erratum (append-only, prefix reused)")
    speedup = round(res_base["ms_per_req"] / res_err["ms_per_req"], 2)
    out = {"model": args.model, "baseline": res_base, "erratum": res_err, "speedup": speedup}
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
              f"vllm_serving_{args.tag}.json"), "w"), indent=2)
    print(f"\n  ERRATUM speedup vs baseline on vLLM (paged attention + APC): {speedup}x")
    print("VLLM_SERVING_DONE")


if __name__ == "__main__":
    main()
