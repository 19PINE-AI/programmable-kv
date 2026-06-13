# OUTLINE — "Models Take Notes at Prefill: KV Cache Can Be Editable and Composable"

Target: general ML venue (NeurIPS-style). Mechanism-led hook; edit + compose as co-headline capabilities.
Spine: a single discovery (memoized inference) generates two capabilities (edit, compose), unified by the keystone.
All numbers from `results/*.json` (local runs, 1x RTX PRO 6000 96GB). Do NOT reuse PAPER.md prose; write fresh.

## Title / Authors / Abstract
- Title: Models Take Notes at Prefill: KV Cache Can Be Editable and Composable
- Abstract (~180 words): puzzle (overwrite field KV -> model ignores it) -> discovery (conclusions memoized
  onto aggregator tokens at prefill; <1% direct effect; 4 causal methods) -> two consequences: EDIT
  (erratum, not recompute; field+erratum matches hoist-to-end; reasoning recovers at ~1% compute) and
  COMPOSE (notes are position-portable -> transplant a skill, O(L) not O(L^2), 13.9x) -> KEYSTONE (edit
  inside a transplant -> one substrate) -> reach (12 models, MoE/FP8/70B, multimodal, MLA/sliding-window,
  to the 2026 sparse-attention frontier) + payoff (tau2-bench, 16x vLLM). Honest: caching machinery is
  prior art (EPIC/CacheBlend/CacheSlide/MPIC); our contributions are the mechanism, the unification, the
  correctness lens, and the attention-variant adapters.

## §1 Introduction  [Fig 1: teaser — the note-taking schematic + the "edit ignored" bars]
- Hook: agents re-read static instructions every turn; prefix caching only reuses exact prefixes; one
  changed field invalidates the suffix. The field-hoisting workaround taxes programmability.
- The puzzle: you CAN overwrite a field's KV in place (region before it is provably reusable, dev=0.0),
  yet the model still acts on the OLD value. Why?
- The discovery (one sentence): at prefill the model computes the field-conditioned CONCLUSION and writes
  it onto downstream aggregator/delimiter tokens; the decision reads those notes, not the field.
- Consequences as the paper's spine: (i) editing must amend the notes (erratum), (ii) the notes are
  localized/position-portable/context-robust => composable (transplant), (iii) one substrate (keystone).
- Contributions list (mechanism, edit, compose, keystone, reach+frontier, systems, honest positioning).

## §2 Related work
- Interpretability of stored computation: ROME/MEMIT (weight edits), IOI circuits, Biology of an LLM
  (delimiter/planning tokens). We locate memoized *inference* in the KV cache (activations), not weights.
- KV reuse / composable caching (prior art we build on, claim none of the mechanism novel): Prompt Cache,
  CacheBlend (selective ~15% recompute), EPIC (PIC + AttnLink boundary recompute), CacheSlide (RPDC
  position-aware), MPIC (multimodal PIC), KVLink. Our adds: the mechanism that explains why boundary
  recompute works; a correctness/decision-governance lens; attention-variant adapters.
- KV editing/streaming: prefix caching (vLLM APC, SGLang), StreamingLLM/sinks, eviction (H2O/SnapKV).

## §3 The discovery: memoized inference in the KV cache  [Fig 2: mechanism — 4-panel causal evidence]
- Probe setup: a gated decision (policy rule + a mutable field => correct action flips). Define stale /
  field_only / full_downstream / oracle.
- Method 1 (KV-patching / locality): patching only the field's KV recovers ~0% of the decision
  (field_only_recovery -0.028 on Llama-3.1-8B; <1% direct effect), while patching the downstream recovers
  1.0. -> the field is read INDIRECTLY.  [data: mech_causal_patch_*]
- Method 2 (suffix concentration): the causal mass is concentrated on mid/late layers and on the
  suffix/aggregator tokens after the field (locality_topk grows slowly; the conclusion lives downstream).
  [mech_causal_patch locality_topk + mech_attention]
- Method 3 (linear probe): the conclusion is linearly decodable from downstream tokens at prefill.
- Method 4 (reasoning-circuit knockout + position dose-response): knocking the memoized tokens flips the
  decision; effect rises with distance/position.  [dose_response data]
- Synthesis: "attention-mediated memoized inference" — connect to Biology of an LLM (aggregator tokens).
- What the note contains (wording ablation, why_erratum_8b): override wording redundant once the value is
  present; "re-evaluate" phrasing hurts (0.81) -> the note is a committed conclusion, not raw text.

## §4 Consequence I — the cache is EDITABLE  [Fig 3: edit — naive-fail/erratum bars + scale-reversal]
- Naive in-place edit fails (refresh field, leave rest stale): decision reverts to old value (consistent
  with §3). The region before the field is reusable (dev 0.0); the suffix is not.
- The robust fix: a salience ERRATUM ("[STATE UPDATE] field->new; overrides earlier value and conclusion")
  appended; field+erratum matches the strong hoist-to-end baseline's oracle correctness WITHOUT prompt
  surgery.  [arch_erratum_v2 reasoning recovery ~0.97]
- The reasoning twist + scale-reversal: under CoT the cheap in_place edit ALONE recovers (8B 0.94 at ~1%
  recompute) because the chain re-reads the field; non-reasoning never (0.00); larger sticky (14B/32B
  partial). field+selective@K: minimal-K sweep, K* model-dependent (4 @8B ... >64 @4B). Honest: an
  unreliable-but-sometimes tool.  [ksweep_diverse_*]
- Baseline frontier (condensed): hoist (cheap, needs surgery) vs in_place (free under reasoning) vs
  field+erratum (no surgery) vs CacheBlend@k. No single winner; a frontier. Full table -> appendix.

## §5 Consequence II — the cache is COMPOSABLE  [Fig 4: compose — TTFT O(L) + transplant fidelity bars]
- The substrate prediction: memoized notes are localized, position-portable, context-robust => precompile
  a SKILL once, RoPE-reposition, splice -> behaviorally indistinguishable from full recompute.
- Fidelity: logit cos 0.90-0.999 across the full model family; 24/24 correct skill-following on competent
  models across 8 domains; 16/16 under CoT.  [composable_kv / compose results]
- Context-robustness + seam: isolation-precompiled chunk matches context-attended, except a boundary SEAM
  -> seam-repair (this IS EPIC/CacheBlend boundary recompute; we explain WHY via §3).
- TTFT O(L) vs O(L^2): 13.9x @32k (3x@2k, 9.8x@8k).  [composable_scaling_*]
- Generality of content/insertion/agentic (condensed; full scorecards -> appendix): rules + facts/RAG;
  system-area + end-of-trajectory tool results; actual tool-calling preserved 1.0 (N=108+CI).

## §6 One substrate: the keystone and the unified agent  [Fig 5: keystone bars + unified-agent panel]
- KEYSTONE: edit a field INSIDE a transplanted skill -> the editable mechanism reproduces verbatim
  (in_place weak ~0.05, selective recovers ~0.8, erratum strongest; composed approx recomputed).
  [compose_edit_*]  This is the proof edit and compose act on ONE object.
- The unified edit+compose agent: precompile policy once + append state errata over turns + longest-prefix
  reuse. Across 10 domains x 100 trajectories x 12 models: agreement 0.81-1.0 (Llama 0.963, Mistral 0.983,
  70B 1.0), speedup up to 14.9x.  [agent_rigorous_*]

## §7 Reach and the 2026 frontier  [Fig 6: scope matrix + scale/multimodal panels]
- Scale/quant/MoE: feasibility holds 0.6B->32B, FP8, 30B-A3B MoE, 70B (4-bit).
- Multimodal: image notes are position-portable too; cache image KV once, splice (skip vision tower +
  >1k-token prefill). Diverse N=120 across Qwen2.5-VL-3B/7B/32B + Qwen3-VL-8B: agreement 0.958-1.0; TTFT
  2.4-8.4x; M-RoPE position-shift (sectioned + interleaved) lossless (0.99).  [composable_vision*, vision_shift*]
- Attention-variant scope map (the durability claim): substrate = any per-token attention KV.
  - Free: FlashAttn, paged/vLLM, GQA/MQA (tested).
  - Adapter (implemented+validated): MLA decoupled-k_pe (DeepSeek-V2/Coder-V2, cos 0.98, agreement 1.0 on
    Coder); interleaved M-RoPE (Qwen3-VL).
  - Fixed: sliding-window (full-cache + masked window -> Gemma H5 0.93-0.94; chunk>window edge case).
  - Partial: hybrids (Falcon-H1: attention KV transplantable, Mamba scan-state not).
  - Open frontier: DeepSeek-V4 CSA/HCA (sequence-dim compression -> block-granular); DSA(V3.2)=MLA+top-K
    (inherits our MLA adapter). Out of scope: RWKV/Mamba/diffusion (no per-token attention KV).

## §8 Systems payoff  [Fig 7: vLLM throughput + tau2-bench]
- tau2-bench retail: single-decision + multi-turn agent; stale agent collapses, editkv preserves task
  success at a fraction of recompute.  [tau2 results]
- Closed vLLM integration: append-only erratum composes with prefix caching -> 16x throughput; online-load
  study.  [vllm_serving / vllm_online_load]

## §9 Limitations
- chunk-exceeds-window edge; hybrids partial; sequence-dim compression open; cross-attention VLMs out of
  in-sequence scope; field+selective unreliable; synthetic policies (mitigated by real tau2 + real images).

## §10 Conclusion
- The KV cache is a notebook of memoized conclusions: edit the notes, paste the notes, and (keystone) edit
  pasted notes. One mechanism, two capabilities, one substrate; holds to the 2026 frontier.

## Figures (NeurIPS style; vector PDF; colorblind palette; tight)
- Fig 1 teaser: schematic (field->aggregator notes->decision) + "edit ignored" mini-bars.
- Fig 2 mechanism: (a) field-only vs full-downstream recovery; (b) locality top-k curve; (c) layer x
  position attribution heatmap; (d) wording ablation.
- Fig 3 editable: (a) naive vs erratum vs hoist; (b) scale-reversal field-only recovery vs model size;
  (c) K-sweep K* per model.
- Fig 4 composable: (a) TTFT full O(L^2) vs precomp O(L) + speedup; (b) transplant logit-cos / agreement
  across models.
- Fig 5 keystone+agent: (a) in_place/sel/erratum composed-vs-recomputed bars; (b) unified-agent agreement
  + speedup across 12 models.
- Fig 6 reach: (a) attention-variant scope matrix (free/adapter/fixed/partial/open/out); (b) multimodal
  agreement across VL models + TTFT vs image tokens.
- Fig 7 systems: (a) vLLM throughput baseline vs erratum (16x); (b) tau2 success vs recompute.

## Files
- paper/main.tex (arxiv.sty), paper/arxiv.sty, paper/references.bib, paper/figs/make_figures.py,
  paper/sections/{intro,related,mechanism,editable,composable,keystone,reach,systems,limitations,conclusion}.tex
