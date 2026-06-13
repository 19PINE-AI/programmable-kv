# Calibration findings (pre-confirmatory), 2026-06-08

Task: gated decision — "proceed only if ALL of n named settings are enabled", value(s) stored
in a Markdown USER MEMORY; balanced gold labels; late layout `[sys][traj][MEM][query]`.

## Oracle (full-recompute) competence
| model | regime | nf=1 | nf=2 | nf=4 |
|---|---|---|---|---|
| Qwen3-0.6B | direct | 0.50 | 0.50 | 0.50 |
| Qwen3-1.7B | direct | 0.50 | 0.50 | 0.50 |
| Qwen3-4B | direct | 0.50 | 0.50 | 0.50 |
| Qwen2.5-1.5B-Instruct | direct | 0.57 | 0.50 | 0.50 |
| Qwen2.5-7B-Instruct | direct | 0.50 | 0.50 | 0.54 |
| Llama-3.1-8B-Instruct | direct | 0.75 | 0.62 | 0.62 |
| gemma-2-2b-it | direct | 0.62 | 0.67 | 0.71 |
| **Qwen3-4B** | **CoT** | **1.00** | **1.00** | **1.00** |
| **Llama-3.1-8B-Instruct** | **CoT** | **0.90** | **1.00** | **0.95** |

## Conclusions that shape the confirmatory design
1. **Direct one-shot memory-gated decisions are at chance for ≤8B models** (constant
   answer: Qwen3 reasoning-native models always emit "yes"; instruct models near floor).
   This is itself a finding: integrating scattered memory facts into a snap one-word
   decision is not reliable without a scratchpad.
2. **Chain-of-thought is the competent, discriminating regime** (≈1.0 oracle accuracy).
   So decision-governance experiments (E1 placement, E3 editing recovery, decision
   agreement) are run under CoT, where oracle decisions vary correctly with gold.
3. **Logit-cosine / top-1 agreement at the decision token is the fast faithfulness
   backstop**: it discriminates faithful vs unfaithful caches even in the floored direct
   regime (it compares the full logit vector to full-recompute), needs no generation, and
   scales to large N across many models. Used as the primary faithfulness endpoint for
   E2/E4; CoT decision-agreement is the decision-governance layer on competent models.
4. **Inclusion gate** (oracle ≥ 0.80) is met under CoT by Qwen3-4B and Llama-3.1-8B; these
   are the primary competent models for decision-level analysis. Smaller models and the
   direct regime are reported as the floored baseline.

GPU: shared node, ~21 GB free → competent decision-level runs use 4B (fast) and 8B
(borderline mem, short memory); faithfulness runs span 0.6B–8B + Gemma sliding-window.
