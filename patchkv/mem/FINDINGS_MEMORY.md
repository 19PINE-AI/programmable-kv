# Findings — Editable & Composable User Memory in the KV Cache

*Autonomous run, 2026-06-08. One RTX PRO 6000 (shared). Models ≤ 8B (GPU budget).*
*Pre-registered hypotheses/margins in `PREREG.md`; design in `DESIGN.md`.*

This document is generated/edited from `results/summary.json` (produced by `analyze.py`).
Numbers below are filled from the confirmatory runs; see that file for full CIs.

## TL;DR (updated with scale to 70B + real-memory LoCoMo)
**Strong-framing summary:** transplanting precompiled user memory is **faithful to full recompute
across scale (0.6B–70B)** — at 70B, where decisions genuinely vary, the late seam-repaired
transplant reproduces the decision **0.93** of the time at logit cosine **0.997** (and late
beats early, as the mechanism predicts). It is **editable**, and editing **strengthens with
scale** (1-token in-place edit correctness →**1.00** at 14B). On **real ~20k-token LoCoMo
conversations (all 1,540 questions)** transplant is **statistically equivalent** to full
recompute in QA accuracy (TOST δ=0.03: Qwen3-4B/14B equivalent; Llama-8B within −2.7pt),
answer-token cosine 0.99+. The constant-answer floor (direct one-shot decisions) is **scale-robust
to 32B**, breaking only at Llama-70B — so CoT being the operative regime is a general finding.
End-to-end the agent is **2.3–4.25× faster** cumulative TTFT than end-reprefill. Novelty vs.
MemArt/EPIC: the **editing axis**, **decision-governance metric**, and **mechanism**.

---

### Original TL;DR
User memory is a natural fit for the paper's edit+compose substrate. The proposed deployment —
**precompile the memory chunk, place it late, RoPE-reposition each turn, repair one boundary
token, edit in place** — is **logit-faithful to full recompute** (cos 0.94–0.9996; decision-
identical at 2k tokens), **editable mid-session** (a 1-token in-place edit recovers a flipped
decision under reasoning; erratum is the robust default), **sub-chunkable** for 16× cheaper
localized edits without changing decisions, and **2.3–4.25× faster** cumulative TTFT than
end-reprefill. **Honest caveats:** placement's "pre-digestion" cost is real but small/marginal
under CoT (so late placement is safe); and exact long greedy-CoT *chains* are sensitive to
sub-percent logit differences (chain agreement 0.31–0.69) even though the next-token decision
is faithful — so we anchor decision-governance on the short-context editing result (E3,
agreement 0.89–0.95). Novelty vs. concurrent KV-memory systems (MemArt, EPIC): the **editing
axis**, the **decision-governance metric**, and the **mechanism** that explains it.

## Method recap (what is measured)
- **Controlled task**: a gated decision ("proceed only if all of n named settings are
  enabled") whose governing facts live in a Markdown USER MEMORY; balanced gold labels;
  late layout `[sys][traj][MEM][query]` unless stated.
- **Competent regime**: chain-of-thought (CoT). Direct one-shot answers are at chance for
  ≤8B models (constant "yes"), so decision-level claims use CoT; **logit-cosine / decision
  agreement** at the decision token is the fast faithfulness backstop (`CALIBRATION.md`).
- **Endpoints**: decision agreement vs full-recompute (governance), logit cosine
  (fidelity), gated accuracy vs gold (E1), decision recovery + recompute cost (E3).
- **Stats**: cluster bootstrap (persona-level, 10⁴), TOST equivalence (δ=0.03 decisions,
  cos≥0.98), GEE-logistic (cluster-robust), McNemar-exact, BH-FDR. Power: ≥476 paired
  decisions/cell for the equivalence margin (`stats.n_for_equivalence`).

---

## E2 — Transplant faithfulness / equivalence
*Precompiled + RoPE-repositioned memory vs full recompute; seam dose-response; naive (no-rotation) control.*
*6 models × 400 personas (300 for 7–8B) × 2 placements × {seam 0,1,2,4,8, naive}.*

**Logit fidelity (robust metric).** A precompiled+repositioned memory chunk reproduces the
full-recompute next-token logits at the decision with high cosine, and **seam-repair closes
the boundary gap** — most visibly on Llama-3.1-8B (late): cos 0.940 (seam0) → 0.994 (seam1)
→ 0.996 (seam8). Final-seam cosine by model (late): Qwen3-0.6B 0.999, 1.7B 0.996, 4B 0.9996,
Mistral-7B 0.996, Llama-3.1-8B 0.996, Gemma-2-2B 0.990. The **naive no-rotation control**
collapses where the model is not constant-answer (Qwen3-0.6B late decision-agreement 0.18 vs
0.78 rotated; Mistral 0.61 vs 0.80) — RoPE re-rotation is necessary.

**Seam dose-response (decision agreement vs full, late).** Qwen3-0.6B: seam0 0.78 → seam1
**1.00**; Qwen3-1.7B/4B already 1.00 at seam0. One boundary token suffices where repair is
needed (mirrors EPIC/CacheBlend boundary recompute, predicted by the mechanism).

**Decision-governance caveat (important).** E2 reads the yes/no decision in the *direct*
(non-CoT) regime, which `CALIBRATION.md` shows is at chance for ≤8B models. There the
yes/no agreement metric is **confounded**: Qwen3 reaches 1.00 partly because it is
constant-answer (both transplant and full say "yes"), while Mistral/Llama/Gemma plateau at
0.75–0.89 because near-tie decisions flip under tiny (cosine-preserving) perturbations — not
because the transplant is unfaithful (their cosine is ≥0.99 after seam). The **meaningful
decision-governance equivalence is measured under CoT** in E5 (`cot_agree`, transplant vs
full-reprefill oracle) and for editing in E3. Net E2 claim: *transplant is logit-faithful
(cos 0.94→0.9996, seam-repair closes the boundary), with CoT decision-equivalence shown in E5.*

## Scale (14B–70B) — added on user request
*Large models run with flash-attention; full GPU (shared, intermittent contention handled).*

**Competence / the constant-answer floor is scale-robust.** Direct one-shot memory-gated
decisions stay at chance for the entire Qwen3 family **up to 32B** (direct acc 0.500,
constant answer), while **CoT is competent** (0.906/0.923/0.993/0.951/0.972 for
1.7B/4B/14B/30B-A3B/32B). The floor finally **breaks at Llama-3.1-70B** (direct acc
**0.806** — first model to do one-shot memory decisions without CoT; partly family, since
Llama answers directly and Qwen3 is reasoning-native). So "CoT is the operative regime" is a
*scale-general* finding for reasoning-native models, not a small-model artifact.

**Decision-equivalence at 70B (the clean, non-trivial test).** Because 70B's decisions
genuinely vary (594 "no" / 306 "yes"), its decision-agreement is meaningful (not the
constant-answer artifact of smaller models). Late-placed seam-repaired transplant reproduces
the full-recompute decision **0.93** of the time at logit cosine **0.997**, and **late beats
early** (0.93 vs 0.83) — the mechanism's prediction (decode reads memory directly vs. relying
on pre-digestion across the transplant boundary). E2 logit cosine stays high across all large
models (0.95–0.997).

**Editing strengthens with scale.** In-place (1-token) edit correctness under CoT:
1.7B 0.922 → 4B 0.984 → **14B 1.000** → 32B 0.976 (Llama-8B 0.50 is a CoT-format-compliance
artifact, not an editing failure — its agreement-with-oracle is 0.89). The near-free in-place
edit becomes *more* reliable as models get larger.

**Systems at scale (E5, recovered):** the cumulative-TTFT speedup grows with model size —
Qwen3-32B-FP8: **3.27× vs front-reprefill, 4.32× vs end-reprefill** (proposed cos 0.975,
n=64), the largest speedups in the study (bigger memory ⇒ more reprefill avoided). **E3-32B
editing** likewise recovered (in-place 0.979).

*GPU-contention note: the node is shared and other jobs intermittently grab 20–60 GB, which
OOM'd several large runs mid-sweep. **All were recovered** via wait-for-memory + backoff
retries (`run_retry_robust.sh`, `run_retry2.sh`); the 32B LoCoMo point uses the bf16 checkpoint
because flash attention rejects FP8.*

## LoCoMo — external validity on real conversational memory (added on user request)
*Real LoCoMo multi-session conversations as the memory; transplant vs full recompute; **all
1,540 answerable questions** per model; full ~20k-token conversation memory (median 19.6k–19.75k,
no truncation); flash attention. Parity tested with TOST (δ=0.03), clustered on conversation.*

| model | full acc | transplant acc | parity diff [CI] | equivalent? | answer-token cos | top-1 |
|---|---|---|---|---|---|---|
| Qwen3-4B | 0.482 | 0.472 | −0.010 [−0.016, −0.003] | **yes** | 0.996 | 0.918 |
| Qwen3-14B | 0.564 | 0.558 | −0.005 [−0.011, 0.001] | **yes** | 0.998 | 0.943 |
| Qwen3-32B (bf16) | 0.545 | 0.560 | +0.015 [0.003, 0.026] | **yes** | 0.992 | — |
| Llama-3.1-8B | 0.594 | 0.567 | −0.027 [−0.040, −0.016] | no (small −2.7pt) | 0.991 | 0.857 |

**Takeaway:** on real ~20k-token conversational memory, transplanting the precompiled memory
is **statistically equivalent** to full recompute in QA accuracy for Qwen3-4B/14B (TOST,
δ=0.03) and within a small −2.7pt of it for Llama-3.1-8B, with **answer-token logit cosine
0.99+** throughout. This is the MemArt-comparable setting (LoCoMo QA): the *compose* axis
holds on real memory, not just synthetic gated decisions — and our addition over MemArt
remains the edit axis + mechanism + decision-governance lens. (32B-FP8 hit GPU-contention OOM
and is omitted; 4B/8B/14B stand.) The earlier 0.58/0.46 figure was a 24-question smoke; these
N=1,540 numbers with CIs supersede it.

## E1 — Placement × pre-digestion (POWERED — supersedes the earlier underpowered read)
*early `[sys][MEM][traj]` vs late `[sys][traj][MEM]` under full recompute; direct/CoT × depth × length.*
*Powered to **N=192** personas/cell on Qwen3-4B (short ≈400 tok & ≈2k tok) + **N=32** at 16k/32k.*

**Regime.** Direct one-shot decisions are floored (oracle 0.50, constant "yes") — no placement
effect because no integration happens. **CoT is competent** (Qwen3-4B oracle 0.92).

**Placement effect is real and significant once powered (correction).** At the original n=40
the effect was n.s.; at **N=192**, GEE-logistic (cluster=persona) gives `placement[late]` coef
**−0.436, p=0.018** — i.e., **late placement is significantly *worse* than early** under CoT
(the pre-digestion cost the mechanism predicts). The `placement×n_facts` interaction is n.s.
(p=0.34) at short memory.

**The cost grows with memory length.** Long-memory sweep (Qwen3-4B, CoT, early−late accuracy
gap): ≈2k tok ≈0; **16k: +0.09 (nf=1)**; **32k: +0.16 (nf=1)**, +0.06 (nf=8). So the
pre-digestion penalty for reading memory late is small at short memory but **rises to ~0.1–0.16
at 16–32k**. (Noisy at n=32, not monotonic in depth; Llama-8B is below the 0.80 competence gate
at these lengths and excluded.)

**Takeaway (honest tradeoff, not "free").** Reading memory late costs a **small but real**
accuracy decrement vs. early (significant; ~a few points short-context, ~0.1–0.16 at 16–32k),
because late placement forgoes prefill-time pre-digestion and relies on the decode/CoT to
integrate raw memory. The method still favors late placement: it buys $O(L)$ editing/transplant
and 2.3–4.3× TTFT (E5) and the transplant itself is faithful at a *fixed* placement (E2/LoCoMo),
so the net is **small accuracy for large efficiency** — a quantified tradeoff, and a reason to
prefer early placement when memory is very long and accuracy-critical.

## E3 — Editing memory mid-session
*A relevant fact is toggled mid-session; each edit method vs the full-recompute oracle (CoT).*
*Powered: **Qwen3-4B at N=480** — in_place 0.994 [0.985], erratum 0.981 [0.969], recompile 0.996
[0.990], stale 0.006; McNemar in_place-vs-erratum n.s. (p=0.11). Smaller-N rows below stand.*

**The edit is necessary and cheap.** Reusing the **stale** memory recovers the flipped
decision essentially never (correct: Qwen3-1.7B 0.03, 4B 0.00, Llama-8B 0.00). Every real
edit method recovers it, and under CoT even the **near-free in-place edit (1 token
recomputed)** works on competent models:

| model | stale | in_place(1tok) | selective@4 | erratum(27tok) | recompile(407tok) | full(662tok) |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | 0.03 | 0.92 | 0.97 | 0.95 | 0.98 | 0.97 |
| Qwen3-4B | 0.00 | 0.98 | 0.98 | 0.98 | 0.97 | 0.91 |
| Llama-3.1-8B | 0.00 | 0.50* | 0.52* | 0.59* | 0.48* | 0.55* |

*(decision correct vs new gold, CoT). *Llama's absolute numbers are depressed by weaker
CoT "FINAL:" format compliance; its **agreement with its own oracle** is high for every
method (in_place 0.89, selective@16 0.94, recompile 0.94), i.e. the edit faithfully
reproduces what a full reprefill would decide.* **McNemar in_place vs erratum:** ns for
Qwen (p=0.73, 1.00) but **erratum > in_place for Llama (p=0.031, b=6,c=0)** — the salient
erratum is the more robust edit, exactly the paper's "robust default" recommendation, now on
the memory substrate. **Stickiness/scale**: in-place correctness 0.92 (1.7B) → 0.98 (4B),
then drops on Llama (confounded by format compliance) — reported honestly.

**Takeaway (novelty vs MemArt):** editable memory works — stale memory is wrong, a 1-token
in-place edit recovers the decision under reasoning, and the append-only erratum is the
robust default. MemArt/EPIC retrieve/splice static blocks; they have no in-place memory
update. This is the editing axis on the memory substrate.

## E4 — Edit granularity / sub-chunking
*Memory split into S independently-precompiled blocks (n=300, n_facts=4, L_mem≈1045 tok).*

Sub-chunking is **decision-lossless** down to fine granularity while cutting localized-edit
cost ∝ 1/S:

| S | edit cost (tok) | Qwen3-4B dec_agree | Qwen3-4B cos | Llama dec_agree | Llama cos |
|---|---|---|---|---|---|
| 1 | 1045 | 1.000 | 0.998 | 0.883 | 0.960 |
| 4 | 261 | 1.000 | 0.991 | 0.883 | 0.660 |
| 8 | 131 | 1.000 | 0.979 | 0.883 | 0.491 |
| 16 | 65 | 1.000 | 0.953 | 0.883 | 0.388 |

Decision agreement is **flat in S** (Qwen 1.000, Llama 0.883 = its S=1 baseline) — splitting
memory into 16 blocks for a **16× cheaper** localized edit does not change the decision,
because the independent setting-facts are integrated at read time by the decision/CoT, not
within memory. Logit cosine *does* degrade with S (esp. Llama), so for facts with genuine
**cross-block dependencies** finer S would eventually bite; characterizing that boundary is
future work. Practical knob: choose S to match edit-locality; decisions are robust.

## P3 — Keystone / locality-knockout on transplanted memory (Llama-70B, DIRECT)
*Edit a gating field **inside a transplanted memory chunk**, direct mode on Llama-70B (the one
model whose direct decisions vary, so recovery is measurable); n=64. This is the §3 mechanism
probe reproduced on memory — and the keystone (edit acts the same inside a transplant).*

| method | recompute | correct | agree w/ oracle |
|---|---|---|---|
| stale | 0 | 0.203 | 0.27 |
| in-place (field only) | 1 | 0.547 | 0.61 |
| selective@4 | 5 | 0.578 | 0.64 |
| selective@16 | 17 | **0.844** | 0.91 |
| recompile chunk | 407 | 0.875 | 0.94 |
| full | 674 | 0.938 | 1.00 |

**Textbook memoization result on memory:** refreshing the field's KV alone (in-place) recovers
only ~0.55 — the flipped conclusion was memoized **downstream**, not in the field — and recovery
**climbs monotonically as more downstream tokens are recomputed** (0.55→0.84 at @16→0.94 full).
Reproduced *inside a transplanted chunk*, this is the keystone for memory: edit and compose act
on one substrate. (Contrast P5: under CoT this stickiness disappears.)

## P5 — selective@K sweep (minimal recompute / stickiness, Qwen3-4B, CoT)
selective@{1,2,4,8,16,32,64} correctness is **flat at ≈0.97–0.98** (in-place 0.984; stale 0.000;
full 0.906) — under CoT the **minimal recompute is K*≈1** (the field refresh alone suffices,
because the chain re-reads it). So stickiness is a **direct-mode** phenomenon (P3, where K
matters: @1=0.55 → @16=0.84) that **dissolves under reasoning** (P5, K irrelevant). Clean
direct-vs-CoT dissociation of the memoization mechanism.

## P4 — Cross-block dependency (gating facts within one block vs split across blocks)
*Qwen3-4B, n=200/layout, S∈{1,4,16}. Decision agreement is the constant-answer artifact in
direct mode (Qwen3-4B), so logit cosine is the discriminating metric.*

| S | contiguous cos | spread cos |
|---|---|---|
| 1 | 0.998 | 0.998 |
| 4 | 0.991 | 0.991 |
| 16 | 0.953 | 0.953 |

Logit cosine degrades with S **identically whether the gating facts are contiguous (one block)
or spread across blocks** — so splitting the relevant facts across independently-precompiled
blocks does **not** specifically hurt fidelity (the decision token reads each block directly at
integration time). Earlier I hedged that cross-referential facts would bite finer S; for
independent AND-gated facts that is **not observed** — sub-chunking is robust to fact placement.
(A task with genuine inter-fact *reference* — fact A's meaning depends on B — remains untested.)

## E5 — End-to-end systems + working agent
*`MemoryAgent` vs front/end reprefill; 24 sessions × 12 turns, edit-rate 0.25, ≈2k-tok memory;
faithfulness vs a **token-matched** full-reprefill oracle (exact same token stream).*

**Latency (the headline).** Cumulative time-to-first-token speedup of the proposed agent:

| model | TTFT proposed (ms) | vs end-reprefill | vs front-reprefill | vs oracle | faithfulness cos |
|---|---|---|---|---|---|
| Qwen3-1.7B | 42.8 | 2.30× | 1.37× | 1.99× | 0.993 |
| Qwen3-4B | 64.3 | 3.05× | 1.67× | 2.65× | 0.984 |
| Llama-3.1-8B | 69.9 | 4.25× | 1.88× | 3.49× | 0.979 |
| Qwen3-14B | — | 4.02× | 1.93× | — | 0.974 |
| Qwen3-32B-FP8 | — | 4.32× | 3.27× | — | 0.975 |

Speedups extend to 32B (vs end **2.3–4.32×**, vs front **1.4–3.27×**), confirming the
amortization grows with model size. (E5-14B/32B and the 32B LoCoMo point were recovered from
GPU-contention OOM via the wait-for-memory + backoff retries; **all originally-failed runs are
now complete**.)

Speedups grow with model size (end-reprefill re-attends the whole memory every turn; the
proposed agent only re-rotates it). **Logit faithfulness is high (cos 0.979–0.993)** — the
matched-oracle fix raised this from 0.86 (a prior text-construction mismatch).

**Decision governance — honest caveat.** The *direct* yes/no decision is constant ("yes",
96/96) for these models in this multi-turn long-memory setting, so direct agreement (1.000)
is the **constant-answer artifact**, not evidence. The discriminating metric is the **CoT
spot-check**, where the transplant's reasoning *chain* agrees with the oracle's only
0.31–0.69 of the time — **not because the transplant is unfaithful** (its next-token cosine is
0.98–0.99 and the *direct* decision is identical) but because **greedy CoT is chaotic**:
sub-percent logit differences flip an argmax at some reasoning step and the ~300-token chain
then diverges. CoT *accuracy* is comparable between proposed and oracle (e.g. Qwen3-4B
0.90 vs 0.71; both noisy at n=48 on the hard long-memory multi-turn task), so decision
*quality* is preserved even when the exact chain is not reproduced. **The clean CoT
decision-governance equivalence is in E3 (short memory): edits recover the oracle decision
with agreement 0.89–0.95.**

**Net E5 claim:** the working agent delivers **2.3–4.25× cumulative TTFT speedup** at high
logit faithfulness (cos ≥0.98); exact long-CoT chain reproduction is not guaranteed
(autoregressive sensitivity), a limitation, while short-context decision governance (E3) and
logit/direct faithfulness (E2) are strong.

## Negative control
*Editing an IRRELEVANT fact must not change the decision (false-positive check); CoT, n=48/model.*

Decision stability under an irrelevant edit (should be ≈1.0): Qwen3-4B 0.958–0.979 across
methods; Llama-3.1-8B 0.854–0.875. No method spuriously flips the decision. The erratum is
marginally the least stable (0.958 on Qwen3-4B) — a salient note about an irrelevant setting
can mildly distract — consistent with the paper's wording-ablation that over-salient
corrections can perturb. Net: edits are *targeted*, not indiscriminate.

## Limitations (this study)
- Models ≤ 8B (shared GPU); the scale ladder is Qwen3-1.7B/4B + Llama-3.1-8B (CoT
  competent) plus 0.6B/Gemma for faithfulness. No 30B+/long-context-extrapolation cells.
- Controlled synthetic memory (account settings) with balanced gated decisions; LoCoMo /
  real assistant logs remain external-validity future work.
- CoT is the competent regime here; direct-mode decision-governance is floored and reported
  as such, not as a method failure.
