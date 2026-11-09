# T1 / T7 / T1b: neurons behind the Programmable-KV conclusion note

**Models.** Llama-3.1-8B-Instruct, using the ungated mirror `unsloth/Llama-3.1-8B-Instruct`. It has the same weights as the paper's `unsloth/Meta-Llama-3.1-8B-Instruct`. Qwen3-8B is used to replicate steps 2–3. Both run in bf16 with sdpa attention.

**Task.** The repo's own polarity task (`esys/mechd_common.POL` + `circuit_common`), using its full field/trigger 2×2. That gives 3 scenarios × 4 order ids × 2 field values = **24 SAFE/UNSAFE pairs**. In each pair the field is byte-identical, and only the rule trigger differs, which flips the conclusion. Neurons are ranked on order ids A4471/B8820/C1093 (18 pairs) and tested on all pairs. The held-out order id D5567 (6 pairs) is reported separately.

**Score and aggregators.** S = logit[safe action] − logit[unsafe action] at the decision token. Every effect is normalized by S_SAFE − S_UNSAFE. Aggregators are the repo's KV-patch-ranked downstream positions (`find_aggregators`): agg8 = top-8 positions, agg1 = top-1.

**Neuron.** A neuron is one SwiGLU intermediate unit, i.e. one input to `mlp.down_proj`. Llama has 32 × 14,336 = 458,752 of them. Ranking uses attribution patching (AtP) at agg8: (a_SAFE − a_UNSAFE)·∂S/∂a, evaluated on the UNSAFE run. This is a first-order estimate of each neuron's patch effect.

**Data.** An earlier run on 12 pairs (field value 0 only) is kept in `v1_12pairs/`. Its numbers agree with the 24-pair run to within a few hundredths.

## Step 1: the baseline reproduces (Llama-3.1-8B)

| quantity | paper record | ours |
|---|---|---|
| field-only KV refresh recovery (`mech_causal_patch`, 12 pairs) | −0.028 | **−0.021** CI(−0.04, −0.012) |
| full-downstream recovery | 1.00 | 1.00 |
| top-8 aggregator transplant (`mechd_specificity`, 9 pairs) | 0.793 | **0.793** CI(0.708, 0.839) |
| random-8 positions | 0.005 | **0.017** |

Records are in `repro/`.

## Steps 2–3: attribution concentration and neuron patching (Llama-3.1-8B, 24 pairs)

@@CAUSAL_LLAMA@@
@@SPLIT@@
**Reading.**
- **The note is real and localized, but not in a few neurons.** Random neurons (layer-matched) are always about 0 (|x| ≤ 0.01). The attributed neurons scale smoothly: top-1 recovers 0.03, top-10 0.15, top-30 0.30, top-100 0.52, top-300 0.72. The held-out order id matches the training pairs.
- **The 30-neuron result matches the paper's SAE result.** The paper's best single SAE feature recovers about 0, and its 3/10/30 features recover 0.30/0.50/0.54.
- **|AtP| is spread across tens of thousands of neurons.** The top-30 hold only 1.9% of |AtP| mass, and 50% of it needs about 28,000 neurons.
- **Signed contributions cancel heavily.** The top-100 positive neurons already sum to 1.36× the net signed AtP. Patching the top-300 (0.72) beats patching all 458,752 neurons at agg8 (0.39), because the remainder pushes the other way.
- **Depth.** The write peaks at layers 11–16 (L13 + L14 ≈ 26% of |AtP|), with a second, smaller band at L19–25.

## Step 7 (optional): Qwen3-8B replication of steps 2–3 (24 pairs)

@@CAUSAL_QWEN@@
Qwen3-8B shows the same picture. It is graded rather than few-neuron (top-30 recovers 0.25, top-300 0.55), random neurons are about 0, and half the |AtP| mass needs about 27,000 neurons. The depth band sits later, at L17–22. Single-aggregator (agg1) effects are smaller than in Llama. That fits the paper's note that Qwen3-8B's conclusion is "so strongly committed": its logit gap ranges from 9 to 30.

## Step 4: labels for the top-100 note neurons and 100 layer-matched random controls (MiMo Pro, essay-style)

MiMo Pro labelled each neuron from its top-20 positive and top-15 negative contexts in 2.5M tokens of fineweb-edu. The activating token was marked, and the model was blind to the task.

@@LABELS@@
**Reading.**
- **MiMo calls almost every neuron legible, note neurons and random controls alike.** Of the top-100, 26 are "one job" and 70 "two related jobs". Of the random-100, 10 and 90. That matches the essay's "~90% of poles have one main job" style of audit.
- **The labels don't identify the note neurons.** Apart from the #1 neuron, the top note neurons look like generic clause/sentence-boundary, continuation and connective detectors on web text. That is expected for units acting at punctuation/newline aggregator tokens, but it says nothing about their role in this conclusion.
- **The #1 neuron is the exception.** L25.11382 is labelled "refusal / declining to comply". Its attribution comes only from the two scenarios whose SAFE action is `refuse`, so it is a legible output feature in the essay's sense.

## Step 5: Gao's simulation test

MiMo predicts each neuron's activation per token from its label alone (integer −10…10). A linear calibration fitted on general text maps that prediction to activation units. Then real activations are replaced by simulated ones. The score is "damage avoided" = 1 − KL(simulated)/KL(mean-ablation), where 1 means a perfect explanation and ≤0 means no better than ablation. General-text windows mix 10 random windows with 10 top/bottom windows from a held-out split. At the decision, single-neuron substitutions are batched, with an unpatched reference row in the same batch.

@@GAO@@
**Reading. The labels fail Gao's test almost completely.**
- **Correlation with real activations.** Median r between simulated and real activations is 0.01 for the top-100 note neurons and 0.09 for random neurons. Only 1/100 note neurons and 11/100 random neurons reach r > 0.3.
- **Damage avoided ≈ 0.** On general text, the median single-neuron score is +0.00, i.e. no better than mean-ablation. Substituting all top-100 jointly on the shared random windows is *worse* than mean-ablation (−0.23); the top-30 give −0.05.
- **The task decision.** Simulated activations at the aggregators barely track the real SAFE−UNSAFE difference (median r 0.01). A "simulated patch" with the top-30 recovers 0.03 of the decision, against 0.30 for the real patch. The top-100 give 0.07 against 0.52. Decision-level damage avoided is −0.19 (top-30) and −0.48 (top-100): substituting the simulations does more harm than mean-ablating the neurons.
- **The failure mode is visible in the strongest neuron.** L25.11382 ("refusal verb") peaks at 10.9 on the token *before* "refused" ("…When the [Post]"). It predicts the verb rather than responding to it, and the label-based simulator misses this. Its general-text r is only 0.10.
- **Caveat.** MiMo Pro is one simulator, with an integer −10…10 scale and a linear calibration. A stronger simulator or explainer would do better. But the gap is large: a neuron the labeler confidently calls one job (confidence 0.6–0.85) cannot be simulated well enough to replace it.

## Step 6: do the late read heads read along these neurons' output directions?

The quantity tested is the OV-circuit projection W_O·W_V·(g ⊙ down_proj column) onto each scenario's unembedding difference. It is direct-path only and is compared against layer-matched random neurons (×5).

@@OV@@
**Reading.**
- **Most read heads show no preference.** In 9 of the 12 read heads, top-30 note neurons have the same or *lower* OV gain and logit projection as random neurons from the same layers.
- **The exceptions are heads 31.1, 30.26 and 30.24.** There the top-30 project about 1.5–2.7× more onto the safe-vs-unsafe logit, but only for the two `refuse` scenarios. This is consistent with the L25 "refuse" neuron writing an answer direction that late heads copy.
- **Overall the read heads don't read neuron directions directly.** The note is re-encoded (through attention at the aggregators and later layers) before the decision heads read it. That fits the paper's "distributed write, concentrated read".

## T7: negative poles

At the agg8 positions, each neuron's activation was split by sign. Positive parts, negative parts, or both were zeroed in BOTH prompts. The measure is share_lost = 1 − D′/D, with D = S_SAFE − S_UNSAFE. Controls are random neurons in the same layer with a matched negative-mass fraction. A neuron is "negative-side" if more than 60% of its |AtP| comes from positions where its activation is negative.

@@T7@@
**Reading.**
- **Across the whole population, negative activations matter about as much as positive ones.** Zeroing every negative part at agg8 loses 0.55 of the decision; zeroing every positive part loses 0.70; zeroing both loses 0.92. Negative activations also carry 0.37–0.49 of the top neurons' |AtP|, and negative-side and positive-side neurons are about equally common (132 vs 144 in the top-300).
- **But no small set of negative-side neurons acts like the essay's Santa result.** Zeroing 8 top negative-side neurons loses 0.011, against 0.005 for random controls. Zeroing 20 top positive-side neurons loses 0.067. In the essay's case, 8 negative-pole neurons dropped Santa to 6% while 20 labelled neurons only dropped it to 76%; nothing here is analogous.
- **Zero-ablation hits the most-attributed neurons weakly.** Zeroing the negative parts of the top-10 and top-30 neurons does about nothing, and zeroing their positive parts loses only 0.03–0.08. Much of their effect is the SAFE-vs-UNSAFE *difference* in magnitude, which zeroing both runs does not reproduce.
- **Verdict.** The negative pole matters in aggregate, as the essay says. It is not concentrated in a few nameable neurons.

## T1b: distributed or redundant?

Carriers on Llama-3.1-8B, all 24 pairs:
- the top-16 aggregator positions (their KV)
- the top write heads from the paper's ranking, at agg8
- the top-30 MLP neurons from step 3, at agg8
- the top-30 conclusion features of a layer-14 TopK SAE, retrained with the repo recipe (FVU about 0.001, best feature AUC 1.0, matching the paper), at agg1
- the top-12 read heads, at the decision token

Retention is measured on the clean run. Batched runs pad every batch to a fixed size, and baselines are computed at that same batch size, because bf16 batch numerics otherwise add a +0.06 bias; this was caught and fixed before the final run. Compensation is the change in the other carriers' attribution-patching (AtP) effect (or, for read heads, their direct logit attribution) after one carrier is patch-ablated.

@@T1B@@
**Additivity.** Two measures are used, both pooled over the 24 pairs with bootstrap 95% CIs:
- the *joint/sum ratio* = Σ_pairs joint drop / Σ_pairs (sum of single-carrier drops). 1 = additive, >1 = superadditive (one-at-a-time removal undercounts), <1 = overlap/redundancy;
- the fraction of the joint drop that single removals capture (sum/joint).
Decision logits are computed in fp32 from the final hidden state, because bf16 logits (step ≈0.125 near |logit| ≈ 20) quantize single-carrier effects to ±0.016.

| carrier | joint drop | Σ single drops | joint/sum (necessity) | singles capture | joint − sum | joint/sum (sufficiency) |
|---|---|---|---|---|---|---|
| aggregator positions (16) | 0.837 | 0.764 | 1.10 [1.01, 1.20] | 0.91 [0.83, 1.00] | +0.073 [+0.003, +0.139] | 0.79 [0.73, 0.86] |
| write heads (top-15 available) | 0.249 | 0.212 | 1.17 [0.91, 1.71] | 0.85 [0.59, 1.09] | +0.037 [−0.025, +0.096] | 0.50 [0.47, 0.53] |
| MLP neurons (30) | 0.181 | 0.097 | 1.87 (CI unstable: denominator near 0 in some pairs) | **0.53 [−0.02, 1.06]** | **+0.084 [−0.010, +0.178]** | 0.80 [0.66, 0.99] |
| SAE L14 features (30) | 0.305 | 0.233 | 1.31 [1.00, 1.97] | 0.76 [0.51, 1.00] | +0.072 [−0.001, +0.145] | 1.16 [1.02, 1.33] |
| read heads (12) | 0.490 | 0.518 | 0.95 [0.90, 0.99] | 1.06 [1.01, 1.10] | −0.028 [−0.051, −0.006] | 0.84 [0.81, 0.87] |

(The write-head ranking file lists 15 heads, not 20.)

**Write side (verdict).** The write is *distributed*: the ablation-size curves fall steadily rather than staying flat and then dropping off a cliff, and no single carrier dominates.
- **Removing carriers one at a time undercounts the write.** This is strongest for MLP neurons. Single-neuron removals add up to only **0.53 of the joint effect** of the top-30 (0.097 vs 0.181 of the decision), so one-at-a-time removal **undercounts by about 1.9×**. The per-pair noise is large: joint − sum is +0.084 with CI [−0.010, +0.178], so this is borderline at 95%.
- **SAE features and aggregator positions show the same direction, more weakly.** Their singles capture 0.76 and 0.91 of the joint effect.
- **The superadditivity is not linear compensation inside the carrier set.** The AtP-measured hydra effect among the other top-30 neurons is ≈0 (pooled ratio +0.007). That leaves interaction (threshold/AND-like combination downstream) or backup outside the measured set.
- **Sufficiency is subadditive for write heads.** Their summed single sufficiency is twice the joint (joint/sum 0.50), and 8/360 single heads have sufficiency > 0.3 with necessity < 0.1. Several write heads therefore each carry *overlapping* copies of the conclusion: classic partial redundancy. Positions (0.79) and neurons (0.80) show the same pattern more mildly.
- **Compensation among write heads is heavy-tailed.** The pooled ratio is +1.27, but the median is +0.03: a few heads strongly take over when a top head is removed.
- **In short:** many small, partly overlapping pieces. One-at-a-time ablation understates the neuron-level contribution by about half.

**Read side (verdict).** The read is *additive and concentrated*.
- **Additivity.** The 12 read heads' single drops sum to 1.06× the joint drop (0.518 vs 0.490), and the curve falls steadily from 1.0 to 0.51 at k = 12.
- **Compensation.** There is none among the top-12 (pooled ratio −0.04). The rest of the downstream attention heads pick up a modest +0.17 of an ablated head's direct effect.
- **In short:** concentrated and close to additive, with weak downstream backup. This matches the paper's "concentrated read".

## Conclusion: clean neurons or distributed?

**Does this support "clean neurons" (Cammarata) or "decodable ≠ causal, distributed" (Gao, and the paper's own SAE finding)? Mostly the latter.**
1. **Few-neuron explanations fail.** The note is causally real and specific: real patches beat layer-matched random neurons by more than 100×. But it is not carried by a few neurons. The top-1 neuron recovers 0.03, the top-30 0.30, the top-300 0.72. Half the |AtP| mass needs ~28k neurons in Llama and ~27k in Qwen3-8B, and signed effects cancel across the population. This reproduces the paper's SAE result (30 features → 0.54) at the neuron level.
2. **Legible labels, failed explanations.** MiMo calls ~96% of note neurons "one job" or "two related jobs", the essay-style audit result. Yet Gao's simulation test fails: median r ≈ 0.01, ≈0 damage avoided, and simulated patches recover 0.03 vs 0.30 for the real ones. Here, legibility is not explanation.
3. **The essay's specific claims hold only in limited form.** There is a genuinely clean, legible output neuron (L25.11382, "refusal"), and negative activations carry as much of the note as positive ones in aggregate (zeroing all negative parts loses 0.55, all positive parts 0.70). But there is no Santa-style small negative-pole set (8 negative-side neurons → 0.011).
4. **Redundancy and synergy.** One-at-a-time neuron ablation undercounts the joint contribution by ~1.9×, and write-head sufficiency is 2× subadditive. So per-neuron ablation audits systematically understate what neurons do here. That is a further reason single-neuron stories don't add up to the mechanism.
5. **Clean versus distributed by side.** Read side: concentrated and additive, ≈12 heads. Write side: distributed across positions, heads and tens of thousands of MLP units, partly redundant. Attention carries most of the write (patching all attention outputs at agg8 recovers 0.89, against 0.39 for all MLP outputs). This sharpens the paper's "distributed write, concentrated read" claim.

## Files

- Scripts: `t1_common.py`, `t1_neurons.py`, `t1_corpus.py`, `t1_mimo.py`, `t1_fix.py`, `t1_gao.py`, `t1_t7.py`, `t1b_redundancy.py`, `t1_split.py`, `make_tables.py`, `build_report.sh`
- Raw results: `t1_neurons_{llama31_8b,qwen3_8b}.json`, `t1_atp_*.pt`, `t1_signdata_*.pt`, `corpus_stats_llama31_8b.pt`, `labels_llama31_8b.json`, `sim_general_llama31_8b.json`, `sim_task_llama31_8b.json`, `gao_llama31_8b.json`, `t7_llama31_8b.json`, `t1b_llama31_8b.json`, `t1b_aggs_llama31_8b.json`, `t1b_sae_L14_llama31_8b.pt`, `ov_llama31_8b.json`, `t1_split_llama31_8b.json`
- Repro records: `repro/`
- 12-pair first pass: `v1_12pairs/`
