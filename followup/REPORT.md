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

n = 24 pairs (18 train / 6 held-out order id)

| positions | test | k=1 | 3 | 10 | 30 | 100 | 300 | all neurons |
|---|---|---|---|---|---|---|---|---|
| agg8 | sufficiency (patch SAFE→UNSAFE) | +0.032 (rand +0.005) | +0.063 (rand +0.002) | +0.145 (rand +0.006) | +0.295 (rand +0.003) | +0.515 (rand +0.004) | +0.719 (rand +0.000) | +0.388 |
| agg8 | necessity (patch UNSAFE→SAFE) | +0.025 (rand +0.001) | +0.033 (rand -0.003) | +0.080 (rand -0.006) | +0.182 (rand -0.004) | +0.386 (rand -0.003) | +0.614 (rand -0.003) | +0.509 |
| agg8 | zero-ablate (drop) | +0.038 (rand +0.002) | +0.048 (rand -0.002) | +0.119 (rand -0.005) | +0.173 (rand -0.004) | +0.262 (rand +0.000) | +0.320 (rand -0.000) |  |
| agg8 | mean-ablate (drop) | +0.031 (rand +0.001) | +0.039 (rand -0.001) | +0.131 (rand -0.003) | +0.194 (rand -0.003) | +0.322 (rand +0.000) | +0.361 (rand -0.005) |  |
| agg1 | sufficiency (patch SAFE→UNSAFE) | +0.032 (rand +0.004) | +0.059 (rand +0.002) | +0.111 (rand +0.003) | +0.210 (rand +0.003) | +0.340 (rand +0.002) | +0.478 (rand +0.001) | +0.256 |
| agg1 | necessity (patch UNSAFE→SAFE) | +0.024 (rand -0.001) | +0.029 (rand -0.002) | +0.050 (rand -0.003) | +0.107 (rand -0.003) | +0.196 (rand -0.004) | +0.346 (rand -0.004) | +0.245 |
| agg1 | zero-ablate (drop) | +0.034 (rand -0.000) | +0.040 (rand -0.002) | +0.089 (rand -0.002) | +0.161 (rand -0.001) | +0.261 (rand -0.001) | +0.273 (rand +0.002) |  |
| agg1 | mean-ablate (drop) | +0.030 (rand -0.001) | +0.037 (rand -0.001) | +0.088 (rand -0.004) | +0.164 (rand -0.004) | +0.270 (rand -0.003) | +0.275 (rand +0.004) |  |

Held-out order id only (sufficiency, agg8): k=1: +0.028, k=3: +0.056, k=10: +0.141, k=30: +0.292, k=100: +0.503, k=300: +0.700

Concentration of |AtP| (train mean): top-1 0.002, top-10 0.009, top-30 0.019, top-100 0.040, top-300 0.071, top-1000 0.126, top-10000 0.335; neurons needed for 50/80/90% of |AtP|: 28129/114145/182082 (per-instance median for 50%: 30122).
Net signed AtP sum 0.468; positive top-k as a fraction of the net: k=1: 0.07, k=3: 0.13, k=10: 0.31, k=30: 0.66, k=100: 1.36, k=300: 2.34, k=1000: 3.92, k=3000: 6.04
In-sample (per-instance top-k) sufficiency: {'1': 0.0505, '10': 0.2584, '100': 0.6749, '1000': 1.2608}
AtP linearity: patching all neurons at agg8 = 0.388 vs sum of AtP 0.464.
Layer share of |AtP| (top layers): L13: 0.130, L14: 0.127, L12: 0.088, L15: 0.080, L16: 0.070, L11: 0.062


Attention vs MLP at agg8 (24 pairs): AtP attention share = 0.781 (AtP attn +1.518, MLP +0.464); actual patch of all attention outputs at agg8 = +0.886, all MLP outputs = +0.388. Repo's direct-projection share (circ_components, T=14): 0.599.


**Reading.**
- **The note is real and localized, but not in a few neurons.** Random neurons (layer-matched) are always about 0 (|x| ≤ 0.01). The attributed neurons scale smoothly: top-1 recovers 0.03, top-10 0.15, top-30 0.30, top-100 0.52, top-300 0.72. The held-out order id matches the training pairs.
- **The 30-neuron result matches the paper's SAE result.** The paper's best single SAE feature recovers about 0, and its 3/10/30 features recover 0.30/0.50/0.54.
- **|AtP| is spread across tens of thousands of neurons.** The top-30 hold only 1.9% of |AtP| mass, and 50% of it needs about 28,000 neurons.
- **Signed contributions cancel heavily.** The top-100 positive neurons already sum to 1.36× the net signed AtP. Patching the top-300 (0.72) beats patching all 458,752 neurons at agg8 (0.39), because the remainder pushes the other way.
- **Depth.** The write peaks at layers 11–16 (L13 + L14 ≈ 26% of |AtP|), with a second, smaller band at L19–25.

## Step 7 (optional): Qwen3-8B replication of steps 2–3 (24 pairs)

n = 24 pairs (18 train / 6 held-out order id)

| positions | test | k=1 | 3 | 10 | 30 | 100 | 300 | all neurons |
|---|---|---|---|---|---|---|---|---|
| agg8 | sufficiency (patch SAFE→UNSAFE) | +0.032 (rand -0.003) | +0.082 (rand -0.002) | +0.160 (rand -0.002) | +0.252 (rand -0.005) | +0.387 (rand -0.007) | +0.545 (rand -0.010) | +0.463 |
| agg8 | necessity (patch UNSAFE→SAFE) | +0.031 (rand +0.005) | +0.063 (rand +0.003) | +0.101 (rand +0.005) | +0.194 (rand +0.003) | +0.293 (rand +0.004) | +0.428 (rand +0.002) | +0.362 |
| agg8 | zero-ablate (drop) | -0.007 (rand +0.005) | +0.023 (rand +0.006) | +0.073 (rand +0.005) | +0.131 (rand +0.004) | +0.253 (rand +0.003) | +0.427 (rand +0.007) |  |
| agg8 | mean-ablate (drop) | -0.002 (rand +0.006) | +0.022 (rand +0.006) | +0.073 (rand +0.003) | +0.136 (rand +0.004) | +0.240 (rand +0.005) | +0.408 (rand +0.003) |  |
| agg1 | sufficiency (patch SAFE→UNSAFE) | -0.003 (rand -0.004) | +0.006 (rand -0.003) | +0.027 (rand -0.005) | +0.047 (rand -0.006) | +0.081 (rand -0.007) | +0.128 (rand -0.006) | +0.110 |
| agg1 | necessity (patch UNSAFE→SAFE) | +0.001 (rand +0.001) | +0.006 (rand +0.001) | +0.021 (rand +0.001) | +0.043 (rand +0.001) | +0.059 (rand +0.003) | +0.095 (rand -0.000) | +0.082 |
| agg1 | zero-ablate (drop) | -0.001 (rand +0.001) | +0.008 (rand +0.002) | +0.015 (rand +0.002) | +0.025 (rand +0.003) | +0.056 (rand +0.004) | +0.092 (rand +0.001) |  |
| agg1 | mean-ablate (drop) | +0.001 (rand +0.001) | +0.008 (rand +0.002) | +0.015 (rand +0.002) | +0.023 (rand +0.004) | +0.052 (rand +0.002) | +0.089 (rand +0.000) |  |

Held-out order id only (sufficiency, agg8): k=1: +0.045, k=3: +0.090, k=10: +0.171, k=30: +0.257, k=100: +0.398, k=300: +0.559

Concentration of |AtP| (train mean): top-1 0.003, top-10 0.013, top-30 0.023, top-100 0.041, top-300 0.069, top-1000 0.120, top-10000 0.330; neurons needed for 50/80/90% of |AtP|: 27304/101441/159868 (per-instance median for 50%: 27103).
Net signed AtP sum 0.380; positive top-k as a fraction of the net: k=1: 0.08, k=3: 0.22, k=10: 0.43, k=30: 0.72, k=100: 1.22, k=300: 1.94, k=1000: 3.13, k=3000: 4.82
In-sample (per-instance top-k) sufficiency: {'1': 0.0449, '10': 0.1864, '100': 0.4965, '1000': 0.9714}
AtP linearity: patching all neurons at agg8 = 0.463 vs sum of AtP 0.379.
Layer share of |AtP| (top layers): L19: 0.108, L20: 0.099, L21: 0.089, L18: 0.080, L22: 0.072, L17: 0.065


Qwen3-8B shows the same picture. It is graded rather than few-neuron (top-30 recovers 0.25, top-300 0.55), random neurons are about 0, and half the |AtP| mass needs about 27,000 neurons. The depth band sits later, at L17–22. Single-aggregator (agg1) effects are smaller than in Llama. That fits the paper's note that Qwen3-8B's conclusion is "so strongly committed": its logit gap ranges from 9 to 30.

## Step 4: labels for the top-100 note neurons and 100 layer-matched random controls (MiMo Pro, essay-style)

MiMo Pro labelled each neuron from its top-20 positive and top-15 negative contexts in 2.5M tokens of fineweb-edu. The activating token was marked, and the model was blind to the task.

top: n=100 jobs={'one_job': 26, 'two_related_jobs': 70, 'unclear': 3, 'multiple_unrelated': 1} mean conf=0.63
random: n=100 jobs={'two_related_jobs': 90, 'one_job': 10} mean conf=0.68

| rank | neuron | label | negative pole | jobs | r(sim,real) general | damage avoided (general, vs mean-abl.) | r task | damage avoided (task) |
|---|---|---|---|---|---|---|---|---|
| 0 | L25.11382 | Refusal/declining to comply (verb 'refuse/decline') | Proper nouns, citations, years (e.g., '2003', '2006', '1530'), author  | one_job | 0.10242868385325321 | +0.005 | -0.02063328559564707 | +0.223 |
| 1 | L19.2048 | Year digits in citations/bibliographic dates | Ordinary prose tokens, especially mid-word fragments, function words,  | one_job | 0.04456682433092845 | +0.008 | -0.07530308413843098 | -0.095 |
| 2 | L19.11138 | Word-boundary/whitespace tokenization artifact detector | Strong negative activation on common function words and ordinary word- | one_job | -0.018776363150503496 | -0.017 | 0.17316362120984535 | -0.671 |
| 3 | L15.1993 | Wiki-table/structured-data formatting vs prose function words | Function words (the, and, U, -, e, ,) in flowing expository prose, esp | two_related_jobs | 0.009445642704445256 | +0.040 | -0.09629612411331691 | -0.160 |
| 4 | L13.8053 | Procedural/instructional conditional clauses vs narrative historical prose | Narrative historical/essay prose recounting past events, especially se | two_related_jobs | 0.040598112930818324 | +0.026 | -0.3724555003488755 | +0.040 |
| 5 | L16.4977 | Sentence-boundary / clause-initial function-word detector | Strong negative activation on sentence-final punctuation and clause-fi | two_related_jobs | -0.02724214238153698 | +0.022 | 0.09348265189662928 | -0.061 |
| 6 | L13.5663 | Sentence-boundary punctuation vs. mid-word/mid-phrase continuation | Strong negative on tokens that are mid-word fragments or mid-phrase co | two_related_jobs | 0.023249909500307184 | +0.022 | 0.08425186334407447 | +0.027 |
| 7 | L19.3042 | Acronym/abbreviation expansion and definition context detector | Parenthetical year/date citations and bibliographic references (author | two_related_jobs | -0.04055755958337694 | +0.007 | None | +0.146 |
| 8 | L12.824 | Sentence-boundary punctuation vs. mid-sentence content words | Strong negative activation on ordinary content words (nouns, adjective | one_job | 0.01959176158626128 | -0.020 | -0.03914077139108274 | +0.086 |
| 9 | L12.6946 | Sentence-final / clause-final function-word boundary detector | Strong negative on function words or suffixes that are NOT clause-fina | two_related_jobs | 0.0013556146058708535 | +0.028 | 0.10887979232734718 | +0.047 |
| 10 | L14.7651 | Function-word/whitespace token boundary detector | Also fires on similar boundary/function tokens but in a distinct conte | two_related_jobs | -0.023291513201929258 | +0.018 | -0.6743795069680922 | +0.029 |
| 11 | L16.1651 | Copula/linking-verb in definitional or classificatory clauses | Same copular/linking tokens but in negation, refutation, or contrastiv | two_related_jobs | 0.03687812832395961 | -0.010 | -0.044150239233513335 | +0.038 |
| 12 | L23.11190 | Parenthetical acronym expansion after capitalized abbreviation | Function words (of, and, or, to, like, in, before) in ordinary prose,  | two_related_jobs | -0.03245665615498283 | +0.032 | 0.2100090467117674 | -0.075 |
| 13 | L14.6974 | Concessive/contrastive discourse connective (but, however, despite, although) | Ordinary content words (nouns, verbs, adjectives) mid-sentence in neut | one_job | 0.0798232050083441 | -0.021 | -0.05129063412641397 | -0.228 |
| 14 | L19.14119 | Capital-letter word-initial / mid-word boundary detector | Strong negative at tokens that are digits or punctuation inside citati | two_related_jobs | 0.09104181593448883 | -0.028 | 0.022812705407940316 | -0.273 |
| 15 | L16.6837 | Function-word/preposition after clause boundary in prescriptive text | Content words (nouns, verbs, adjectives: are, win, required, kids, tow | two_related_jobs | 0.011040206248095529 | +0.009 | -0.042839631253840715 | -0.098 |
| 16 | L14.12797 | Parenthetical/aside clause boundary punctuation and connectives | Content words and definite articles mid-sentence in flowing narrative  | two_related_jobs | 0.06242384765149223 | -0.030 | 0.13607805689316715 | -0.087 |
| 17 | L13.4511 | Sentence-boundary/newline punctuation and function-word continuation | Negative on tokens that are mid-word fragments or hyphenated continuat | two_related_jobs | -0.01054481008262447 | +0.011 | 0.06465202866349515 | +0.333 |
| 18 | L12.13337 | Institutional/civic nouns vs. punctuation in explanatory prose | Punctuation marks (commas, dashes, parentheses) and function words in  | two_related_jobs | -0.029585539174739633 | +0.006 | -0.1681616210302998 | +0.210 |
| 19 | L13.13019 | Sentence-initial function word after clause boundary (But, He, The, It, I'm, and, to, of, 's, was, had, for, very, free, emerged, triumph, problems) | Strong negative on content-heavy tokens, especially longer/rarer words | two_related_jobs | 0.05813020168628724 | -0.001 | 0.24379221845028667 | -0.182 |
| 20 | L12.4570 | Word-final suffix/morpheme boundary detector (positive) vs. word-internal continuation (negative) | Strong negative at tokens that are word-INTERNAL continuations or the  | two_related_jobs | 0.0025946230368748876 | -0.010 | -0.10870772012654877 | -0.028 |
| 21 | L20.8650 | Function-word boundary / continuation vs. name-splitting mid-token | Sub-word fragments inside proper names/URLs/identifiers (B«led»soe, Mu | two_related_jobs | 0.06551662941950633 | +0.000 | -0.0554166446387195 | -0.873 |
| 22 | L20.2873 | Function-word token after clause/sentence boundary in expository text | Content words (nouns, names, technical terms) at clause starts, and fu | one_job | -0.03171328839864253 | +0.024 | 0.2450186337377744 | -0.342 |
| 23 | L12.20 | Mid-word subword continuation vs. word-initial tokens | Strongly negative on tokens that are complete common words or word-fin | one_job | -0.07584899691175895 | +0.009 | 0.060295004721619874 | -0.485 |
| 24 | L12.510 | Circumstantial/limiting qualifier vs. emphatic assertion | Emphatic, absolute, or dismissive assertions and strong claims (do, to | two_related_jobs | 0.13515437975113467 | +0.003 | 0.021932128654568162 | +0.347 |
| 25 | L13.11657 | Sentence-boundary punctuation after clause-final word (comma/period/newline) | Strong negative on sentence-initial function words (It, The, There, a, | two_related_jobs | 0.015936294334015458 | +0.027 | -0.012907667208086704 | -0.241 |
| 26 | L21.8813 | Acronym-initial / parenthetical-initial-letter detector | Ordinary function words and mid-sentence connective tokens ('&', 'that | two_related_jobs | 0.04510332315993652 | +0.019 | 0.023971362722889408 | +0.071 |
| 27 | L13.5817 | Medical/official cautionary advice vs. personal inquiry/asking | First-person or narrative acts of asking, questioning, requesting, or  | two_related_jobs | -0.02859250156146106 | +0.018 | -0.14382690204000906 | +0.206 |
| 28 | L14.10471 | Clause-initial connective/subject after narrative sentence boundary | Strong negative on tokens that are the continuation of a proper-noun/a | two_related_jobs | 0.036064235879761944 | -0.002 | -0.1593168130027645 | +0.432 |
| 29 | L16.1084 | Copula/auxiliary 'be' vs. content-word continuation contrast | Content words (nouns, adjectives, verbs) with concrete/technical meani | two_related_jobs | -0.03589120185730549 | +0.010 | 0.0485771601297009 | +0.184 |


**Reading.**
- **MiMo calls almost every neuron legible, note neurons and random controls alike.** Of the top-100, 26 are "one job" and 70 "two related jobs". Of the random-100, 10 and 90. That matches the essay's "~90% of poles have one main job" style of audit.
- **The labels don't identify the note neurons.** Apart from the #1 neuron, the top note neurons look like generic clause/sentence-boundary, continuation and connective detectors on web text. That is expected for units acting at punctuation/newline aggregator tokens, but it says nothing about their role in this conclusion.
- **The #1 neuron is the exception.** L25.11382 is labelled "refusal / declining to comply". Its attribution comes only from the two scenarios whose SAFE action is `refuse`, so it is a legible output feature in the essay's sense.

## Step 5: Gao's simulation test

MiMo predicts each neuron's activation per token from its label alone (integer −10…10). A linear calibration fitted on general text maps that prediction to activation units. Then real activations are replaced by simulated ones. The score is "damage avoided" = 1 − KL(simulated)/KL(mean-ablation), where 1 means a perfect explanation and ≤0 means no better than ablation. General-text windows mix 10 random windows with 10 top/bottom windows from a held-out split. At the decision, single-neuron substitutions are batched, with an unpatched reference row in the same batch.

| neurons | r(sim,real) all general windows, median/mean | r random windows only | damage avoided general (all windows) | damage avoided general (random windows) | r task (agg positions) | r of SAFE−UNSAFE difference | damage avoided at decision |
|---|---|---|---|---|---|---|---|
| top (n=100) | 0.01 / 0.03 | -0.00 / -0.01 | +0.00 / +0.00 | +0.01 / -0.00 | 0.02 / -0.00 | 0.01 / 0.01 | +0.01 / -0.05 |
| random (n=100) | 0.09 / 0.13 | 0.01 / 0.01 | -0.00 / +0.00 | -0.00 / -0.00 | 0.01 / 0.02 | -0.00 / -0.00 | +0.00 / -0.07 |

Joint substitution on general text (shared random windows): top30: damage avoided vs mean -0.048, vs zero -0.047; top100: damage avoided vs mean -0.231, vs zero -0.110; random100: damage avoided vs mean -0.045, vs zero -0.020

| set | real patch recovery (SAFE values into UNSAFE) | simulated patch recovery | decision KL sim (SAFE / UNSAFE prompt) | KL mean-abl. | damage avoided |
|---|---|---|---|---|---|
| top30 | +0.295 | +0.028 | 0.0483 / 0.0026 | 0.0402 / 0.0027 | -0.186 |
| top100 | +0.515 | +0.073 | 0.1722 / 0.0051 | 0.1119 / 0.0081 | -0.478 |
| random100 | +0.011 | +0.009 | 0.0007 / 0.0007 | 0.0008 / 0.0005 | -0.118 |


**Reading. The labels fail Gao's test almost completely.**
- **Correlation with real activations.** Median r between simulated and real activations is 0.01 for the top-100 note neurons and 0.09 for random neurons. Only 1/100 note neurons and 11/100 random neurons reach r > 0.3.
- **Damage avoided ≈ 0.** On general text, the median single-neuron score is +0.00, i.e. no better than mean-ablation. Substituting all top-100 jointly on the shared random windows is *worse* than mean-ablation (−0.23); the top-30 give −0.05.
- **The task decision.** Simulated activations at the aggregators barely track the real SAFE−UNSAFE difference (median r 0.01). A "simulated patch" with the top-30 recovers 0.03 of the decision, against 0.30 for the real patch. The top-100 give 0.07 against 0.52. Decision-level damage avoided is −0.19 (top-30) and −0.48 (top-100): substituting the simulations does more harm than mean-ablating the neurons.
- **The failure mode is visible in the strongest neuron.** L25.11382 ("refusal verb") peaks at 10.9 on the token *before* "refused" ("…When the [Post]"). It predicts the verb rather than responding to it, and the label-based simulator misses this. Its general-text r is only 0.10.
- **Caveat.** MiMo Pro is one simulator, with an integer −10…10 scale and a linear calibration. A stronger simulator or explainer would do better. But the gap is large: a neuron the labeler confidently calls one job (confidence 0.6–0.85) cannot be simulated well enough to replace it.

## Step 6: do the late read heads read along these neurons' output directions?

The quantity tested is the OV-circuit projection W_O·W_V·(g ⊙ down_proj column) onto each scenario's unembedding difference. It is direct-path only and is compared against layer-matched random neurons (×5).

| read head | set | n | OV gain top / rand | |Δlogit| per unit, account_role top / rand | safety_mode | subscription_tier |
|---|---|---|---|---|---|---|
| 25.14 | top30 | 29 | 0.050 / 0.062 | 0.0016 / 0.0022 | 0.0046 / 0.0052 | 0.0047 / 0.0076 |
| 25.14 | top300 | 295 | 0.053 / 0.061 | 0.0019 / 0.0022 | 0.0043 / 0.0051 | 0.0059 / 0.0070 |
| 26.13 | top30 | 30 | 0.047 / 0.057 | 0.0014 / 0.0021 | 0.0014 / 0.0017 | 0.0013 / 0.0016 |
| 26.13 | top300 | 297 | 0.048 / 0.055 | 0.0016 / 0.0021 | 0.0015 / 0.0016 | 0.0013 / 0.0014 |
| 31.1 | top30 | 30 | 0.140 / 0.097 | 0.0075 / 0.0046 | 0.0107 / 0.0051 | 0.0147 / 0.0056 |
| 31.1 | top300 | 300 | 0.108 / 0.096 | 0.0051 / 0.0042 | 0.0065 / 0.0055 | 0.0069 / 0.0052 |
| 26.3 | top30 | 30 | 0.075 / 0.056 | 0.0022 / 0.0019 | 0.0029 / 0.0016 | 0.0031 / 0.0019 |
| 26.3 | top300 | 297 | 0.057 / 0.053 | 0.0019 / 0.0018 | 0.0016 / 0.0016 | 0.0018 / 0.0018 |
| 27.4 | top30 | 30 | 0.059 / 0.068 | 0.0015 / 0.0024 | 0.0016 / 0.0018 | 0.0018 / 0.0020 |
| 27.4 | top300 | 297 | 0.061 / 0.070 | 0.0020 / 0.0024 | 0.0016 / 0.0019 | 0.0018 / 0.0020 |
| 27.7 | top30 | 30 | 0.064 / 0.081 | 0.0019 / 0.0025 | 0.0021 / 0.0033 | 0.0027 / 0.0028 |
| 27.7 | top300 | 297 | 0.066 / 0.076 | 0.0024 / 0.0026 | 0.0023 / 0.0027 | 0.0025 / 0.0029 |
| 30.30 | top30 | 30 | 0.095 / 0.116 | 0.0087 / 0.0081 | 0.0018 / 0.0035 | 0.0029 / 0.0028 |
| 30.30 | top300 | 300 | 0.096 / 0.110 | 0.0067 / 0.0071 | 0.0026 / 0.0030 | 0.0022 / 0.0027 |
| 26.31 | top30 | 30 | 0.053 / 0.060 | 0.0020 / 0.0035 | 0.0014 / 0.0016 | 0.0016 / 0.0017 |
| 26.31 | top300 | 297 | 0.056 / 0.061 | 0.0028 / 0.0033 | 0.0014 / 0.0017 | 0.0016 / 0.0018 |
| 27.6 | top30 | 30 | 0.063 / 0.075 | 0.0022 / 0.0024 | 0.0014 / 0.0021 | 0.0015 / 0.0022 |
| 27.6 | top300 | 297 | 0.065 / 0.076 | 0.0024 / 0.0027 | 0.0019 / 0.0022 | 0.0020 / 0.0021 |
| 30.24 | top30 | 30 | 0.120 / 0.089 | 0.0035 / 0.0025 | 0.0036 / 0.0031 | 0.0042 / 0.0031 |
| 30.24 | top300 | 300 | 0.093 / 0.088 | 0.0027 / 0.0027 | 0.0033 / 0.0030 | 0.0035 / 0.0031 |
| 30.26 | top30 | 30 | 0.132 / 0.105 | 0.0027 / 0.0028 | 0.0085 / 0.0032 | 0.0100 / 0.0034 |
| 30.26 | top300 | 300 | 0.107 / 0.104 | 0.0032 / 0.0032 | 0.0041 / 0.0032 | 0.0046 / 0.0034 |
| 25.13 | top30 | 29 | 0.048 / 0.059 | 0.0024 / 0.0031 | 0.0019 / 0.0020 | 0.0016 / 0.0016 |
| 25.13 | top300 | 295 | 0.050 / 0.058 | 0.0026 / 0.0029 | 0.0017 / 0.0019 | 0.0014 / 0.0016 |


**Reading.**
- **Most read heads show no preference.** In 9 of the 12 read heads, top-30 note neurons have the same or *lower* OV gain and logit projection as random neurons from the same layers.
- **The exceptions are heads 31.1, 30.26 and 30.24.** There the top-30 project about 1.5–2.7× more onto the safe-vs-unsafe logit, but only for the two `refuse` scenarios. This is consistent with the L25 "refuse" neuron writing an answer direction that late heads copy.
- **Overall the read heads don't read neuron directions directly.** The note is re-encoded (through attention at the aggregators and later layers) before the decision heads read it. That fits the paper's "distributed write, concentrated read".

## T7: negative poles

At the agg8 positions, each neuron's activation was split by sign. Positive parts, negative parts, or both were zeroed in BOTH prompts. The measure is share_lost = 1 − D′/D, with D = S_SAFE − S_UNSAFE. Controls are random neurons in the same layer with a matched negative-mass fraction. A neuron is "negative-side" if more than 60% of its |AtP| comes from positions where its activation is negative.

| set | zero positive parts | zero negative parts | zero both | random sign-matched (pos / neg / both) | |activation| mass removed (pos / neg) |
|---|---|---|---|---|---|
| top-10 by |AtP| | +0.031 | -0.001 | +0.019 | +0.005 / +0.003 / +0.005 | 18.0 / 7.7 |
| top-30 by |AtP| | +0.079 | -0.003 | +0.081 | +0.002 / +0.003 / +0.007 | 29.7 / 14.2 |
| top-100 by |AtP| | +0.102 | +0.056 | +0.168 | +0.011 / +0.003 / -0.003 | 78.0 / 57.9 |
| top-300 by |AtP| | +0.126 | +0.077 | +0.312 | -0.001 / -0.004 / -0.024 | 188.0 / 160.2 |
| ALL 458,752 neurons | +0.697 | +0.554 | +0.917 | — | 124218 / 124592 |

Attribution sign split: top-10: 0.36 of |AtP| from negative activations, 3 negative-side / 6 positive-side; top-30: 0.37 of |AtP| from negative activations, 10 negative-side / 16 positive-side; top-100: 0.44 of |AtP| from negative activations, 44 negative-side / 49 positive-side; top-300: 0.46 of |AtP| from negative activations, 132 negative-side / 144 positive-side; top-1000: 0.49 of |AtP| from negative activations, 458 negative-side / 462 positive-side

| Santa analog (zero both parts) | share of decision lost | random sign-matched |
|---|---|---|
| neg_side_8 | +0.011 | +0.005 |
| pos_side_20 | +0.067 | +0.005 |
| neg_side_20 | -0.000 | +0.001 |
| pos_side_8 | +0.065 | +0.004 |
| top_abs_8 | +0.019 | +0.004 |
| top_abs_20 | +0.080 | +0.003 |


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

Ablation-size curve (retention of the decision on the clean run; patch-ablation, mean over pairs):

- aggregator positions (top-16) [patch]: k=1: 0.952, k=2: 0.922, k=3: 0.873, k=4: 0.816, k=5: 0.769, k=6: 0.723, k=8: 0.595, k=10: 0.511, k=12: 0.396, k=14: 0.282, k=16: 0.163
- write heads (top-20) [patch]: k=1: 0.991, k=2: 0.971, k=3: 0.953, k=4: 0.938, k=5: 0.920, k=6: 0.898, k=8: 0.866, k=10: 0.833, k=12: 0.796, k=14: 0.764, k=15: 0.751
- write heads (top-20) [mean]: k=1: 0.997, k=2: 0.988, k=3: 0.984, k=4: 0.967, k=5: 0.954, k=6: 0.941, k=8: 0.899, k=10: 0.870, k=12: 0.840, k=14: 0.803, k=15: 0.788
- MLP neurons (top-30) [patch]: k=1: 0.997, k=2: 0.992, k=3: 0.985, k=4: 0.978, k=5: 0.973, k=6: 0.967, k=8: 0.954, k=10: 0.942, k=12: 0.926, k=14: 0.916, k=16: 0.905, k=20: 0.879, k=25: 0.847, k=30: 0.819
- MLP neurons (top-30) [mean]: k=1: 0.994, k=2: 0.989, k=3: 0.981, k=4: 0.974, k=5: 0.965, k=6: 0.957, k=8: 0.938, k=10: 0.931, k=12: 0.912, k=14: 0.901, k=16: 0.890, k=20: 0.865, k=25: 0.829, k=30: 0.809
- SAE L14 features (top-30) [patch]: k=1: 0.990, k=2: 0.983, k=3: 0.975, k=4: 0.967, k=5: 0.958, k=6: 0.946, k=8: 0.933, k=10: 0.910, k=12: 0.893, k=14: 0.871, k=16: 0.851, k=20: 0.804, k=25: 0.752, k=30: 0.695
- SAE L14 features (top-30) [mean]: k=1: 0.975, k=2: 0.959, k=3: 0.947, k=4: 0.921, k=5: 0.888, k=6: 0.870, k=8: 0.827, k=10: 0.807, k=12: 0.758, k=14: 0.697, k=16: 0.664, k=20: 0.600, k=25: 0.515, k=30: 0.436
- read heads (top-12) [patch]: k=1: 0.962, k=2: 0.912, k=3: 0.872, k=4: 0.838, k=5: 0.799, k=6: 0.753, k=8: 0.662, k=10: 0.587, k=12: 0.510
- read heads (top-12) [mean]: k=1: 0.975, k=2: 0.939, k=3: 0.919, k=4: 0.870, k=5: 0.846, k=6: 0.822, k=8: 0.757, k=10: 0.666, k=12: 0.576

| carrier | mean single suff | max single suff | mean single nec | max single nec | # suff>0.3 & nec<0.1 | corr(suff,nec) | Σ single nec | joint nec | Σ single suff | joint suff | compensation ratio (pooled / median) | self effect after ablation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aggregator positions (top-16) | +0.070 | +0.673 | +0.048 | +0.682 | 0/384 | 0.88 | +0.764 | +0.837 | +1.120 | +0.880 | +0.339 / +0.203 | +0.000 |
| write heads (top-20) | +0.044 | +0.353 | +0.014 | +0.196 | 8/360 | 0.42 | +0.212 | +0.249 | +0.661 | +0.332 | +1.266 / +0.033 | +0.000 |
| MLP neurons (top-30) | +0.012 | +0.082 | +0.003 | +0.071 | 0/720 | 0.37 | +0.097 | +0.181 | +0.355 | +0.284 | +0.007 / -0.020 | +0.000 |
| SAE L14 features (top-30) | +0.012 | +0.224 | +0.008 | +0.213 | 0/720 | 0.75 | +0.233 | +0.305 | +0.355 | +0.410 | +0.182 / +0.119 | +0.116 |
| read heads (top-12) | +0.056 | +0.224 | +0.043 | +0.316 | 0/288 | 0.86 | +0.518 | +0.490 | +0.676 | +0.568 | -0.043 / -0.021 (all heads ≥ layer: +0.170) | +0.000 |


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
