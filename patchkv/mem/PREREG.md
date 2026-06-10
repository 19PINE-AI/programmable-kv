# Pre-registration — memory-KV experiments

Fixed **before** confirmatory runs. Analyses not listed here are exploratory and labeled as such.

## Hypotheses (confirmatory)
- **H1** (E1): late precompiled memory has lower gated-decision accuracy than full-recompute,
  and the deficit increases with integration depth `n_facts`. Primary test: GEE-logistic
  coefficient on `placement[late]` (one-sided, < 0) and on the `placement:n_facts`
  interaction (one-sided, < 0).
- **H1b** (E1): the `placement[late]` deficit is smaller under CoT than under direct
  decoding. Primary test: `placement:reasoning` interaction (one-sided).
- **H2** (E2): precompiled+repositioned memory is **equivalent** to full recompute in
  decision agreement. Primary test: TOST, margin δ = max(0.03, 2× oracle test–retest
  disagreement). Secondary: logit cosine ≥ 0.98 (equivalence).
- **H3** (E3): `in_place < erratum` and `in_place < recompile_chunk` in decision recovery
  (one-sided superiority); `erratum ≈ recompile_chunk` (TOST). Scale: recovery of
  `in_place` decreases with log(params) (the stickiness law).
- **H4** (E4): decision agreement decreases as memory is split into more sub-blocks **only
  when** required facts are split across blocks (`S:cross_block` interaction < 0).
- **H5** (E5): cumulative TTFT(precompiled-editable) < min(front-recompute, end-recompute)
  for edit-rate r ≤ 0.5, **and** decision agreement vs full reprefill is equivalent (δ as H2).

## Primary endpoints
- Binary gated-decision correctness (vs oracle clean prefill) and decision agreement
  (method vs full recompute).
- Continuous: logit cosine and recovery fraction (secondary/mechanistic).

## Sample size
- ≥ 480 paired decisions per (model × condition) cell for equivalence claims.

## Inclusion / exclusion
- A (model, task-family) cell enters primary analysis iff oracle accuracy ≥ 0.80.
- Models that OOM at a given L_mem are excluded at that length only; reported.

## Inference
- Cluster bootstrap (10⁴, persona-level) for CIs; GEE-logistic (exchangeable, cluster=persona)
  for moderated effects; McNemar-exact for paired binary contrasts; TOST for equivalence.
- Multiplicity: BH-FDR q=0.05 within each experiment; Holm for headline equivalence.
- Effect sizes (risk difference, OR, Cohen's h) reported with all p-values.
