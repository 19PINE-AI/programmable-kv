#!/bin/bash
# Full GPU + LLM pipeline, in the order it was run for the paper (single RTX PRO 6000, ~1 day).
# Outputs land in ../results (override with FOLLOWUP_OUT). Large intermediates (*.pt, *.npy, the
# MiMo request cache, sim_general_*.json) are written there but are git-ignored.
# Labelling/simulation needs an OpenAI-compatible endpoint described by $MIMO_ENV
# (MIMO_URLS=url1,url2  MIMO_KEY_2=...  MIMO_KEY_3=...  MIMO_MODEL=...); keys are never logged.
set -e
cd "$(dirname "$0")"
export GPU_GB=${GPU_GB:-30}
python t1_neurons.py                                   # steps 2-3: AtP ranking + neuron patching (Llama)
python t1_corpus.py                                    # step 4a: top/random neurons over fineweb-edu
python t1_mimo.py label                                # step 4b: labels (MiMo)
python t1_gao.py prep                                  # task prompts + aggregator positions (no GPU)
python t1_mimo.py simgen                               # step 5: simulations on general text
python t1_mimo.py simtask                              # step 5: simulations on task prompts
python t1_t7.py                                        # T7: sign split
python t1_gao.py ov                                    # step 6: OV projection of read heads
python t1_neurons.py --model Qwen/Qwen3-8B --tag qwen3_8b   # step 7: Qwen3-8B replication
python t1_split.py                                     # attention vs MLP at agg8
python t1_mimo.py simgen && python t1_fix.py           # re-run simgen (cached) + repair unparsable labels/NaN sims
python t1b_redundancy.py                               # T1b: joint vs summed single ablation (bf16-batch-fixed)
python t1_gao.py eval                                  # step 5: Gao simulation-substitution test
python additivity_ci.py                                # CPU: pooled joint/sum ratios + bootstrap CIs
./build_report.sh                                      # CPU: regenerate ../REPORT.md
