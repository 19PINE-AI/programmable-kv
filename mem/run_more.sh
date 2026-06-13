#!/bin/bash
# Remaining experiments in priority order (P1..P5), robust to intermittent GPU contention.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
LOG=results/more_run.log; : > $LOG
wait_gpu() { for i in $(seq 1 240); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "${free:-0}" -ge "$1" ] && return 0; echo "  [wait] ${free}MiB free, need $1 (try $i)" >> $LOG; sleep 60
  done; return 1; }
retry() { minfree=$1; shift
  for attempt in $(seq 1 10); do wait_gpu "$minfree" || true
    echo "=== attempt $attempt: $* ($(date +%H:%M:%S)) ===" | tee -a $LOG
    if timeout 50000 python "$@" >>$LOG 2>&1; then echo "ok $*" | tee -a $LOG; return 0; fi
    echo "  attempt $attempt failed; backoff" | tee -a $LOG; sleep 120
  done; echo "FAIL (all attempts) $*" | tee -a $LOG; return 1; }

# ---------- P1: long-memory E1 (16k, 32k) — pre-digestion where it should bite ----------
retry 30000 run_e1.py --model Qwen/Qwen3-4B --n 32 --facts 1,8 --mtotals 880,1700 --regimes cot --traj_turns 2 --attn flash_attention_2 --tag Qwen3-4B-long
retry 40000 run_e1.py --model unsloth/Meta-Llama-3.1-8B-Instruct --n 32 --facts 1,8 --mtotals 880,1700 --regimes cot --traj_turns 2 --attn flash_attention_2 --tag Llama-8B-long

# ---------- P2: power top-up (overwrite the underpowered main files at higher N) ----------
retry 30000 run_e3.py --model Qwen/Qwen3-4B --n 480 --nfacts 2 --regime cot
retry 30000 run_e1.py --model Qwen/Qwen3-4B --n 192 --facts 1,2,4 --mtotals 24,120 --regimes direct,cot
retry 30000 run_e5.py --model Qwen/Qwen3-4B --sessions 120 --turns 12 --decide_every 3 --edit_rate 0.25 --mtotal 120

# ---------- P3: keystone / locality-knockout on memory (direct mode, 70B where decisions vary) ----------
retry 50000 run_e3.py --model unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit --n 64 --nfacts 2 --regime direct --methods stale,in_place,selective@1,selective@4,selective@16,recompile_chunk,full_recompute --tag kstmp
[ -f results/e3_kstmp.jsonl ] && mv results/e3_kstmp.jsonl results/keystone_Llama70B_direct.jsonl

# ---------- P4: cross-block dependency (relevant facts within one block vs split) ----------
retry 30000 run_e4.py --model Qwen/Qwen3-4B --n 200 --nfacts 4 --mtotal 60 --S 1,4,16 --layout contiguous --tag cbc
[ -f results/e4_cbc.jsonl ] && mv results/e4_cbc.jsonl results/e4cb_Qwen3-4B_contiguous.jsonl
retry 30000 run_e4.py --model Qwen/Qwen3-4B --n 200 --nfacts 4 --mtotal 60 --S 1,4,16 --layout spread --tag cbs
[ -f results/e4_cbs.jsonl ] && mv results/e4_cbs.jsonl results/e4cb_Qwen3-4B_spread.jsonl

# ---------- P5: E3 selective@K full sweep (minimal-recompute / stickiness law) ----------
retry 30000 run_e3.py --model Qwen/Qwen3-4B --n 64 --nfacts 2 --regime cot --methods stale,in_place,selective@1,selective@2,selective@4,selective@8,selective@16,selective@32,selective@64,full_recompute --tag kstmp2
[ -f results/e3_kstmp2.jsonl ] && mv results/e3_kstmp2.jsonl results/ksweep_Qwen3-4B.jsonl

python analyze.py >>$LOG 2>&1 && echo "ok analyze" | tee -a $LOG || echo "FAIL analyze" | tee -a $LOG
python make_figs.py >>$LOG 2>&1 && echo "ok figs" | tee -a $LOG || echo "FAIL figs" | tee -a $LOG
echo "MORE_ALL_DONE $(date)" | tee -a $LOG
