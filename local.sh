#!/bin/bash
set -e

# Simulate run.sh (8xH100, train_batch_tokens=786432) on a single GPU via microbatching.
# Effective batch is identical: GRAD_ACCUM_STEPS steps × micro-batch = 786432 tokens/step.
#
# TRAIN_SEQ_LEN=1024 (half of run.sh's 2048) to further reduce per-step memory.
# Micro-batch sizes at each GRAD_ACCUM_STEPS level:
#   GRAD_ACCUM_STEPS=64  → 12 seqs × 1024 = 12288 tokens  (try first)
#   GRAD_ACCUM_STEPS=128 →  6 seqs × 1024 =  6144 tokens
#   GRAD_ACCUM_STEPS=192 →  4 seqs × 1024 =  4096 tokens
#   GRAD_ACCUM_STEPS=384 →  2 seqs × 1024 =  2048 tokens
#   GRAD_ACCUM_STEPS=768 →  1 seq  × 1024 =  1024 tokens  (minimum)
#
# If you get OOM, increase GRAD_ACCUM_STEPS to the next level above.

BIGLU_VOCAB_SIZE=2048 \
BIGLU_MULT=1.0 \
TRAIN_SEQ_LEN=1024 \
GRAD_ACCUM_STEPS=64 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=1 train_gpt.py
