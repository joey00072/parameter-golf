#!/bin/bash
set -e

# Simulate run.sh (8xH100, train_batch_tokens=786432) on a single GPU via microbatching.
# Effective batch is identical: GRAD_ACCUM_STEPS × micro-batch = 786432 tokens/step.
#
# GRAD_ACCUM_STEPS=128 → micro-batch = 6 seqs × 1024 = 6144 tokens  (fits on ~7.6GB GPU)
# GRAD_ACCUM_STEPS=64  → micro-batch = 12 seqs × 1024              (OOMs)
# Each training step takes ~128 micro-batch passes — slow but correct.

BIGLU_VOCAB_SIZE=2048 \
BIGLU_MULT=1.0 \
TRAIN_SEQ_LEN=1024 \
GRAD_ACCUM_STEPS=128 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=1 train_gpt.py
