#!/bin/bash
set -e

# PR #180 local 1-GPU test run
# Reduces batch tokens to fit a single GPU (65536 vs 786432 for 8xH100)
# GRAD_ACCUM logic: 786432 / 8 GPUs = 98304 tokens/step → approximate with 65536

TRAIN_BATCH_TOKENS=65536 \
TRAIN_SEQ_LEN=1024 \
MAX_WALLCLOCK_SECONDS=120 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=1 train_gpt.py
