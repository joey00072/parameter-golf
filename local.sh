#!/bin/bash
set -e

# 1-GPU local test: PR #180 + interleaved BigLU

MLP_MULT=1.5 \
BIGLU_VOCAB_SIZE=2048 \
TRAIN_BATCH_TOKENS=65536 \
TRAIN_SEQ_LEN=1024 \
MAX_WALLCLOCK_SECONDS=120 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=1 train_gpt.py
