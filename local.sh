#!/bin/bash
set -e

# Local 1-GPU test run — same as run_submission.sh but single node, reduced batch
# train_batch_tokens=65536 (vs 786432 for 8xH100), train_seq_len=1024

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TRAIN_BATCH_TOKENS=65536 \
TRAIN_SEQ_LEN=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
