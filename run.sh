#!/bin/bash
set -e

# PR #180 + interleaved BigLU (every other layer)
# Same total MLP params as PR #180: mlp_mult=1.5 (hidden=768)
# 5 MLP + 5 BigLU(vocab=2048, dim=hidden, scale=1) = same 15.7M MLP params
# Odd layers (1,3,5,7,9) → MLP | Even layers (0,2,4,6,8) → BigLU

MLP_MULT=1.5 \
BIGLU_VOCAB_SIZE=2048 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
