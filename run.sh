#!/bin/bash
set -e

# PR #180 + interleaved BigLU (every other layer)
# mlp_mult=3.0 (hidden=1536) for MLP layers (unchanged from PR #180)
# biglu_mult=1.0 (hidden=512) for BigLU layers → same params as MLP (1,572,864 each)
# BigLU: 2×512×512 + 2048×512 = 524,288 + 1,048,576 = 1,572,864 = MLP params ✓

BIGLU_VOCAB_SIZE=2048 \
BIGLU_MULT=1.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
