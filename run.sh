#!/bin/bash
set -e

# PR #180 reproduction — 10L Int5-MLP + BigramHash(10240) + SWA(0.4) + WD=0.04
# 3-seed mean val_bpb: 1.14276 | Best seed: 1.14260 (seed 2024)
# All hyperparameters are defaults in train_gpt.py — no env vars needed.

# Run with default seed (42):
#   bash run.sh
# Run with specific seed:
#   SEED=1337 bash run.sh

.venv/bin/torchrun --standalone --nproc_per_node=8 train_gpt.py
