#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# run_submission.sh — Train 3 seeds and create a submission record
# Usage: bash run_submission.sh "MySubmissionName"
# ---------------------------------------------------------------------------

NAME="${1:-BigLU_11L}"
DATE="$(date +%Y-%m-%d)"
RECORD_DIR="records/track_10min_16mb/${DATE}_${NAME// /_}"

echo "=== Submission: $NAME ==="
echo "=== Record dir: $RECORD_DIR ==="

mkdir -p "$RECORD_DIR"

SEEDS=(42 1337 2024)
declare -A BPB
declare -A BYTES

for SEED in "${SEEDS[@]}"; do
    LOG="$RECORD_DIR/train_seed${SEED}.log"
    echo ""
    echo ">>> Running seed=$SEED  (log: $LOG)"

    PYTORCH_ALLOC_CONF=expandable_segments:True \
    SEED=$SEED \
    .venv/bin/torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$LOG"

    # Parse final sliding-window s64 bpb (most accurate metric)
    BPB[$SEED]=$(grep "final_int6_sliding_window_s64_exact" "$LOG" | grep -oP "val_bpb:\K[0-9.]+" | tail -1)
    # Fall back to roundtrip bpb if sliding window s64 not present
    if [ -z "${BPB[$SEED]}" ]; then
        BPB[$SEED]=$(grep "final_int6_sliding_window_exact\|final_int6_roundtrip_exact" "$LOG" | grep -oP "val_bpb:\K[0-9.]+" | tail -1)
    fi

    BYTES[$SEED]=$(grep "Total submission size" "$LOG" | grep -oP ":\s*\K[0-9]+" | tail -1)

    echo "  seed=$SEED  val_bpb=${BPB[$SEED]}  bytes=${BYTES[$SEED]}"
done

# Compute mean bpb across 3 seeds
MEAN_BPB=$(python3 -c "
vals = [${BPB[42]}, ${BPB[1337]}, ${BPB[2024]}]
mean = sum(vals) / len(vals)
std  = (sum((v - mean)**2 for v in vals) / len(vals))**0.5
print(f'{mean:.5f} {std:.5f}')
")
MEAN=$(echo $MEAN_BPB | cut -d' ' -f1)
STD=$(echo  $MEAN_BPB | cut -d' ' -f2)

echo ""
echo "=== Results ==="
echo "  seed=42   val_bpb=${BPB[42]}  bytes=${BYTES[42]}"
echo "  seed=1337 val_bpb=${BPB[1337]}  bytes=${BYTES[1337]}"
echo "  seed=2024 val_bpb=${BPB[2024]}  bytes=${BYTES[2024]}"
echo "  mean=${MEAN}  std=${STD}"

# Copy train_gpt.py snapshot
cp train_gpt.py "$RECORD_DIR/train_gpt.py"

# Write submission.json
cat > "$RECORD_DIR/submission.json" <<EOF
{
  "name": "$NAME",
  "val_loss": $MEAN,
  "bytes_total": ${BYTES[42]},
  "blurb": "11 layers with BigLU (bigram-gated MLP, hidden=0.5x, vocab=5120) + per-layer prime hashing. Int6+zstd quantization. Mean of 3 seeds: ${MEAN} (std ${STD}). SmearGate + OrthoInit + EMA + LN_scale + XSA last 4 + RoPE_dims=16 + LateQAT.",
  "author": "joey00072",
  "github_id": "joey00072",
  "date": "$DATE"
}
EOF

# Write README.md
cat > "$RECORD_DIR/README.md" <<EOF
# $NAME

**val_bpb: $MEAN** (mean of 3 seeds, sliding window stride=64, post int6+zlib quantization roundtrip)

## Run Command

\`\`\`bash
# Train + evaluate (all params baked in as defaults)
torchrun --standalone --nproc_per_node=8 train_gpt.py
\`\`\`

## 3-Seed Results

| Seed | val_bpb | artifact_bytes |
|------|---------|----------------|
| 42   | ${BPB[42]} | ${BYTES[42]} |
| 1337 | ${BPB[1337]} | ${BYTES[1337]} |
| 2024 | ${BPB[2024]} | ${BYTES[2024]} |
| **Mean** | **$MEAN** | |
| **Std**  | **$STD** | |

## Key Innovations

### BigLU — Bigram-Gated MLP
Replaces standard MLP at every layer:
\`\`\`
out = down(silu(up(x)) * bigram(token_ids))
\`\`\`
- hidden = 0.5 × dim (256), bigram vocab = 5120 → same param count as MLP (mlp_mult=3)
- Each layer uses distinct prime pair for bigram hash → uncorrelated collisions across depth
- BigLU params trained with AdamW at 10× base LR (not Muon)

## Architecture
- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- BigLU MLP: hidden=256, bigram_vocab=5120, per-layer primes
- SmearGate, BigramHash(2048, dim=128) at input
- U-Net skip connections, tied embeddings
- XSA on last 4 layers, RoPE dims=16, LN scale

## Training Hyperparameters
- Muon: matrix_lr=0.025, WD=0.04, momentum=0.99
- AdamW embeddings/scalars: WD=0.04, tied_embed_lr=0.035
- BigLU AdamW: lr=10× base
- warmdown=3000, iterations=9000, warmup=20
- seq_len=2048, batch=786K tokens
- EMA decay=0.997, LateQAT threshold=0.1
- Sliding window eval stride=64
EOF

echo ""
echo "=== Record written to $RECORD_DIR ==="
echo "  submission.json  README.md  train_gpt.py  train_seed{42,1337,2024}.log"
