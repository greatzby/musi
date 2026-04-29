#!/bin/bash
set -e

CKPTS=(450 675)
# ========== 评估所有 RL checkpoints ==========
for CKPT in "${CKPTS[@]}"; do
    echo "############################################"
    echo "# Eval RL from SFT checkpoint-${CKPT}"
    echo "############################################"

    python eval_compositional.py \
        --auto_discover checkpoints/qwen2.5-3b-grpo-from-${CKPT} \
        --batch_size 8 \
        --max_new_tokens 128 \
        --out_dir eval_results_grpo_from_${CKPT}
done