#!/bin/bash
set -e

CKPTS=(225 450 675)

# ========== RL 训练 ==========
for CKPT in "${CKPTS[@]}"; do
    echo "############################################"
    echo "# RL from SFT checkpoint-${CKPT}"
    echo "############################################"

    torchrun --nproc_per_node=2 rl_grpo_2hop.py \
        --model_name_or_path     checkpoints/qwen2.5-3b-2hop-sft/checkpoint-${CKPT} \
        --ref_model_name_or_path checkpoints/qwen2.5-3b-2hop-sft/checkpoint-${CKPT} \
        --train_file prepared_data_2hop/train_2hop.jsonl \
        --eval_file  prepared_data_2hop/eval_2hop.jsonl \
        --output_dir checkpoints/qwen2.5-3b-grpo-from-${CKPT} \
        --reward_mode process_final \
        --w_format 0.10 --w_bridge 0.45 --w_final 0.45 \
        --bridge_gate_floor 0.15 \
        --max_steps 400 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --num_generations 4 \
        --learning_rate 5e-7 \
        --kl_beta 0.02 \
        --temperature 0.8 \
        --top_p 0.95 \
        --eval_steps 99999 \
        --save_steps 100 \
        --logging_steps 10 \
        --seed 42
done

# ========== 评估所有 RL checkpoints ==========
for CKPT in "${CKPTS[@]}"; do
    echo "############################################"
    echo "# Eval RL from SFT checkpoint-${CKPT}"
    echo "############################################"

    python eval_compositional.py \
        --auto_discover checkpoints/qwen2.5-3b-grpo-from-${CKPT} \
        --include_root \
        --batch_size 8 \
        --max_new_tokens 128 \
        --out_dir eval_results_grpo_from_${CKPT}
done