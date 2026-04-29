#!/bin/bash
set -e

# ===== 想跑哪几个 SFT ckpt =====
CKPTS=(225 450 675)

# ===== 实验配置 =====
REWARD_THRESHOLD=0.95     # mining 阈值: reward < 这个 → hard
REWARD_MODE=chain_binary  # 训练 reward
MAX_STEPS=400
TEMPERATURE=0.9           # 略提高采样多样性
LR=5e-7

############################################
# Step 1: Mine hard examples for each ckpt
############################################
for CKPT in "${CKPTS[@]}"; do
    SFT_DIR=checkpoints/qwen2.5-3b-2hop-sft/checkpoint-${CKPT}
    HARD_DIR=hard_data_from_${CKPT}
    HARD_FILE=${HARD_DIR}/train_2hop_hard.jsonl

    if [ ! -f "${HARD_FILE}" ]; then
        echo ""
        echo "############################################"
        echo "# Mine hard examples from SFT ckpt-${CKPT}"
        echo "############################################"
        mkdir -p ${HARD_DIR}
        python mine_hard_examples.py \
            --model_dir ${SFT_DIR} \
            --input_file prepared_data_2hop/train_2hop.jsonl \
            --output_file ${HARD_FILE} \
            --reward_threshold ${REWARD_THRESHOLD} \
            --reward_mode ${REWARD_MODE} \
            --include_bridge_count \
            --batch_size 16 \
            --max_new_tokens 128 \
            --max_input_len 4096
    else
        echo "[skip mining] ${HARD_FILE} already exists"
    fi
done

############################################
# Step 2: GRPO from each SFT checkpoint
############################################
for CKPT in "${CKPTS[@]}"; do
    SFT_DIR=checkpoints/qwen2.5-3b-2hop-sft/checkpoint-${CKPT}
    HARD_FILE=hard_data_from_${CKPT}/train_2hop_hard.jsonl
    OUT_DIR=checkpoints/qwen2.5-3b-grpo-from-${CKPT}-v2

    if [ -d "${OUT_DIR}/final" ]; then
        echo "[skip RL] ${OUT_DIR}/final already exists"
        continue
    fi

    echo ""
    echo "############################################"
    echo "# GRPO from SFT ckpt-${CKPT} on hard examples"
    echo "############################################"
    torchrun --nproc_per_node=2 rl_grpo_2hop.py \
        --model_name_or_path     ${SFT_DIR} \
        --ref_model_name_or_path ${SFT_DIR} \
        --train_file ${HARD_FILE} \
        --eval_file  prepared_data_2hop/eval_2hop.jsonl \
        --output_dir ${OUT_DIR} \
        --include_bridge_count \
        --reward_mode ${REWARD_MODE} \
        --bridge_gate_floor 0.0 \
        --max_steps ${MAX_STEPS} \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --num_generations 4 \
        --learning_rate ${LR} \
        --kl_beta 0.02 \
        --temperature ${TEMPERATURE} \
        --top_p 0.95 \
        --eval_steps 100 \
        --eval_subset_size 200 \
        --save_steps 100 \
        --logging_steps 10 \
        --seed 42
done

############################################
# Step 3: Eval (with include_bridge_count, oracle hop count)
############################################
for CKPT in "${CKPTS[@]}"; do
    OUT_DIR=checkpoints/qwen2.5-3b-grpo-from-${CKPT}-v2
    echo ""
    echo "############################################"
    echo "# Eval (oracle hop) RL from ckpt-${CKPT}"
    echo "############################################"
    python eval_compositional.py \
        --auto_discover ${OUT_DIR} \
        --include_bridge_count \
        --batch_size 8 \
        --max_new_tokens 128 \
        --out_dir eval_results_grpo_from_${CKPT}_v2_oracle
done



echo ""
echo "All done!"