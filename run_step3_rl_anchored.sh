#!/bin/bash
# run_step3_rl_anchored.sh
# Phase 3: Anchored GRPO from SFT-50, chain_binary reward + KL=0.05
set -e

# ============ 配置 ============
N_GPUS=2

SFT_CKPT="checkpoints/qwen2.5-3b-2hop-sft-light/checkpoint-50"
TRAIN_FILE="prepared_data_2hop/train_2hop_mined.jsonl"   # ← 用 mined 数据
EVAL_FILE="prepared_data_2hop/eval_2hop.jsonl"

EXP_NAME="qwen2.5-3b-rl-anchored-chain"
OUTPUT_DIR="checkpoints/${EXP_NAME}"

# Train
MAX_STEPS=300
PER_DEVICE_BSZ=2          # local prompts/GPU
GRAD_ACCUM=4              # → effective 16 prompts × 4 gens = 64 seq/update
NUM_GEN=4
LR=1e-6                   # ↑ from default 5e-7, 仍保守
WARMUP_RATIO=0.05

# RL
REWARD_MODE="chain_binary"
KL_BETA=0.05              # ↑ from default 0.02 (paper uses 0.2 with anneal)
TEMPERATURE=0.9
TOP_P=0.95
CLIP_RANGE=0.2

# Eval-during-training
EVAL_STEPS=50
EVAL_SUBSET=200
SAVE_STEPS=50
LOGGING_STEPS=10
# ==============================

mkdir -p logs

echo "=========================================================="
echo " STEP 3: Anchored GRPO (chain_binary + KL=${KL_BETA})"
echo " From:    ${SFT_CKPT}"
echo " Data:    ${TRAIN_FILE} (mined hard subset)"
echo " Output:  ${OUTPUT_DIR}"
echo " Started: $(date)"
echo "=========================================================="

accelerate launch \
    --num_processes ${N_GPUS} \
    --num_machines 1 \
    --mixed_precision bf16 \
    rl_grpo_2hop.py \
        --model_name_or_path     ${SFT_CKPT} \
        --ref_model_name_or_path ${SFT_CKPT} \
        --train_file             ${TRAIN_FILE} \
        --eval_file              ${EVAL_FILE} \
        --output_dir             ${OUTPUT_DIR} \
        \
        --max_steps                  ${MAX_STEPS} \
        --per_device_train_batch_size ${PER_DEVICE_BSZ} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --num_ppo_epochs             1 \
        --learning_rate              ${LR} \
        --warmup_ratio               ${WARMUP_RATIO} \
        --max_grad_norm              1.0 \
        --bf16 \
        --gradient_checkpointing \
        --seed                       42 \
        \
        --num_generations            ${NUM_GEN} \
        --max_input_len              2048 \
        --max_new_tokens             128 \
        --temperature                ${TEMPERATURE} \
        --top_p                      ${TOP_P} \
        \
        --reward_mode                ${REWARD_MODE} \
        --kl_beta                    ${KL_BETA} \
        --clip_range                 ${CLIP_RANGE} \
        \
        --eval_steps                 ${EVAL_STEPS} \
        --eval_subset_size           ${EVAL_SUBSET} \
        --save_steps                 ${SAVE_STEPS} \
        --save_best \
        --logging_steps              ${LOGGING_STEPS} \
    2>&1 | tee logs/rl_${EXP_NAME}.log

echo ""
echo "=========================================================="
echo " RL DONE at $(date)"
echo "=========================================================="
echo ""
echo " Now running full OOD evaluation on best + final checkpoints..."
echo ""

# 自动评估关键 checkpoint
python eval_compositional.py \
    --auto_discover ${OUTPUT_DIR} \
    --out_dir       eval_results_${EXP_NAME} \
    --batch_size    16 \
    --splits        2hop 3hop_linear 4hop_linear \
    2>&1 | tee logs/eval_${EXP_NAME}.log

echo ""
echo "=========================================================="
echo " ALL DONE at $(date)"
echo "=========================================================="