#!/bin/bash
# run_step3_rl_anchored.sh
# Phase 3: Anchored GRPO from SFT-50, chain_binary reward + KL=0.05
# v3.1: + PYTORCH_CUDA_ALLOC_CONF, set -o pipefail, auto-backup of old run

set -e
set -o pipefail   # ← 让 tee 不吞掉错误，崩溃立即停

# ★ 关键环境变量：让 PyTorch 用新分配器，对 GRPO 这种频繁 alloc/free 极有效
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false

# ============ 配置 ============
N_GPUS=2

SFT_CKPT="checkpoints/qwen2.5-3b-2hop-sft-light/checkpoint-50"
TRAIN_FILE="prepared_data_2hop/train_2hop_mined.jsonl"
EVAL_FILE="prepared_data_2hop/eval_2hop.jsonl"

EXP_NAME="qwen2.5-3b-rl-anchored-chain"
OUTPUT_DIR="checkpoints/${EXP_NAME}"

# Train
MAX_STEPS=300
PER_DEVICE_BSZ=2
GRAD_ACCUM=4
NUM_GEN=4
LR=1e-6
WARMUP_RATIO=0.05

# RL
REWARD_MODE="chain_binary"
KL_BETA=0.05
TEMPERATURE=0.9
TOP_P=0.95
CLIP_RANGE=0.2

# Eval-during-training
EVAL_STEPS=50
EVAL_SUBSET=200
SAVE_STEPS=50
LOGGING_STEPS=10

# ★ NEW: chunk size for token_logprobs.
#   512 ≈ 5 GB peak per chunk on Qwen2.5-3B.
#   If still OOM after this fix, lower to 256 or 128.
LOGPROB_CHUNK=512
# ==============================

mkdir -p logs

# ★ 自动备份旧的 RL 输出，避免与上次失败的 checkpoint-50 / best 混淆
if [ -d "${OUTPUT_DIR}" ]; then
    BACKUP="${OUTPUT_DIR}_FAILED_$(date +%Y%m%d_%H%M%S)"
    echo "=========================================================="
    echo " Found existing ${OUTPUT_DIR}"
    echo " Backing up to:  ${BACKUP}"
    echo "=========================================================="
    mv "${OUTPUT_DIR}" "${BACKUP}"
fi

echo "=========================================================="
echo " STEP 3: Anchored GRPO (chain_binary + KL=${KL_BETA})"
echo " From:    ${SFT_CKPT}"
echo " Data:    ${TRAIN_FILE} (mined hard subset)"
echo " Output:  ${OUTPUT_DIR}"
echo " Chunk:   logprob_chunk_size=${LOGPROB_CHUNK}"
echo " Alloc:   PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo " Started: $(date)"
echo "=========================================================="

accelerate launch \
    --num_processes ${N_GPUS} \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
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
        \
        --logprob_chunk_size         ${LOGPROB_CHUNK} \
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