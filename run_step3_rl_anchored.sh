#!/bin/bash
# run_step3_rl_anchored.sh
# Phase 3: Anchored GRPO from SFT-50, chain_binary reward + KL=0.05
# v3.2: per_device=1 + accum=8 (halve forward batch), aggressive empty_cache

set -e
set -o pipefail

# expandable_segments 你的平台不支持，但留着也无害
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1     # ← 新名字（旧的 NCCL_ASYNC_ERROR_HANDLING 已 deprecated）
export TOKENIZERS_PARALLELISM=false

# ============ 配置 ============
N_GPUS=2

SFT_CKPT="checkpoints/qwen2.5-3b-2hop-sft-light/checkpoint-50"
TRAIN_FILE="prepared_data_2hop/train_2hop_mined.jsonl"
EVAL_FILE="prepared_data_2hop/eval_2hop.jsonl"

EXP_NAME="qwen2.5-3b-rl-anchored-chain"
OUTPUT_DIR="checkpoints/${EXP_NAME}"

# Train  ★ v3.2: 单卡 micro batch 砍半，accum 翻倍，有效 batch 不变
MAX_STEPS=300
PER_DEVICE_BSZ=1          # ← 从 2 改到 1
GRAD_ACCUM=8              # ← 从 4 改到 8 (effective: 1 prompt × 8 accum × 2 GPU × 4 gen = 64 seq/update, 不变)
NUM_GEN=4
LR=1e-6
WARMUP_RATIO=0.05

# RL
REWARD_MODE="chain_binary"
KL_BETA=0.05
TEMPERATURE=0.9
TOP_P=0.95
CLIP_RANGE=0.2

# Eval
EVAL_STEPS=50
EVAL_SUBSET=200
SAVE_STEPS=50
LOGGING_STEPS=10

# Memory ★ v3.2
LOGPROB_CHUNK=256          # log_softmax 时间维 chunk 大小
EMPTY_CACHE_STEPS=25       # 每 N optimizer step 强制清缓存
# ==============================

mkdir -p logs

# 自动备份旧目录（含上次保存的 best/EM=70.00 这次不会丢）
if [ -d "${OUTPUT_DIR}" ]; then
    BACKUP="${OUTPUT_DIR}_FAILED_$(date +%Y%m%d_%H%M%S)"
    echo "=========================================================="
    echo " Found existing ${OUTPUT_DIR}"
    echo " Backing up to:  ${BACKUP}"
    echo "=========================================================="
    mv "${OUTPUT_DIR}" "${BACKUP}"
fi

echo "=========================================================="
echo " STEP 3: Anchored GRPO (chain_binary + KL=${KL_BETA}) v3.2"
echo " From:    ${SFT_CKPT}"
echo " Data:    ${TRAIN_FILE}"
echo " Output:  ${OUTPUT_DIR}"
echo " Memory:  per_dev=${PER_DEVICE_BSZ}  accum=${GRAD_ACCUM}  chunk=${LOGPROB_CHUNK}  empty_cache_every=${EMPTY_CACHE_STEPS}"
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
        --empty_cache_steps          ${EMPTY_CACHE_STEPS} \
    2>&1 | tee logs/rl_${EXP_NAME}.log

echo ""
echo "=========================================================="
echo " RL DONE at $(date)"
echo "=========================================================="
echo ""
echo " Now running full OOD evaluation on best + final checkpoints..."
echo ""

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