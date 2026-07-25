#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# User configuration
# ============================================================

MODEL="unsloth/Llama-3.2-3B"

RAW_DATA_DIR="./data"
DATA_DIR="prepared_data_2hop"

ANSWER_DIR="checkpoints/llama32-3b-musique-answer-only-gold"
BRIDGE_DIR="checkpoints/llama32-3b-musique-bridge-aware-gold"

EVAL_OUT_DIR="eval_results/llama32-main"

GPU_LIST="${GPU_LIST:-0,1}"
NUM_GPUS="${NUM_GPUS:-2}"

# If SDPA still causes a native crash, run:
#   ATTN_IMPL=eager bash run_llama_experiments.sh
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

DTYPE="${DTYPE:-bf16}"

# Two GPUs:
#   4 examples/GPU × 4 accumulation × 2 GPUs = global batch 32
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

MAX_STEPS="${MAX_STEPS:-200}"
MAX_LENGTH="${MAX_LENGTH:-2048}"

# Set RUN_SMOKE=0 to skip the two-step smoke test.
RUN_SMOKE="${RUN_SMOKE:-1}"

# ============================================================
# Stable cluster environment
# ============================================================

export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "============================================================"
echo "Configuration"
echo "============================================================"
echo "GPUs                : ${GPU_LIST}"
echo "Number of GPUs      : ${NUM_GPUS}"
echo "Attention           : ${ATTN_IMPL}"
echo "Dtype               : ${DTYPE}"
echo "Per-device batch    : ${PER_DEVICE_BATCH}"
echo "Gradient accumulation: ${GRAD_ACCUM}"
echo "Global batch        : $((PER_DEVICE_BATCH * GRAD_ACCUM * NUM_GPUS))"
echo "Max steps           : ${MAX_STEPS}"
echo "============================================================"

# ============================================================
# Helper: launch two-GPU DDP
# ============================================================

run_ddp() {
    CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NUM_GPUS}" \
        "$@"
}

# ============================================================
# 1. Prepare data
# ============================================================

echo ""
echo "============================================================"
echo "1. Preparing MuSiQue"
echo "============================================================"

python prepare_musique_2hop.py \
    --musique_dir "${RAW_DATA_DIR}" \
    --output_dir "${DATA_DIR}" \
    --seed 42

# ============================================================
# 2. Two-GPU smoke test
# ============================================================

if [[ "${RUN_SMOKE}" == "1" ]]; then
    echo ""
    echo "============================================================"
    echo "2. Running two-GPU DDP smoke test"
    echo "============================================================"

    SMOKE_DIR="checkpoints/llama32-ddp-smoke"
    rm -rf "${SMOKE_DIR}"

    run_ddp sft_llama32_musique.py \
        --model_name "${MODEL}" \
        --train_file "${DATA_DIR}/train_2hop.jsonl" \
        --output_dir "${SMOKE_DIR}" \
        --target_style answer_only \
        --context_mode gold \
        --prompt_style anchored \
        --max_length "${MAX_LENGTH}" \
        --max_steps 2 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --learning_rate 2e-4 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --dtype "${DTYPE}" \
        --attn_implementation "${ATTN_IMPL}" \
        --gradient_checkpointing \
        --num_proc 0 \
        --dataloader_num_workers 0 \
        --no-dataloader_pin_memory \
        --optim adamw_torch \
        --seed 42 \
        2>&1 | tee llama32_ddp_smoke.log

    if [[ ! -d "${SMOKE_DIR}/final" ]]; then
        echo "ERROR: smoke test did not produce ${SMOKE_DIR}/final"
        exit 1
    fi

    echo "Smoke test succeeded."
    rm -rf "${SMOKE_DIR}"
else
    echo ""
    echo "Skipping smoke test because RUN_SMOKE=${RUN_SMOKE}"
fi

# ============================================================
# 3. Train answer-only
# ============================================================

echo ""
echo "============================================================"
echo "3. Training answer-only SFT"
echo "============================================================"

if [[ -d "${ANSWER_DIR}/final" ]]; then
    echo "Answer-only final model already exists:"
    echo "  ${ANSWER_DIR}/final"
    echo "Skipping answer-only training."
else
    run_ddp sft_llama32_musique.py \
        --model_name "${MODEL}" \
        --train_file "${DATA_DIR}/train_2hop.jsonl" \
        --output_dir "${ANSWER_DIR}" \
        --target_style answer_only \
        --context_mode gold \
        --prompt_style anchored \
        --max_length "${MAX_LENGTH}" \
        --max_steps "${MAX_STEPS}" \
        --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --learning_rate 2e-4 \
        --warmup_ratio 0.03 \
        --weight_decay 0.0 \
        --max_grad_norm 1.0 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --dtype "${DTYPE}" \
        --attn_implementation "${ATTN_IMPL}" \
        --gradient_checkpointing \
        --num_proc 0 \
        --dataloader_num_workers 0 \
        --no-dataloader_pin_memory \
        --optim adamw_torch \
        --ddp_timeout 1800 \
        --ddp_bucket_cap_mb 25 \
        --seed 42 \
        2>&1 | tee llama32_answer_only_train.log
fi

# ============================================================
# 4. Train bridge-aware
# ============================================================

echo ""
echo "============================================================"
echo "4. Training bridge-aware SFT"
echo "============================================================"

if [[ -d "${BRIDGE_DIR}/final" ]]; then
    echo "Bridge-aware final model already exists:"
    echo "  ${BRIDGE_DIR}/final"
    echo "Skipping bridge-aware training."
else
    run_ddp sft_llama32_musique.py \
        --model_name "${MODEL}" \
        --train_file "${DATA_DIR}/train_2hop.jsonl" \
        --output_dir "${BRIDGE_DIR}" \
        --target_style bridge_aware \
        --context_mode gold \
        --prompt_style anchored \
        --max_length "${MAX_LENGTH}" \
        --max_steps "${MAX_STEPS}" \
        --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --learning_rate 2e-4 \
        --warmup_ratio 0.03 \
        --weight_decay 0.0 \
        --max_grad_norm 1.0 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --dtype "${DTYPE}" \
        --attn_implementation "${ATTN_IMPL}" \
        --gradient_checkpointing \
        --num_proc 0 \
        --dataloader_num_workers 0 \
        --no-dataloader_pin_memory \
        --optim adamw_torch \
        --ddp_timeout 1800 \
        --ddp_bucket_cap_mb 25 \
        --seed 42 \
        2>&1 | tee llama32_bridge_aware_train.log
fi

# ============================================================
# 5. Check outputs
# ============================================================

echo ""
echo "============================================================"
echo "5. Checking trained models"
echo "============================================================"

if [[ ! -d "${ANSWER_DIR}/final" ]]; then
    echo "ERROR: answer-only model was not found."
    exit 1
fi

if [[ ! -d "${BRIDGE_DIR}/final" ]]; then
    echo "ERROR: bridge-aware model was not found."
    exit 1
fi

echo "Answer-only:"
find "${ANSWER_DIR}/final" -maxdepth 1 -type f -printf "  %f\n" | sort

echo "Bridge-aware:"
find "${BRIDGE_DIR}/final" -maxdepth 1 -type f -printf "  %f\n" | sort

# ============================================================
# 6. Main evaluation
#
# Evaluation is performed sequentially on GPU 0. The evaluation
# script loads LoRA adapters and their base model.
# ============================================================

echo ""
echo "============================================================"
echo "6. Main evaluation without oracle bridge count"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 \
python eval_llama32_musique.py \
    --model_dirs \
        "${ANSWER_DIR}/final" \
        "${BRIDGE_DIR}/final" \
    --eval_dir "${DATA_DIR}" \
    --out_dir "${EVAL_OUT_DIR}" \
    --splits \
        2hop \
        3hop_linear \
        4hop_linear \
    --target_style auto \
    --context_mode auto \
    --prompt_variant standard \
    --batch_size 8 \
    --max_input_length 4096 \
    --max_new_tokens 128 \
    --device cuda:0 \
    --dtype "${DTYPE}" \
    --attn_implementation "${ATTN_IMPL}" \
    2>&1 | tee llama32_main_eval.log

echo ""
echo "============================================================"
echo "All Llama-3.2-3B experiments completed"
echo "============================================================"
echo "Answer-only model : ${ANSWER_DIR}/final"
echo "Bridge-aware model: ${BRIDGE_DIR}/final"
echo "Evaluation results: ${EVAL_OUT_DIR}"
echo "============================================================"