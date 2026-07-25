#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

cd "${SCRIPT_DIR}"

# ============================================================
# User configuration
# ============================================================

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-3B}"
LLAMA_MODEL="${LLAMA_MODEL:-unsloth/Llama-3.2-3B}"

DATA_DIR="${DATA_DIR:-prepared_data_2hop}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train_2hop.jsonl}"

ROOT_DIR="${ROOT_DIR:-checkpoints/musique-sft-factorial-v1}"
LOG_DIR="${LOG_DIR:-logs/musique-sft-factorial-v1}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-eval_results/musique-sft-factorial-v1}"

GPU_LIST="${GPU_LIST:-0,1}"
NUM_GPUS="${NUM_GPUS:-2}"
EVAL_GPU="${EVAL_GPU:-0}"

ATTN_IMPL="${ATTN_IMPL:-sdpa}"
DTYPE="${DTYPE:-bf16}"

PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_STEPS="${MAX_STEPS:-2000}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"

SAVE_STEPS="${SAVE_STEPS:-250}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_MAX_INPUT="${EVAL_MAX_INPUT:-4096}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# Default: one preliminary seed.
# Final paper experiment:
#   SEEDS="42 43 44"
SEEDS_STRING="${SEEDS:-42}"

# Default full factorial:
#   2 targets × 2 prompts × 2 bases = 8 models.
TARGET_STYLES_STRING="${
    TARGET_STYLES:-answer_only bridge_aware
}"

PROMPT_STYLES_STRING="${
    PROMPT_STYLES:-canonical anchored
}"

# Smoke test both base models before launching all runs.
RUN_SMOKE="${RUN_SMOKE:-1}"

# Run full evaluation after training.
RUN_EVAL="${RUN_EVAL:-1}"

# Set OFFLINE=1 if both models are already cached.
OFFLINE="${OFFLINE:-0}"

mkdir -p \
    "${ROOT_DIR}" \
    "${LOG_DIR}" \
    "${EVAL_OUT_DIR}"

# ============================================================
# Environment
# ============================================================

export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HF_HUB_DISABLE_TELEMETRY=1

LOCAL_FLAGS=()

if [[ "${OFFLINE}" == "1" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    LOCAL_FLAGS+=(--local_files_only)
fi

read -r -a SEED_ARRAY \
    <<< "${SEEDS_STRING}"

read -r -a TARGET_STYLE_ARRAY \
    <<< "${TARGET_STYLES_STRING}"

read -r -a PROMPT_STYLE_ARRAY \
    <<< "${PROMPT_STYLES_STRING}"

MODEL_KEYS=(
    "qwen25-3b"
    "llama32-3b"
)

MODEL_NAMES=(
    "${QWEN_MODEL}"
    "${LLAMA_MODEL}"
)

# ============================================================
# Preflight
# ============================================================

echo "============================================================"
echo "MuSiQue SFT factorial experiment"
echo "============================================================"
echo "Qwen model          : ${QWEN_MODEL}"
echo "Llama model         : ${LLAMA_MODEL}"
echo "Data directory      : ${DATA_DIR}"
echo "Training file       : ${TRAIN_FILE}"
echo "GPU list            : ${GPU_LIST}"
echo "Number of GPUs      : ${NUM_GPUS}"
echo "Global batch        : $((PER_DEVICE_BATCH * GRAD_ACCUM * NUM_GPUS))"
echo "Maximum steps       : ${MAX_STEPS}"
echo "Learning rate       : ${LEARNING_RATE}"
echo "Warmup steps        : ${WARMUP_STEPS}"
echo "Seeds               : ${SEED_ARRAY[*]}"
echo "Target styles       : ${TARGET_STYLE_ARRAY[*]}"
echo "Prompt styles       : ${PROMPT_STYLE_ARRAY[*]}"
echo "Offline             : ${OFFLINE}"
echo "Checkpoint root     : ${ROOT_DIR}"
echo "Evaluation output   : ${EVAL_OUT_DIR}"
echo "============================================================"

python - <<'PY'
import torch
import transformers
import peft
import datasets

print("Python package versions")
print("  torch       :", torch.__version__)
print("  transformers:", transformers.__version__)
print("  peft        :", peft.__version__)
print("  datasets    :", datasets.__version__)
print("  CUDA        :", torch.cuda.is_available())
print("  GPU count   :", torch.cuda.device_count())
PY

REQUIRED_FILES=(
    "${TRAIN_FILE}"
    "${DATA_DIR}/eval_2hop.jsonl"
    "${DATA_DIR}/eval_3hop_linear.jsonl"
    "${DATA_DIR}/eval_4hop_linear.jsonl"
)

for required_file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: required file not found:"
        echo "  ${required_file}"
        exit 1
    fi
done

# ============================================================
# Helper: one DDP smoke test
# ============================================================

smoke_test_one() {
    local base_key="$1"
    local model_name="$2"
    local smoke_dir="${ROOT_DIR}/_smoke_${base_key}"
    local smoke_log="${LOG_DIR}/smoke_${base_key}.log"

    rm -rf "${smoke_dir}"

    echo ""
    echo "============================================================"
    echo "Smoke test: ${base_key}"
    echo "Model     : ${model_name}"
    echo "============================================================"

    if CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
        torchrun \
            --standalone \
            --nnodes=1 \
            --nproc_per_node="${NUM_GPUS}" \
            sft_musique_factorial.py \
            --model_name "${model_name}" \
            --base_key "${base_key}" \
            --train_file "${TRAIN_FILE}" \
            --output_dir "${smoke_dir}" \
            --target_style answer_only \
            --prompt_style canonical \
            --context_mode gold \
            --max_length "${MAX_LENGTH}" \
            --max_steps 2 \
            --per_device_train_batch_size 2 \
            --gradient_accumulation_steps 1 \
            --learning_rate "${LEARNING_RATE}" \
            --warmup_steps 0 \
            --logging_steps 1 \
            --save_steps 100 \
            --save_total_limit 1 \
            --lora_r "${LORA_R}" \
            --lora_alpha "${LORA_ALPHA}" \
            --lora_dropout "${LORA_DROPOUT}" \
            --dtype "${DTYPE}" \
            --attn_implementation "${ATTN_IMPL}" \
            --num_proc 0 \
            --dataloader_num_workers 0 \
            --resume_from_checkpoint none \
            --seed 42 \
            "${LOCAL_FLAGS[@]}" \
            2>&1 | tee "${smoke_log}"
    then
        if [[ ! -d "${smoke_dir}/final" ]]; then
            echo "ERROR: smoke test produced no final model."
            exit 1
        fi
    else
        echo "ERROR: smoke test failed for ${base_key}."
        exit 1
    fi

    rm -rf "${smoke_dir}"

    echo "Smoke test succeeded: ${base_key}"
}

# ============================================================
# Helper: train one full experiment
# ============================================================

FAILED_RUNS=()

train_one() {
    local base_key="$1"
    local model_name="$2"
    local target_style="$3"
    local prompt_style="$4"
    local seed="$5"

    local run_name
    run_name="${
        base_key
    }__${
        target_style
    }__${
        prompt_style
    }__seed${
        seed
    }"

    local output_dir="${ROOT_DIR}/${run_name}"
    local log_file="${LOG_DIR}/${run_name}.log"

    echo ""
    echo "============================================================"
    echo "Training run"
    echo "============================================================"
    echo "Run          : ${run_name}"
    echo "Base         : ${base_key}"
    echo "Model        : ${model_name}"
    echo "Target       : ${target_style}"
    echo "Prompt       : ${prompt_style}"
    echo "Seed         : ${seed}"
    echo "Output       : ${output_dir}"
    echo "Log          : ${log_file}"
    echo "============================================================"

    if [[ -d "${output_dir}/final" ]]; then
        echo "Completed final model exists. Skipping."
        return 0
    fi

    if CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
        torchrun \
            --standalone \
            --nnodes=1 \
            --nproc_per_node="${NUM_GPUS}" \
            sft_musique_factorial.py \
            --model_name "${model_name}" \
            --base_key "${base_key}" \
            --train_file "${TRAIN_FILE}" \
            --output_dir "${output_dir}" \
            --target_style "${target_style}" \
            --prompt_style "${prompt_style}" \
            --context_mode gold \
            --max_length "${MAX_LENGTH}" \
            --max_steps "${MAX_STEPS}" \
            --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
            --gradient_accumulation_steps "${GRAD_ACCUM}" \
            --learning_rate "${LEARNING_RATE}" \
            --warmup_steps "${WARMUP_STEPS}" \
            --weight_decay 0.0 \
            --max_grad_norm 1.0 \
            --lr_scheduler_type cosine \
            --logging_steps 10 \
            --save_steps "${SAVE_STEPS}" \
            --save_total_limit "${SAVE_TOTAL_LIMIT}" \
            --lora_r "${LORA_R}" \
            --lora_alpha "${LORA_ALPHA}" \
            --lora_dropout "${LORA_DROPOUT}" \
            --dtype "${DTYPE}" \
            --attn_implementation "${ATTN_IMPL}" \
            --gradient_checkpointing \
            --num_proc 0 \
            --dataloader_num_workers 0 \
            --resume_from_checkpoint auto \
            --ddp_timeout 1800 \
            --ddp_bucket_cap_mb 25 \
            --seed "${seed}" \
            "${LOCAL_FLAGS[@]}" \
            2>&1 | tee "${log_file}"
    then
        if [[ -d "${output_dir}/final" ]]; then
            echo "Training completed: ${run_name}"
            return 0
        fi

        echo "Training command returned success but no final model exists."
        FAILED_RUNS+=("${run_name}")
        return 1
    else
        echo "Training failed: ${run_name}"
        FAILED_RUNS+=("${run_name}")
        return 1
    fi
}

# ============================================================
# 1. Smoke tests
# ============================================================

if [[ "${RUN_SMOKE}" == "1" ]]; then
    for index in "${!MODEL_KEYS[@]}"; do
        smoke_test_one \
            "${MODEL_KEYS[$index]}" \
            "${MODEL_NAMES[$index]}"
    done
else
    echo "Skipping smoke tests."
fi

# ============================================================
# 2. Train complete factorial
# ============================================================

for seed in "${SEED_ARRAY[@]}"; do
    for index in "${!MODEL_KEYS[@]}"; do
        base_key="${MODEL_KEYS[$index]}"
        model_name="${MODEL_NAMES[$index]}"

        for target_style in "${TARGET_STYLE_ARRAY[@]}"; do
            for prompt_style in "${PROMPT_STYLE_ARRAY[@]}"; do
                train_one \
                    "${base_key}" \
                    "${model_name}" \
                    "${target_style}" \
                    "${prompt_style}" \
                    "${seed}" || true
            done
        done
    done
done

# ============================================================
# 3. Collect completed models
# ============================================================

MODEL_DIRS=()

for seed in "${SEED_ARRAY[@]}"; do
    for base_key in "${MODEL_KEYS[@]}"; do
        for target_style in "${TARGET_STYLE_ARRAY[@]}"; do
            for prompt_style in "${PROMPT_STYLE_ARRAY[@]}"; do
                run_name="${
                    base_key
                }__${
                    target_style
                }__${
                    prompt_style
                }__seed${
                    seed
                }"

                final_dir="${ROOT_DIR}/${run_name}/final"

                if [[ -d "${final_dir}" ]]; then
                    MODEL_DIRS+=("${final_dir}")
                fi
            done
        done
    done
done

echo ""
echo "============================================================"
echo "Completed models"
echo "============================================================"

for model_dir in "${MODEL_DIRS[@]}"; do
    echo "  ${model_dir}"
done

echo "Number completed: ${#MODEL_DIRS[@]}"

# ============================================================
# 4. Full evaluation
# ============================================================

EVAL_FAILED=0

if [[ "${RUN_EVAL}" == "1" ]]; then
    if [[ "${#MODEL_DIRS[@]}" -eq 0 ]]; then
        echo "ERROR: no completed models available for evaluation."
        EVAL_FAILED=1
    else
        echo ""
        echo "============================================================"
        echo "Running full evaluation"
        echo "============================================================"

        if CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
            python eval_musique_factorial.py \
                --model_dirs "${MODEL_DIRS[@]}" \
                --eval_dir "${DATA_DIR}" \
                --out_dir "${EVAL_OUT_DIR}" \
                --splits \
                    2hop \
                    3hop_linear \
                    4hop_linear \
                --batch_size "${EVAL_BATCH_SIZE}" \
                --max_input_length "${EVAL_MAX_INPUT}" \
                --max_new_tokens "${EVAL_MAX_NEW_TOKENS}" \
                --device cuda:0 \
                --dtype "${DTYPE}" \
                --attn_implementation "${ATTN_IMPL}" \
                --resume \
                "${LOCAL_FLAGS[@]}" \
                2>&1 | tee "${LOG_DIR}/full_evaluation.log"
        then
            echo "Evaluation completed."
        else
            echo "One or more evaluation runs failed."
            EVAL_FAILED=1
        fi
    fi
else
    echo "Skipping evaluation because RUN_EVAL=${RUN_EVAL}"
fi

# ============================================================
# 5. Final status
# ============================================================

echo ""
echo "============================================================"
echo "Experiment finished"
echo "============================================================"
echo "Checkpoint root : ${ROOT_DIR}"
echo "Logs            : ${LOG_DIR}"
echo "Evaluation      : ${EVAL_OUT_DIR}"
echo "Completed models: ${#MODEL_DIRS[@]}"
echo "Training failures: ${#FAILED_RUNS[@]}"
echo "Evaluation failed: ${EVAL_FAILED}"

if [[ "${#FAILED_RUNS[@]}" -gt 0 ]]; then
    echo ""
    echo "Failed training runs:"

    for failed_run in "${FAILED_RUNS[@]}"; do
        echo "  ${failed_run}"
    done
fi

if (
    [[ "${#FAILED_RUNS[@]}" -eq 0 ]]
    && [[ "${EVAL_FAILED}" -eq 0 ]]
); then
    date -Is > "${LOG_DIR}/ALL_DONE.txt"
    echo "Status: ALL DONE"
    exit 0
fi

date -Is > "${LOG_DIR}/FINISHED_WITH_FAILURES.txt"
echo "Status: FINISHED WITH FAILURES"
exit 1