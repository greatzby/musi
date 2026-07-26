#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ============================================================
# Configuration
# ============================================================

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/musique-two-base-factorial-1000}"

# Default checkpoint to evaluate.
# You can override this with checkpoint-500, checkpoint-750, etc.
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint-250}"

# eval_musique_fixed.py searches for:
#   <checkpoint_root>/<run_name>/final
#
# Therefore, this script creates a temporary directory view in
# which each "final" directory points to the selected checkpoint.
VIEW_ROOT="${VIEW_ROOT:-checkpoints/.musique-eval-view-${CHECKPOINT_NAME}}"

OUT_DIR="${OUT_DIR:-eval_results/musique-two-base-factorial-1000-${CHECKPOINT_NAME}-fixed}"

EVAL_DIR="${EVAL_DIR:-prepared_data_2hop}"
EVAL_GPU="${EVAL_GPU:-0}"

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

DTYPE="${DTYPE:-bf16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

# Transformers 4.57.3 may attempt online metadata checks even
# when model files are cached, so online mode is the default.
OFFLINE="${OFFLINE:-0}"

# Expected:
# 2 base models × 2 target styles × 2 prompt styles = 8 runs.
EXPECTED_RUNS="${EXPECTED_RUNS:-8}"

# Remove any old output from an incomplete previous evaluation.
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"

# ============================================================
# Environment
# ============================================================

export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
export HF_HUB_DISABLE_TELEMETRY=1

LOCAL_FLAGS=()

if [[ "${OFFLINE}" == "1" ]]; then
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

    LOCAL_FLAGS+=(--local_files_only)
else
    unset HF_HUB_OFFLINE || true
    unset HF_DATASETS_OFFLINE || true
    unset TRANSFORMERS_OFFLINE || true
fi

# Avoid literal unmatched glob patterns.
shopt -s nullglob

# ============================================================
# Checks
# ============================================================

python -m py_compile eval_musique_fixed.py

if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
    echo "ERROR: checkpoint root does not exist:"
    echo "  ${CHECKPOINT_ROOT}"
    exit 1
fi

if [[ ! -d "${EVAL_DIR}" ]]; then
    echo "ERROR: evaluation data directory does not exist:"
    echo "  ${EVAL_DIR}"
    exit 1
fi

REQUIRED_EVAL_FILES=(
    "${EVAL_DIR}/eval_2hop.jsonl"
    "${EVAL_DIR}/eval_3hop_linear.jsonl"
    "${EVAL_DIR}/eval_4hop_linear.jsonl"
)

for REQUIRED_FILE in "${REQUIRED_EVAL_FILES[@]}"; do
    if [[ ! -f "${REQUIRED_FILE}" ]]; then
        echo "ERROR: required evaluation file is missing:"
        echo "  ${REQUIRED_FILE}"
        exit 1
    fi
done

# ============================================================
# Helper: link one file if it exists
# ============================================================

link_file_if_present() {
    local SOURCE_FILE="$1"
    local DESTINATION_DIR="$2"

    if [[ -f "${SOURCE_FILE}" ]]; then
        ln -sfn \
            "$(readlink -f "${SOURCE_FILE}")" \
            "${DESTINATION_DIR}/$(basename "${SOURCE_FILE}")"
    fi
}

# ============================================================
# Build checkpoint evaluation view
# ============================================================

echo "============================================================"
echo "Preparing MuSiQue checkpoint evaluation"
echo "============================================================"
echo "Checkpoint root    : ${CHECKPOINT_ROOT}"
echo "Checkpoint         : ${CHECKPOINT_NAME}"
echo "Evaluation view    : ${VIEW_ROOT}"
echo "Output directory   : ${OUT_DIR}"
echo "Evaluation data    : ${EVAL_DIR}"
echo "============================================================"

# This removes only the temporary symbolic-link view.
# It does not remove any real model checkpoint.
rm -rf "${VIEW_ROOT}"
mkdir -p "${VIEW_ROOT}"

if [[ "${CLEAN_OUTPUT}" == "1" ]]; then
    rm -rf "${OUT_DIR}"
fi

mkdir -p "${OUT_DIR}"

NUMBER_PREPARED=0

for RUN_DIR in "${CHECKPOINT_ROOT}"/*; do
    if [[ ! -d "${RUN_DIR}" ]]; then
        continue
    fi

    RUN_NAME="$(basename "${RUN_DIR}")"

    # Ignore hidden directories and smoke-test directories.
    if [[ "${RUN_NAME}" == .* || "${RUN_NAME}" == _* ]]; then
        continue
    fi

    SELECTED_CHECKPOINT="${RUN_DIR}/${CHECKPOINT_NAME}"
    FINAL_DIR="${RUN_DIR}/final"

    if [[ ! -d "${SELECTED_CHECKPOINT}" ]]; then
        echo "[skip] Selected checkpoint not found:"
        echo "       ${SELECTED_CHECKPOINT}"
        continue
    fi

    if [[ ! -d "${FINAL_DIR}" ]]; then
        echo "ERROR: final tokenizer directory is missing:"
        echo "  ${FINAL_DIR}"
        exit 1
    fi

    if [[ ! -f "${SELECTED_CHECKPOINT}/adapter_config.json" ]]; then
        echo "ERROR: adapter_config.json is missing:"
        echo "  ${SELECTED_CHECKPOINT}/adapter_config.json"
        exit 1
    fi

    if [[ ! -f "${SELECTED_CHECKPOINT}/adapter_model.safetensors" && ! -f "${SELECTED_CHECKPOINT}/adapter_model.bin" ]]; then
        echo "ERROR: adapter weights are missing:"
        echo "  ${SELECTED_CHECKPOINT}"
        exit 1
    fi

    # eval_musique_fixed.py expects:
    #   VIEW_ROOT/<run_name>/final
    TARGET_RUN_DIR="${VIEW_ROOT}/${RUN_NAME}"
    TARGET_MODEL_DIR="${TARGET_RUN_DIR}/final"

    mkdir -p "${TARGET_MODEL_DIR}"

    # ========================================================
    # Adapter configuration and adapter weights
    # These must come from checkpoint-250.
    # ========================================================

    link_file_if_present \
        "${SELECTED_CHECKPOINT}/adapter_config.json" \
        "${TARGET_MODEL_DIR}"

    link_file_if_present \
        "${SELECTED_CHECKPOINT}/adapter_model.safetensors" \
        "${TARGET_MODEL_DIR}"

    link_file_if_present \
        "${SELECTED_CHECKPOINT}/adapter_model.bin" \
        "${TARGET_MODEL_DIR}"

    # Link possible sharded adapter index files.
    for ADAPTER_INDEX_FILE in "${SELECTED_CHECKPOINT}"/adapter_model*.json; do
        if [[ -f "${ADAPTER_INDEX_FILE}" ]]; then
            link_file_if_present \
                "${ADAPTER_INDEX_FILE}" \
                "${TARGET_MODEL_DIR}"
        fi
    done

    # ========================================================
    # Tokenizer files
    #
    # Trainer checkpoints may not contain the tokenizer. The
    # tokenizer is unchanged during LoRA training, so we use
    # the tokenizer files stored under final/.
    # ========================================================

    TOKENIZER_FILES=(
        "tokenizer.json"
        "tokenizer_config.json"
        "special_tokens_map.json"
        "added_tokens.json"
        "tokenizer.model"
        "vocab.json"
        "merges.txt"
        "chat_template.jinja"
        "generation_config.json"
    )

    for TOKENIZER_FILE_NAME in "${TOKENIZER_FILES[@]}"; do
        link_file_if_present \
            "${FINAL_DIR}/${TOKENIZER_FILE_NAME}" \
            "${TARGET_MODEL_DIR}"
    done

    # ========================================================
    # Experiment metadata
    #
    # The evaluator uses this to identify:
    #   base_key
    #   model_name
    #   target_style
    #   prompt_style
    #   context_mode
    #   seed
    # ========================================================

    EXPERIMENT_CONFIG_SOURCE=""

    if [[ -f "${FINAL_DIR}/experiment_config.json" ]]; then
        EXPERIMENT_CONFIG_SOURCE="${FINAL_DIR}/experiment_config.json"

    elif [[ -f "${RUN_DIR}/experiment_config.json" ]]; then
        EXPERIMENT_CONFIG_SOURCE="${RUN_DIR}/experiment_config.json"

    else
        echo "ERROR: experiment_config.json is missing:"
        echo "  ${RUN_DIR}"
        exit 1
    fi

    link_file_if_present \
        "${EXPERIMENT_CONFIG_SOURCE}" \
        "${TARGET_MODEL_DIR}"

    # ========================================================
    # Verify the temporary model directory
    # ========================================================

    if [[ ! -e "${TARGET_MODEL_DIR}/adapter_config.json" ]]; then
        echo "ERROR: failed to link adapter_config.json:"
        echo "  ${TARGET_MODEL_DIR}"
        exit 1
    fi

    if [[ ! -e "${TARGET_MODEL_DIR}/adapter_model.safetensors" && ! -e "${TARGET_MODEL_DIR}/adapter_model.bin" ]]; then
        echo "ERROR: failed to link adapter weights:"
        echo "  ${TARGET_MODEL_DIR}"
        exit 1
    fi

    if [[ ! -e "${TARGET_MODEL_DIR}/tokenizer_config.json" ]]; then
        echo "ERROR: failed to link tokenizer_config.json:"
        echo "  ${TARGET_MODEL_DIR}"
        exit 1
    fi

    if [[ ! -e "${TARGET_MODEL_DIR}/experiment_config.json" ]]; then
        echo "ERROR: failed to link experiment_config.json:"
        echo "  ${TARGET_MODEL_DIR}"
        exit 1
    fi

    NUMBER_PREPARED=$((NUMBER_PREPARED + 1))

    echo "[prepared] ${RUN_NAME}"
    echo "           adapter   : ${SELECTED_CHECKPOINT}"
    echo "           tokenizer : ${FINAL_DIR}"
done

echo ""
echo "Prepared models: ${NUMBER_PREPARED}"

if [[ "${NUMBER_PREPARED}" -ne "${EXPECTED_RUNS}" ]]; then
    echo ""
    echo "ERROR: expected ${EXPECTED_RUNS} models, but prepared ${NUMBER_PREPARED}."
    echo ""
    echo "Selected checkpoints currently available:"

    find "${CHECKPOINT_ROOT}" \
        -mindepth 2 \
        -maxdepth 2 \
        -type d \
        -name "${CHECKPOINT_NAME}" \
        | sort

    exit 1
fi

# ============================================================
# Show exactly which adapter files will be evaluated
# ============================================================

echo ""
echo "============================================================"
echo "Prepared checkpoint adapter files"
echo "============================================================"

find "${VIEW_ROOT}" \
    -type l \
    \( \
        -name "adapter_config.json" \
        -o -name "adapter_model.safetensors" \
        -o -name "adapter_model.bin" \
    \) \
    -print \
    -exec readlink -f {} \;

# ============================================================
# Run evaluation
# ============================================================

echo ""
echo "============================================================"
echo "MuSiQue ${CHECKPOINT_NAME} evaluation"
echo "============================================================"
echo "Models             : ${NUMBER_PREPARED}"
echo "Checkpoint         : ${CHECKPOINT_NAME}"
echo "Evaluation data    : ${EVAL_DIR}"
echo "Output directory   : ${OUT_DIR}"
echo "GPU                : ${EVAL_GPU}"
echo "Batch size         : ${BATCH_SIZE}"
echo "Maximum input      : ${MAX_INPUT_LENGTH}"
echo "Maximum new tokens : ${MAX_NEW_TOKENS}"
echo "Offline            : ${OFFLINE}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
python eval_musique_fixed.py \
    --mode generate \
    --checkpoint_root "${VIEW_ROOT}" \
    --eval_dir "${EVAL_DIR}" \
    --out_dir "${OUT_DIR}" \
    --splits \
        2hop \
        3hop_linear \
        4hop_linear \
    --batch_size "${BATCH_SIZE}" \
    --max_input_length "${MAX_INPUT_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --device cuda:0 \
    --dtype "${DTYPE}" \
    --attn_implementation "${ATTN_IMPL}" \
    "${LOCAL_FLAGS[@]}"

echo ""
echo "============================================================"
echo "Checkpoint evaluation complete"
echo "============================================================"
echo "Checkpoint evaluated : ${CHECKPOINT_NAME}"
echo "Results              : ${OUT_DIR}"
echo "Temporary view       : ${VIEW_ROOT}"
echo "============================================================"