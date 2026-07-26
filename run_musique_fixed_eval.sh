#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ============================================================
# Configuration
# ============================================================

MODE="${MODE:-generate}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/musique-two-base-factorial-1000}"
SOURCE_RESULTS_DIR="${SOURCE_RESULTS_DIR:-eval_results/musique-two-base-factorial-1000}"

if [[ "${MODE}" == "reparse" ]]; then
    OUT_DIR="${OUT_DIR:-eval_results/musique-two-base-factorial-1000-reparsed}"
else
    OUT_DIR="${OUT_DIR:-eval_results/musique-two-base-factorial-1000-regenerated-fixed}"
fi

EVAL_DIR="${EVAL_DIR:-prepared_data_2hop}"
EVAL_GPU="${EVAL_GPU:-0}"

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

DTYPE="${DTYPE:-bf16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

# Use normal cached/online loading.
OFFLINE="${OFFLINE:-0}"

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

# ============================================================
# Checks
# ============================================================

python -m py_compile \
    eval_musique_fixed.py

if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
    echo "ERROR: checkpoint root not found:"
    echo "  ${CHECKPOINT_ROOT}"
    exit 1
fi

if [[ "${MODE}" == "reparse" && ! -d "${SOURCE_RESULTS_DIR}" ]]; then
    echo "ERROR: source results directory not found:"
    echo "  ${SOURCE_RESULTS_DIR}"
    exit 1
fi

echo "============================================================"
echo "MuSiQue corrected evaluation"
echo "============================================================"
echo "Mode               : ${MODE}"
echo "Checkpoint root    : ${CHECKPOINT_ROOT}"
echo "Source results     : ${SOURCE_RESULTS_DIR}"
echo "Output directory   : ${OUT_DIR}"
echo "Evaluation data    : ${EVAL_DIR}"
echo "GPU                : ${EVAL_GPU}"
echo "Batch size         : ${BATCH_SIZE}"
echo "Maximum input      : ${MAX_INPUT_LENGTH}"
echo "Maximum new tokens : ${MAX_NEW_TOKENS}"
echo "Offline            : ${OFFLINE}"
echo "============================================================"

# ============================================================
# Run
# ============================================================

CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
python eval_musique_fixed.py \
    --mode "${MODE}" \
    --checkpoint_root "${CHECKPOINT_ROOT}" \
    --source_results_dir "${SOURCE_RESULTS_DIR}" \
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