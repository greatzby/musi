#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ============================================================
# Configuration
# ============================================================

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/musique-two-base-factorial-1000}"

# Default: evaluate checkpoint-250.
# The same script can later evaluate checkpoint-500/750/1000 by
# overriding CHECKPOINT_NAME.
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint-250}"

# eval_musique_fixed.py discovers */final. We therefore create
# a temporary evaluation view whose "final" adapter files point
# to the selected checkpoint.
VIEW_ROOT="${VIEW_ROOT:-checkpoints/.musique-eval-view-${CHECKPOINT_NAME}}"

OUT_DIR="${OUT_DIR:-eval_results/musique-two-base-factorial-1000-${CHECKPOINT_NAME}-fixed}"

EVAL_DIR="${EVAL_DIR:-prepared_data_2hop}"
EVAL_GPU="${EVAL_GPU:-0}"

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

DTYPE="${DTYPE:-bf16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

# Keep online mode because Transformers 4.57.3 may attempt a Hub
# metadata lookup even when the model files are cached.
OFFLINE="${OFFLINE:-0}"

# Expected number of experiments:
# 2 bases × 2 target styles × 2 prompt styles = 8.
EXPECTED_RUNS="${EXPECTED_RUNS:-8}"

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

python -m py_compile eval_musique_fixed.py

if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
    echo "ERROR: checkpoint root does not exist:"
    echo "  ${CHECKPOINT_ROOT}"
    exit 1
fi

# ============================================================
# Helper: create an absolute symbolic link
# ============================================================

link_file_if_present() {
    local source_file="$1"
    local destination_dir="$2"

    if [[ -f "${source_file}" ]]; then
        ln -s \
            "$(readlink -f "${source_file}")" \
            "${destination_dir}/$(basename "${source_file}")"
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
echo "============================================================"

# Remove only the temporary evaluation view, never the real
# training checkpoint directory.
rm -rf "${VIEW_ROOT}"
mkdir -p "${VIEW_ROOT}"

NUMBER_PREPARED=0

for RUN_DIR in "${CHECKPOINT_ROOT}"/*; do
    if [[ ! -d "${RUN_DIR}" ]]; then
        continue
    fi

    RUN_NAME="$(basename "${RUN_DIR}")"

    # Ignore smoke-test and hidden directories.
    if [[ "${RUN_NAME}" == _* || "${RUN_NAME}" == .* ]]; then
        continue
    fi

    SELECTED_CHECKPOINT="${RUN_DIR}/${CHECKPOINT_NAME}"
    FINAL_DIR="${RUN_DIR}/final"

    # Only include runs that have the selected checkpoint.
    if [[ ! -d "${SELECTED_CHECKPOINT}" ]]; then
        echo "[skip] No ${CHECKPOINT_NAME}: ${RUN_NAME}"
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

    if (
        [[ ! -f "${SELECTED_CHECKPOINT}/adapter_model.safetensors" ]]
        && [[ ! -f "${SELECTED_CHECKPOINT}/adapter_model.bin" ]]
    ); then
        echo "ERROR: adapter weights are missing:"
        echo "  ${SELECTED_CHECKPOINT}"
        exit 1
    fi

    # The fixed evaluator expects CHECKPOINT_ROOT/<run>/final.
    TARGET_RUN_DIR="${VIEW_ROOT}/${RUN_NAME}"
    TARGET_MODEL_DIR="${TARGET_RUN_DIR}/final"

    mkdir -p "${TARGET_MODEL_DIR}"

    # --------------------------------------------------------
    # Adapter files: always come from checkpoint-250.
    # --------------------------------------------------------

    link_file_if_present \
        "${SELECTED_CHECKPOINT}/adapter_config.json" \
        "${TARGET_MODEL_DIR}"

    link_file_if_present \
        "${SELECTED_CHECKPOINT}/adapter_model.safetensors" \
        "${TARGET_MODEL_DIR}"

    link_file_if_present \
        "${SELECTED_CHECKPOINT}/adapter_model.bin" \
        "${TARGET_MODEL_DIR}"

    # Include any additional PEFT adapter index files.
    for FILE in "${SELECTED_CHECKPOINT}"/adapter_model*.json; do
        if [[ -f "${FILE}" ]]; then
            link_file_if_present \
                "${FILE}" \
                "${TARGET_MODEL_DIR}"
        fi
    done

    # --------------------------------------------------------
    # Tokenizer files: come from final because Trainer
    # checkpoints may not contain tokenizer files.
    #
    # These files do not contain trained model weights.
    # --------------------------------------------------------

    for FILE in \
        "${FINAL_DIR}"/tokenizer.json \
        "${FINAL_DIR}"/tokenizer_config.json \
        "${FINAL_DIR}"/special_tokens_map.json \
        "${FINAL_DIR}"/added_tokens.json \
        "${FINAL_DIR}"/tokenizer.model \
        "${FINAL_DIR}"/vocab.json \
        "${FINAL_DIR}"/merges.txt \
        "${FINAL_DIR}"/chat_template.jinja
    do
        if [[ -f "${FILE}" ]]; then
            link_file_if_present \
                "${FILE}" \
                "${TARGET_MODEL_DIR}"
        fi
    done

    # --------------------------------------------------------
    # Experiment metadata: used to recover target_style,
    # prompt_style, model_name and seed.
    # --------------------------------------------------------

    EXPERIMENT_CONFIG=""

    if [[ -f "${FINAL_DIR}/experiment_config.json" ]]; then
        EXPERIMENT_CONFIG="${FINAL_DIR}/experiment_config.json"

    elif [[ -f "${RUN_DIR}/experiment_config.json" ]]; then
        EXPERIMENT_CONFIG="${RUN_DIR}/experiment_config.json"

    else
        echo "ERROR: experiment_config.json is missing:"
        echo "  ${RUN_DIR}"
        exit 1
    fi

    link_file_if_present \
        "${EXPERIMENT_CONFIG}" \
        "${TARGET_MODEL_DIR}"

    # --------------------------------------------------------
    # Final verification of the temporary model directory.
    # --------------------------------------------------------

    if [[ ! -e "${TARGET_MODEL_DIR}/adapter_config.json" ]]; then
        echo "ERROR: failed to link adapter configuration:"
        echo "  ${TARGET_MODEL_DIR}"
        exit 1
    fi

    if (
        [[ ! -e "${TARGET_MODEL_DIR}/adapter_model.safetensors" ]]
        && [[ ! -e "${TARGET_MODEL_DIR}/adapter_model.bin" ]]
    ); then
        echo "ERROR: failed to link adapter weights:"
        echo "  ${TARGET_MODEL_DIR}"
        exit 1
    fi

    if [[ ! -e "${TARGET_MODEL_DIR}/tokenizer_config.json" ]]; then
        echo "ERROR: failed to link tokenizer configuration:"
        echo "  ${TARGET_MODEL_DIR}"
        exit 1
    fi

    NUMBER_PREPARED=$((NUMBER_PREPARED + 1))

    echo "[prepared] ${RUN_NAME}"
    echo "           adapter: ${SELECTED_CHECKPOINT}"
    echo "           tokenizer: ${FINAL_DIR}"
done

echo ""
echo "Prepared models: ${NUMBER_PREPARED}"

if [[ "${NUMBER_PREPARED}" -ne "${EXPECTED_RUNS}" ]]; then
    echo "ERROR: expected ${EXPECTED_RUNS} models but prepared ${NUMBER_PREPARED}."
    echo ""
    echo "Available selected checkpoints:"

    find "${CHECKPOINT_ROOT}" \
        -mindepth 2 \
        -maxdepth 2 \
        -type d \
        -name "${CHECKPOINT_NAME}" \
        | sort

    exit 1
fi

# ============================================================
# Evaluation
# ============================================================

echo ""
echo "============================================================"
echo "MuSiQue ${CHECKPOINT_NAME} evaluation"
echo "============================================================"
echo "Models             : ${NUMBER_PREPARED}"
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
echo "Checkpoint evaluated: ${CHECKPOINT_NAME}"
echo "Results             : ${OUT_DIR}"
echo "Temporary view      : ${VIEW_ROOT}"
echo "============================================================"