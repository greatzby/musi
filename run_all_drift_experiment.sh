#!/bin/bash
# ============================================================
# DRIFT EXPERIMENT — END-TO-END PIPELINE
#   Step 1: SFT with fine-grained checkpointing (50/100/150/200)
#   Step 2: Eval all SFT checkpoints
#   Step 3: RL no-anchor (kl=0) from SFT-200, save every 200 steps
#   Step 4: Eval all RL checkpoints
#   Step 5: Drift analysis
#
# 总耗时预估: SFT ~1h + Eval ~20min + RL ~10h + Eval ~40min + 分析 ~1min
# ============================================================
set -e
set -u
set -o pipefail

# ============================================================
# 配置
# ============================================================
SFT_DIR="checkpoints/qwen2.5-3b-2hop-sft-fine"
RL_DIR="checkpoints/qwen2.5-3b-grpo-noanchor-from-200"
SFT_INIT_CKPT="${SFT_DIR}/checkpoint-200"

EVAL_SFT_DIR="eval_results_sft_fine"
EVAL_RL_DIR="eval_results_rl_noanchor_from_200"

LOG_DIR="logs_drift_experiment"
mkdir -p "${LOG_DIR}"

NPROC=2   # GPU 数,按你的机器改

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ============================================================
# 工具函数
# ============================================================
banner() {
    echo ""
    echo "############################################################"
    echo "# $1"
    echo "# $(date '+%Y-%m-%d %H:%M:%S')"
    echo "############################################################"
}

ckpt_exists() {
    # 至少一个 checkpoint-* 目录存在 -> 视为已训练完
    local d="$1"
    local n
    n=$(ls -d "${d}"/checkpoint-* 2>/dev/null | wc -l || true)
    [[ "${n}" -gt 0 ]]
}

eval_done() {
    # eval 输出目录里至少有一个 summary.json 就算 eval 过
    local d="$1"
    [[ -f "${d}/all_checkpoints_summary.json" ]] || \
    [[ -n "$(find "${d}" -name 'summary.json' 2>/dev/null | head -n1)" ]]
}

# ============================================================
# 前置检查
# ============================================================
banner "Pre-flight checks"

if [[ ! -f "prepared_data_2hop/train_2hop.jsonl" ]]; then
    echo "[ERROR] missing prepared_data_2hop/train_2hop.jsonl"
    exit 1
fi
if [[ ! -f "prepared_data_2hop/eval_2hop.jsonl" ]]; then
    echo "[ERROR] missing prepared_data_2hop/eval_2hop.jsonl"
    exit 1
fi
if [[ ! -f "sft_train_2hop.py" ]]; then
    echo "[ERROR] missing sft_train_2hop.py"
    exit 1
fi
if [[ ! -f "rl_grpo_2hop.py" ]]; then
    echo "[ERROR] missing rl_grpo_2hop.py"
    exit 1
fi
if [[ ! -f "eval_compositional.py" ]]; then
    echo "[ERROR] missing eval_compositional.py"
    exit 1
fi
if [[ ! -f "analyze_drift.py" ]]; then
    echo "[ERROR] missing analyze_drift.py"
    exit 1
fi

nvidia-smi || { echo "[ERROR] nvidia-smi failed"; exit 1; }
echo "[OK] all pre-flight checks passed"

# ============================================================
# Step 1: SFT (fine-grained checkpointing)
# ============================================================
banner "STEP 1/5  SFT fine-grained (50/100/150/200 steps)"

if ckpt_exists "${SFT_DIR}"; then
    echo "[skip] SFT checkpoints already exist in ${SFT_DIR}"
    ls -d "${SFT_DIR}"/checkpoint-* 2>/dev/null || true
else
    echo "[run] launching SFT..."
    torchrun --nproc_per_node=${NPROC} sft_train_2hop.py \
        2>&1 | tee "${LOG_DIR}/01_sft.log"
fi

# 必须有 checkpoint-200 才能继续
if [[ ! -d "${SFT_INIT_CKPT}" ]]; then
    echo "[ERROR] expected ${SFT_INIT_CKPT} not found after SFT"
    echo "  available checkpoints:"
    ls -d "${SFT_DIR}"/checkpoint-* 2>/dev/null || echo "  (none)"
    exit 1
fi
echo "[OK] SFT done. Init for RL: ${SFT_INIT_CKPT}"

# ============================================================
# Step 2: Eval all SFT checkpoints
# ============================================================
banner "STEP 2/5  Eval all SFT checkpoints"

if eval_done "${EVAL_SFT_DIR}"; then
    echo "[skip] SFT eval results already exist in ${EVAL_SFT_DIR}"
else
    echo "[run] evaluating SFT..."
    python eval_compositional.py \
        --auto_discover "${SFT_DIR}" \
        --batch_size 8 \
        --max_new_tokens 128 \
        --out_dir "${EVAL_SFT_DIR}" \
        2>&1 | tee "${LOG_DIR}/02_eval_sft.log"
fi
echo "[OK] SFT eval done -> ${EVAL_SFT_DIR}"

# ============================================================
# Step 3: RL no-anchor (kl=0) drift experiment
# ============================================================
banner "STEP 3/5  RL no-anchor (kl=0) from SFT-200, 1500 steps"

if ckpt_exists "${RL_DIR}"; then
    echo "[skip] RL checkpoints already exist in ${RL_DIR}"
    ls -d "${RL_DIR}"/checkpoint-* 2>/dev/null || true
else
    echo "[run] launching RL no-anchor..."
    torchrun --nproc_per_node=${NPROC} rl_grpo_2hop.py \
        --model_name_or_path     "${SFT_INIT_CKPT}" \
        --train_file             prepared_data_2hop/train_2hop.jsonl \
        --eval_file              prepared_data_2hop/eval_2hop.jsonl \
        --output_dir             "${RL_DIR}" \
        --reward_mode            chain_binary \
        --bridge_gate_floor      0.0 \
        --max_steps              1500 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --num_generations        4 \
        --learning_rate          5e-7 \
        --kl_beta                0.0 \
        --temperature            0.9 \
        --top_p                  0.95 \
        --eval_steps             99999 \
        --save_steps             200 \
        --logging_steps          10 \
        --seed                   42 \
        2>&1 | tee "${LOG_DIR}/03_rl_noanchor.log"
fi
echo "[OK] RL done. Checkpoints in ${RL_DIR}"

# ============================================================
# Step 4: Eval all RL checkpoints
# ============================================================
banner "STEP 4/5  Eval all RL no-anchor checkpoints"

if eval_done "${EVAL_RL_DIR}"; then
    echo "[skip] RL eval results already exist in ${EVAL_RL_DIR}"
else
    echo "[run] evaluating RL checkpoints..."
    python eval_compositional.py \
        --auto_discover "${RL_DIR}" \
        --batch_size 8 \
        --max_new_tokens 128 \
        --out_dir "${EVAL_RL_DIR}" \
        2>&1 | tee "${LOG_DIR}/04_eval_rl.log"
fi
echo "[OK] RL eval done -> ${EVAL_RL_DIR}"

# ============================================================
# Step 5: Drift analysis
# ============================================================
banner "STEP 5/5  Drift analysis"

python analyze_drift.py 2>&1 | tee "${LOG_DIR}/05_drift_analysis.log"

# ============================================================
# Summary
# ============================================================
banner "ALL DONE"
echo ""
echo "Outputs:"
echo "  SFT checkpoints       : ${SFT_DIR}/checkpoint-{50,100,150,200}"
echo "  RL  checkpoints       : ${RL_DIR}/checkpoint-*"
echo "  SFT eval results      : ${EVAL_SFT_DIR}/"
echo "  RL  eval results      : ${EVAL_RL_DIR}/"
echo "  Drift analysis output : drift_analysis_output/"
echo "  Logs                  : ${LOG_DIR}/"
echo ""
echo "Key files to inspect:"
echo "  - drift_analysis_output/drift_analysis.json"
echo "  - drift_analysis_output/drift_2hop.png"
echo "  - ${LOG_DIR}/05_drift_analysis.log  (verdict at the bottom)"
echo ""
echo "Done at $(date '+%Y-%m-%d %H:%M:%S')"