#!/bin/bash
# ============================================================
# RESTART: continue drift experiment after disk-quota crash
#
# Recovers state:
#   - from-50 RL: 1500 steps DONE; only "final" save failed
#   - from-100   : not started
#
# Plan:
#   1. Diagnose + clean broken ckpt dirs
#   2. Skip from-50 RL (intermediate ckpts exist) -> eval
#   3. Run from-100 RL fresh -> eval
#   4. Print summary table
# ============================================================
set -u
set -o pipefail
# 注意: 故意不用 `set -e`, 让磁盘/save 之类的尾部错误不中断整个流程

# ---------- 配置 ----------
SFT_DIR="checkpoints/qwen2.5-3b-2hop-sft-fine"
INIT_STEPS=(50 100)
KL_BETA=0.0
RL_MAX_STEPS=1500
LR=5e-7
NUM_GEN=4
TEMP=0.9
TOP_P=0.95
SAVE_EVERY=200
NPROC=2

LOG_DIR="logs_drift_early_init"
mkdir -p "${LOG_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ---------- 工具函数 ----------
banner() {
    echo ""
    echo "############################################################"
    echo "# $1"
    echo "# $(date '+%Y-%m-%d %H:%M:%S')"
    echo "############################################################"
}

ckpt_exists() {
    local d="$1"
    [[ -d "$d" ]] && [[ -n "$(ls -d "$d"/checkpoint-* 2>/dev/null)" ]]
}

eval_done() {
    local d="$1"
    [[ -f "${d}/all_checkpoints_summary.json" ]] || \
    [[ -n "$(find "${d}" -name 'summary.json' 2>/dev/null | head -n1)" ]]
}

# 删除没有任何权重文件的"半成品"checkpoint
clean_broken_ckpt() {
    local d="$1"
    [[ -d "$d" ]] || return 0
    local nuked=0
    for ck in "${d}"/checkpoint-* "${d}"/final; do
        [[ -d "$ck" ]] || continue
        if ! ls "${ck}"/*.safetensors >/dev/null 2>&1 \
           && ! ls "${ck}"/pytorch_model*.bin >/dev/null 2>&1; then
            echo "  [clean] removing incomplete: ${ck}"
            rm -rf "${ck}"
            nuked=$((nuked+1))
        fi
    done
    [[ $nuked -gt 0 ]] && echo "  [clean] removed ${nuked} dir(s) in ${d}"
    return 0
}

# ============================================================
# Step 0: 诊断
# ============================================================
banner "STEP 0  Diagnose & clean"

echo "--- disk usage ---"
df -h . 2>/dev/null | head -n 5 || true
echo ""

for s in "${INIT_STEPS[@]}"; do
    rl_dir="checkpoints/qwen2.5-3b-grpo-noanchor-from-${s}"
    echo "--- ${rl_dir} ---"
    if [[ -d "${rl_dir}" ]]; then
        clean_broken_ckpt "${rl_dir}"
        echo "  surviving checkpoints:"
        ls -d "${rl_dir}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n || echo "    (none)"
    else
        echo "  (does not exist yet)"
    fi
    echo ""
done

# ============================================================
# 主循环
# ============================================================
for s in "${INIT_STEPS[@]}"; do
    SFT_INIT_CKPT="${SFT_DIR}/checkpoint-${s}"
    RL_DIR="checkpoints/qwen2.5-3b-grpo-noanchor-from-${s}"
    EVAL_RL_DIR="eval_results_rl_noanchor_from_${s}"

    # ---------- RL training ----------
    banner "RL kl=${KL_BETA} from SFT-${s}"

    if ckpt_exists "${RL_DIR}"; then
        echo "[skip] RL checkpoints already exist:"
        ls -d "${RL_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n
    else
        if [[ ! -d "${SFT_INIT_CKPT}" ]]; then
            echo "[ERROR] SFT init missing: ${SFT_INIT_CKPT}; skipping"
            continue
        fi
        echo "[run] launching RL from ${SFT_INIT_CKPT}..."
        torchrun --nproc_per_node=${NPROC} rl_grpo_2hop.py \
            --model_name_or_path          "${SFT_INIT_CKPT}" \
            --train_file                  prepared_data_2hop/train_2hop.jsonl \
            --eval_file                   prepared_data_2hop/eval_2hop.jsonl \
            --output_dir                  "${RL_DIR}" \
            --reward_mode                 chain_binary \
            --bridge_gate_floor           0.0 \
            --max_steps                   ${RL_MAX_STEPS} \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 8 \
            --num_generations             ${NUM_GEN} \
            --learning_rate               ${LR} \
            --kl_beta                     ${KL_BETA} \
            --temperature                 ${TEMP} \
            --top_p                       ${TOP_P} \
            --eval_steps                  99999 \
            --save_steps                  ${SAVE_EVERY} \
            --logging_steps               10 \
            --seed                        42 \
            2>&1 | tee -a "${LOG_DIR}/rl_from_${s}.log"
        rc=${PIPESTATUS[0]}
        echo "[info] RL exit code = ${rc}"
        # 不管退出码，再清一次半成品
        clean_broken_ckpt "${RL_DIR}"
        if ! ckpt_exists "${RL_DIR}"; then
            echo "[ERROR] no usable checkpoint produced for from-${s}; skipping eval"
            continue
        fi
        if [[ ${rc} -ne 0 ]]; then
            echo "[WARN] RL had non-zero exit (likely the trailing 'final' save); intermediate ckpts OK, proceeding"
        fi
    fi

    # ---------- Eval ----------
    banner "Eval RL from-${s}"

    if eval_done "${EVAL_RL_DIR}"; then
        echo "[skip] eval already done: ${EVAL_RL_DIR}"
    else
        mkdir -p "${EVAL_RL_DIR}"
        echo "[run] evaluating ${RL_DIR}..."
        python eval_compositional.py \
            --auto_discover "${RL_DIR}" \
            --batch_size    8 \
            --max_new_tokens 128 \
            --out_dir       "${EVAL_RL_DIR}" \
            2>&1 | tee -a "${LOG_DIR}/eval_from_${s}.log"
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            echo "[WARN] eval had errors for from-${s}; continuing"
        fi
    fi
done

# ============================================================
# 摘要
# ============================================================
banner "Quick summary"

python - <<'PYEOF'
import json, os, glob, re

def step_key(p):
    m = re.search(r'checkpoint-(\d+)', p)
    return int(m.group(1)) if m else 10**9

for init in (50, 100):
    eval_dir = f"eval_results_rl_noanchor_from_{init}"
    if not os.path.isdir(eval_dir):
        print(f"\n[skip] {eval_dir} not found")
        continue
    print(f"\n=== init = SFT-{init} ===")
    print(f"{'ckpt':>14}  {'2hop EM':>8}  {'2hop F1':>8}  {'ChainEM':>8}  {'BridgeR':>8}")
    summaries = sorted(
        glob.glob(f"{eval_dir}/checkpoint-*/summary.json") +
        glob.glob(f"{eval_dir}/final/summary.json"),
        key=step_key)
    for s in summaries:
        try:
            d = json.load(open(s))
            r = d.get("2hop", {}) or d.get("eval_2hop", {}) or {}
            ckpt = os.path.basename(os.path.dirname(s))
            em  = r.get('EM',  float('nan'))
            f1  = r.get('F1',  float('nan'))
            cem = r.get('ChainEM', r.get('chain_em', float('nan')))
            br  = r.get('BridgeR', r.get('bridge_recall', float('nan')))
            print(f"{ckpt:>14}  {em:>8.2f}  {f1:>8.2f}  {cem:>8.2f}  {br:>8.2f}")
        except Exception as e:
            print(f"  [err] {s}: {e}")
PYEOF

banner "DONE"
echo ""
echo "Outputs:"
for s in "${INIT_STEPS[@]}"; do
    echo "  RL ckpts (init=${s}) : checkpoints/qwen2.5-3b-grpo-noanchor-from-${s}/"
    echo "  Eval     (init=${s}) : eval_results_rl_noanchor_from_${s}/"
done
echo "  Logs                 : ${LOG_DIR}/"