#!/bin/bash
# ============================================================
# DRIFT EXPERIMENT — kl=0 RL from EARLIER SFT checkpoints
#
# Goal: 验证从更弱的 SFT base 启动时, 没有 KL anchor 的 GRPO
#       是否会发生 drift / collapse.
#
# Strategy: 从 SFT-50 和 SFT-100 分别启动 RL (kl=0, 1500 steps),
#           保存每 200 步的 checkpoint, 然后 eval 全部.
#
# 总耗时预估: 2 × (~10h RL + ~40min eval) ≈ 22h
# ============================================================
set -e
set -u
set -o pipefail

# ============================================================
# 配置
# ============================================================
SFT_DIR="checkpoints/qwen2.5-3b-2hop-sft-fine"

# 从这两个早期 checkpoint 启动 RL — 改这里就能加更多 init 点
INIT_STEPS=(50 100)

# RL 超参 (kl_beta=0 是这次实验的核心变量)
KL_BETA=0.0
RL_MAX_STEPS=1500
LR=5e-7
NUM_GEN=4
TEMP=0.9
TOP_P=0.95
SAVE_EVERY=200

LOG_DIR="logs_drift_early_init"
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
    local d="$1"
    local n
    n=$(ls -d "${d}"/checkpoint-* 2>/dev/null | wc -l || true)
    [[ "${n}" -gt 0 ]]
}

eval_done() {
    local d="$1"
    [[ -f "${d}/all_checkpoints_summary.json" ]] || \
    [[ -n "$(find "${d}" -name 'summary.json' 2>/dev/null | head -n1)" ]]
}

# ============================================================
# 前置检查
# ============================================================
banner "Pre-flight checks"

for s in "${INIT_STEPS[@]}"; do
    ckpt="${SFT_DIR}/checkpoint-${s}"
    if [[ ! -d "${ckpt}" ]]; then
        echo "[ERROR] required SFT checkpoint missing: ${ckpt}"
        echo "  Available SFT checkpoints:"
        ls -d "${SFT_DIR}"/checkpoint-* 2>/dev/null || echo "  (none)"
        exit 1
    fi
    echo "[OK] found ${ckpt}"
done

for f in prepared_data_2hop/train_2hop.jsonl \
         prepared_data_2hop/eval_2hop.jsonl \
         rl_grpo_2hop.py \
         eval_compositional.py; do
    if [[ ! -f "${f}" ]]; then
        echo "[ERROR] missing: ${f}"
        exit 1
    fi
done

nvidia-smi || { echo "[ERROR] nvidia-smi failed"; exit 1; }
echo "[OK] all pre-flight checks passed"

# ============================================================
# 核心循环: 对每个 init step 依次跑 RL + Eval
# ============================================================
for s in "${INIT_STEPS[@]}"; do
    SFT_INIT_CKPT="${SFT_DIR}/checkpoint-${s}"
    RL_DIR="checkpoints/qwen2.5-3b-grpo-noanchor-from-${s}"
    EVAL_RL_DIR="eval_results_rl_noanchor_from_${s}"

    # ------------------------------------------------------------
    # RL training
    # ------------------------------------------------------------
    banner "RL kl=${KL_BETA} from SFT-${s} (steps=${RL_MAX_STEPS})"

    if ckpt_exists "${RL_DIR}"; then
        echo "[skip] RL checkpoints already exist in ${RL_DIR}"
        ls -d "${RL_DIR}"/checkpoint-* 2>/dev/null || true
    else
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
            2>&1 | tee "${LOG_DIR}/rl_from_${s}.log"
    fi
    echo "[OK] RL from-${s} done -> ${RL_DIR}"

    # ------------------------------------------------------------
    # Eval all RL checkpoints from this init
    # ------------------------------------------------------------
    banner "Eval RL checkpoints from SFT-${s}"

    if eval_done "${EVAL_RL_DIR}"; then
        echo "[skip] eval results already exist in ${EVAL_RL_DIR}"
    else
        echo "[run] evaluating ${RL_DIR}..."
        python eval_compositional.py \
            --auto_discover "${RL_DIR}" \
            --batch_size    8 \
            --max_new_tokens 128 \
            --out_dir        "${EVAL_RL_DIR}" \
            2>&1 | tee "${LOG_DIR}/eval_from_${s}.log"
    fi
    echo "[OK] eval from-${s} done -> ${EVAL_RL_DIR}"
done

# ============================================================
# 快速摘要: 抓出每个 ckpt 的关键指标 (无需依赖 analyze_drift.py)
# ============================================================
banner "Quick summary of drift trajectory"

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
    summaries = sorted(glob.glob(f"{eval_dir}/checkpoint-*/summary.json") +
                       glob.glob(f"{eval_dir}/final/summary.json"), key=step_key)
    for s in summaries:
        try:
            d = json.load(open(s))
            r = d.get("2hop", {}) or d.get("eval_2hop", {}) or {}
            ckpt = os.path.basename(os.path.dirname(s))
            print(f"{ckpt:>14}  "
                  f"{r.get('EM',  float('nan')):>8.2f}  "
                  f"{r.get('F1',  float('nan')):>8.2f}  "
                  f"{r.get('ChainEM', r.get('chain_em', float('nan'))):>8.2f}  "
                  f"{r.get('BridgeR', r.get('bridge_recall', float('nan'))):>8.2f}")
        except Exception as e:
            print(f"  [err] {s}: {e}")
PYEOF

# ============================================================
# 结尾
# ============================================================
banner "ALL DONE"
echo ""
echo "Outputs:"
for s in "${INIT_STEPS[@]}"; do
    echo "  RL ckpts  (init=${s}): checkpoints/qwen2.5-3b-grpo-noanchor-from-${s}/"
    echo "  Eval      (init=${s}): eval_results_rl_noanchor_from_${s}/"
done
echo "  Logs                  : ${LOG_DIR}/"
echo ""
echo "================================================================"
echo "崩溃判断标准 (看上面的 trajectory):"
echo "  ✅ 真崩了    : ChainEM 单调下降到接近 0, B/A_EM 严重背离"
echo "  ⚠️  轻微 drift: ChainEM 先升后降, 或 BridgeR 显著掉"
echo "  ❌ 仍没学习  : 所有指标基本不动 (R=1.000 zero%=100 的现象重现)"
echo "                  → 说明 SFT-${INIT_STEPS[*]} 也太收敛了, 需要换 reward"
echo "                    (例如 answer_only) 或者更早的 ckpt"
echo "================================================================"
echo ""
echo "Done at $(date '+%Y-%m-%d %H:%M:%S')"