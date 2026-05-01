#!/bin/bash
# ============================================================
# run_drift_experiment.sh
#
# Engineered drift experiment (corresponds to ALPINE paper §5.2):
#   reward = 0.15*format + 0.85*bridge - λ*answer_em
#
#   A) DRIFT     : kl_beta=0.0     -> expect Bridge↔, EM↓
#   B) ANCHORED  : kl_beta=0.05    -> expect Bridge↔, EM↔ (drift fixed)
#
# 总耗时预估 (2x GPU): ~3-4h (drift 1.5h + anchored 1.5h + eval ~30min)
# ============================================================
set -e
set -u
set -o pipefail

# ============================================================
# 配置 (按需改)
# ============================================================
SFT_INIT=checkpoints/qwen2.5-3b-2hop-sft-fine/checkpoint-100
TRAIN_FILE=prepared_data_2hop/train_2hop.jsonl
EVAL_FILE=prepared_data_2hop/eval_2hop.jsonl

ANTI_W=1.0         # λ in reward = base - λ*answer_em
MAX_STEPS=1500
SAVE_STEPS=200
LR=1e-6
TEMP=1.0
TOP_P=0.97
NPROC=2

# 实验 A: no anchor (drift)
RL_DRIFT_DIR=checkpoints/qwen2.5-3b-grpo-anti_answer-from-100
EVAL_DRIFT_DIR=eval_results_anti_answer_from_100

# 实验 B: anchored (drift fixed)
RL_ANCHOR_DIR=checkpoints/qwen2.5-3b-grpo-anti_answer-anchored-from-100
EVAL_ANCHOR_DIR=eval_results_anti_answer_anchored_from_100

# SFT-100 baseline (做对照参考点)
SFT100_EVAL_DIR=eval_results_sft_100_baseline

LOG_DIR=logs_drift_v3
mkdir -p "$LOG_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ============================================================
# 工具函数
# ============================================================
banner() {
    echo ""
    echo "############################################################"
    echo "# $1"
    echo "# $(date '+%F %T')"
    echo "############################################################"
}

clean_broken_ckpt() {
    # 清掉磁盘满 / Ctrl-C 时留下的不完整 checkpoint
    local d="$1"
    [[ -d "$d" ]] || return 0
    for ck in "${d}"/checkpoint-* "${d}"/final "${d}"/best; do
        [[ -d "$ck" ]] || continue
        local has_w=0 has_t=0
        ls "${ck}"/*.safetensors        >/dev/null 2>&1 && has_w=1
        ls "${ck}"/pytorch_model*.bin   >/dev/null 2>&1 && has_w=1
        ls "${ck}"/tokenizer*.json      >/dev/null 2>&1 && has_t=1
        ls "${ck}"/vocab.json           >/dev/null 2>&1 && has_t=1
        if [[ $has_w -eq 0 ]] || [[ $has_t -eq 0 ]]; then
            echo "  [clean] removing incomplete: ${ck}"
            rm -rf "${ck}"
        fi
    done
}

ckpt_exists() {
    local d="$1"
    local n
    n=$(ls -d "${d}"/checkpoint-* 2>/dev/null | wc -l || true)
    [[ "${n}" -gt 0 ]]
}

# ============================================================
# Step 0: 前置检查 + 清理
# ============================================================
banner "Step 0: Pre-flight checks & cleanup"

[[ -d "$SFT_INIT" ]]    || { echo "[ERR] SFT init not found: $SFT_INIT"; exit 1; }
[[ -f "$TRAIN_FILE" ]]  || { echo "[ERR] missing $TRAIN_FILE"; exit 1; }
[[ -f "$EVAL_FILE" ]]   || { echo "[ERR] missing $EVAL_FILE"; exit 1; }
[[ -f "rl_grpo_2hop.py" ]]      || { echo "[ERR] missing rl_grpo_2hop.py"; exit 1; }
[[ -f "eval_compositional.py" ]]|| { echo "[ERR] missing eval_compositional.py"; exit 1; }

clean_broken_ckpt "$RL_DRIFT_DIR"
clean_broken_ckpt "$RL_ANCHOR_DIR"
nvidia-smi | head -20
df -h ~ 2>/dev/null | head -2 || true
echo "[OK]"

# ============================================================
# Step 1: Eval SFT-100 baseline
# ============================================================
banner "Step 1: Eval SFT-100 baseline"

if [[ -f "${SFT100_EVAL_DIR}/checkpoint-100/summary.json" ]]; then
    echo "  [skip] baseline already evaluated"
else
    python eval_compositional.py \
        --model_dirs "$SFT_INIT" \
        --batch_size 8 \
        --max_new_tokens 128 \
        --out_dir "$SFT100_EVAL_DIR" \
        2>&1 | tee "${LOG_DIR}/01_eval_sft_100.log"
fi

# ============================================================
# Step 2: Train  DRIFT (kl_beta = 0)
# ============================================================
banner "Step 2: Train DRIFT (kl_beta=0, λ=${ANTI_W})"

if ckpt_exists "$RL_DRIFT_DIR"; then
    echo "  [skip] DRIFT checkpoints already exist:"
    ls "$RL_DRIFT_DIR" | head
    echo "  rm -rf $RL_DRIFT_DIR  to retrain"
else
    torchrun --nproc_per_node=$NPROC rl_grpo_2hop.py \
        --model_name_or_path           "$SFT_INIT" \
        --train_file                   "$TRAIN_FILE" \
        --eval_file                    "$EVAL_FILE" \
        --output_dir                   "$RL_DRIFT_DIR" \
        --reward_mode                  bridge_minus_answer \
        --anti_answer_weight           "$ANTI_W" \
        --kl_beta                      0.0 \
        --max_steps                    "$MAX_STEPS" \
        --per_device_train_batch_size  2 \
        --gradient_accumulation_steps  4 \
        --num_generations              4 \
        --learning_rate                "$LR" \
        --temperature                  "$TEMP" \
        --top_p                        "$TOP_P" \
        --max_new_tokens               128 \
        --disable_std_norm \
        --eval_steps                   99999 \
        --save_steps                   "$SAVE_STEPS" \
        --logging_steps                10 \
        --seed                         42 \
        2>&1 | tee "${LOG_DIR}/02_train_drift.log"
fi

# ============================================================
# Step 3: Eval DRIFT checkpoints
# ============================================================
banner "Step 3: Eval all DRIFT checkpoints"

clean_broken_ckpt "$RL_DRIFT_DIR"
python eval_compositional.py \
    --auto_discover "$RL_DRIFT_DIR" \
    --batch_size 8 \
    --max_new_tokens 128 \
    --out_dir "$EVAL_DRIFT_DIR" \
    2>&1 | tee "${LOG_DIR}/03_eval_drift.log"

# ============================================================
# Step 4: Train  ANCHORED (kl_beta = 0.05)
# ============================================================
banner "Step 4: Train ANCHORED (kl_beta=0.05, λ=${ANTI_W})"

if ckpt_exists "$RL_ANCHOR_DIR"; then
    echo "  [skip] ANCHORED checkpoints already exist:"
    ls "$RL_ANCHOR_DIR" | head
    echo "  rm -rf $RL_ANCHOR_DIR  to retrain"
else
    torchrun --nproc_per_node=$NPROC rl_grpo_2hop.py \
        --model_name_or_path           "$SFT_INIT" \
        --train_file                   "$TRAIN_FILE" \
        --eval_file                    "$EVAL_FILE" \
        --output_dir                   "$RL_ANCHOR_DIR" \
        --reward_mode                  bridge_minus_answer \
        --anti_answer_weight           "$ANTI_W" \
        --kl_beta                      0.05 \
        --max_steps                    "$MAX_STEPS" \
        --per_device_train_batch_size  2 \
        --gradient_accumulation_steps  4 \
        --num_generations              4 \
        --learning_rate                "$LR" \
        --temperature                  "$TEMP" \
        --top_p                        "$TOP_P" \
        --max_new_tokens               128 \
        --disable_std_norm \
        --eval_steps                   99999 \
        --save_steps                   "$SAVE_STEPS" \
        --logging_steps                10 \
        --seed                         42 \
        2>&1 | tee "${LOG_DIR}/04_train_anchored.log"
fi

# ============================================================
# Step 5: Eval ANCHORED checkpoints
# ============================================================
banner "Step 5: Eval all ANCHORED checkpoints"

clean_broken_ckpt "$RL_ANCHOR_DIR"
python eval_compositional.py \
    --auto_discover "$RL_ANCHOR_DIR" \
    --batch_size 8 \
    --max_new_tokens 128 \
    --out_dir "$EVAL_ANCHOR_DIR" \
    2>&1 | tee "${LOG_DIR}/05_eval_anchored.log"

# ============================================================
# Step 6: Print comparison table
# ============================================================
banner "Step 6: Drift vs Anchored summary"

python - <<EOF
import json, os, math
from pathlib import Path

def fmt(x, d="  -  "):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return d
    return f"{x:6.2f}"

def load_summary(p):
    p = Path(p) / "summary.json"
    return json.load(open(p)) if p.exists() else {}

def get(d, split, *keys):
    if split not in d: return None
    r = d[split]
    for k in keys:
        if k in r: return r[k]
    return None

def step_key(name):
    if name.startswith('checkpoint-'):
        try: return (0, int(name.split('-')[-1]))
        except: return (0, 0)
    if name == 'final': return (2, 0)
    if name == 'best':  return (3, 0)
    return (1, name)

def show(title, eval_dir, splits=('2hop','3hop_linear','4hop_linear')):
    print(f"\n=== {title} ===")
    if not os.path.isdir(eval_dir):
        print(f"  (dir not found: {eval_dir})")
        return
    ckpts = sorted([d for d in os.listdir(eval_dir)
                    if os.path.isdir(os.path.join(eval_dir, d))],
                   key=step_key)
    for split in splits:
        rows = []
        for c in ckpts:
            d = load_summary(os.path.join(eval_dir, c))
            if split not in d: continue
            em  = get(d, split, 'em', 'EM')
            f1  = get(d, split, 'f1', 'F1')
            br  = get(d, split, 'bridge_recall', 'BridgeR')
            bf1 = get(d, split, 'bridge_f1', 'BridgeF1')
            cem = get(d, split, 'chain_em', 'ChainEM')
            rows.append((c, em, f1, br, bf1, cem))
        if not rows: continue
        print(f"\n  -- {split} --")
        print(f"  {'Ckpt':<22}{'EM':>8}{'F1':>8}{'BridgeR':>10}{'BridgeF1':>10}{'ChainEM':>10}")
        print("  " + "-"*68)
        for c, em, f1, br, bf1, cem in rows:
            print(f"  {c:<22}{fmt(em)}{fmt(f1)}{fmt(br)}{fmt(bf1)}{fmt(cem)}")

show("[Baseline]  SFT-100",                      "${SFT100_EVAL_DIR}")
show("[A] DRIFT (no anchor, kl=0)",              "${EVAL_DRIFT_DIR}")
show("[B] ANCHORED (kl=0.05)  - drift fixed?",   "${EVAL_ANCHOR_DIR}")

print()
print("=" * 80)
print("解读 (对应 ALPINE paper §5.2 Figure 4 vs 5.3 Figure 6):")
print("")
print("  [A] DRIFT:    随 step ↑")
print("                BridgeR / BridgeF1 :  保持 90+   (bridge 在 reward 里)")
print("                EM   / F1          :  显著下降  (answer 被 -λ 惩罚)")
print("                ChainEM            :  跟 EM 一起掉")
print("                ↑↑↑ 这就是 paper 5.2 节的 'goal-grounding collapse'")
print("")
print("  [B] ANCHORED: 随 step ↑")
print("                BridgeR / BridgeF1 :  保持 90+")
print("                EM   / F1          :  跟 baseline 持平 / 微跌")
print("                ↑↑↑ 这就是 paper 5.3 节 'KL anchoring stabilizes RL'")
print("=" * 80)
EOF

banner "ALL DONE"
echo "Outputs:"
echo "  Baseline eval : ${SFT100_EVAL_DIR}/"
echo "  DRIFT  ckpt   : ${RL_DRIFT_DIR}/"
echo "  DRIFT  eval   : ${EVAL_DRIFT_DIR}/"
echo "  ANCHOR ckpt   : ${RL_ANCHOR_DIR}/"
echo "  ANCHOR eval   : ${EVAL_ANCHOR_DIR}/"
echo "  Logs          : ${LOG_DIR}/"
echo ""
echo "训练 reward 轨迹快查:"
echo "  for d in $RL_DRIFT_DIR $RL_ANCHOR_DIR; do"
echo "    echo \"== \$d ==\"; tail -5 \$d/train_log.jsonl"
echo "  done"