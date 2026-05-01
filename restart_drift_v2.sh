#!/bin/bash
# restart_drift_v2.sh
# 目标 drift: bridge 保持高, final answer accuracy 掉
# 方法: reward = process_only (只奖励 bridge), kl=0 (无 anchor)
#       + 略大 LR + 略高 temperature + Dr.GRPO

set -e

############################################################
# 配置
############################################################
SFT_INIT=checkpoints/qwen2.5-3b-2hop-sft-fine/checkpoint-100
TRAIN_FILE=prepared_data_2hop/train_2hop.jsonl
EVAL_FILE=prepared_data_2hop/eval_2hop.jsonl

RL_DIR=checkpoints/qwen2.5-3b-grpo-process_only-from-100
EVAL_DIR=eval_results_grpo_process_only_from_100
SFT100_EVAL_DIR=eval_results_sft_100_baseline

LOG_DIR=logs_drift_v2
mkdir -p "$LOG_DIR"

NPROC=2

############################################################
# Helper: 清理不完整 checkpoint (修复上次磁盘满导致的 final/ 损坏)
############################################################
clean_broken_ckpt() {
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
            echo "  [clean] removing incomplete: ${ck} (weights=$has_w tokenizer=$has_t)"
            rm -rf "${ck}"
        fi
    done
}

############################################################
# Step 0: 清理 + 诊断
############################################################
echo "############################################################"
echo "# Step 0: Diagnose & clean   $(date '+%F %T')"
echo "############################################################"
df -h ~ 2>/dev/null | head -2
echo ""

# 清理上次磁盘满留下的损坏目录 (尤其是 from-50/final)
clean_broken_ckpt checkpoints/qwen2.5-3b-grpo-noanchor-from-50
clean_broken_ckpt checkpoints/qwen2.5-3b-grpo-noanchor-from-100
clean_broken_ckpt "$RL_DIR"

if [[ ! -d "$SFT_INIT" ]]; then
    echo "[ERROR] SFT init not found: $SFT_INIT"
    exit 1
fi
echo "  [ok] SFT init: $SFT_INIT"

############################################################
# Step 1: Eval SFT-100 baseline (drift 起点参照)
############################################################
echo ""
echo "############################################################"
echo "# Step 1: Eval SFT-100 baseline   $(date '+%F %T')"
echo "############################################################"
if [[ -f "${SFT100_EVAL_DIR}/checkpoint-100/summary.json" ]]; then
    echo "  [skip] baseline already evaluated at $SFT100_EVAL_DIR"
else
    python eval_compositional.py \
        --model_dirs "$SFT_INIT" \
        --batch_size 8 \
        --max_new_tokens 128 \
        --out_dir "$SFT100_EVAL_DIR" \
        2>&1 | tee "${LOG_DIR}/eval_sft_100_baseline.log"
fi

############################################################
# Step 2: RL with process_only, kl=0
############################################################
echo ""
echo "############################################################"
echo "# Step 2: RL  reward=process_only  kl=0   $(date '+%F %T')"
echo "############################################################"

if ls "$RL_DIR"/checkpoint-*/config.json >/dev/null 2>&1; then
    echo "  [info] 发现已有 RL checkpoint, 跳过训练:"
    ls "$RL_DIR" | head
    echo "  如要重跑: rm -rf $RL_DIR"
else
    torchrun --nproc_per_node=$NPROC rl_grpo_2hop.py \
        --model_name_or_path           "$SFT_INIT" \
        --train_file                   "$TRAIN_FILE" \
        --eval_file                    "$EVAL_FILE" \
        --output_dir                   "$RL_DIR" \
        --reward_mode                  process_only \
        --kl_beta                      0.0 \
        --max_steps                    1500 \
        --per_device_train_batch_size  2 \
        --gradient_accumulation_steps  4 \
        --num_generations              4 \
        --learning_rate                1e-6 \
        --temperature                  1.0 \
        --top_p                        0.97 \
        --max_new_tokens               128 \
        --disable_std_norm \
        --eval_steps                   99999 \
        --save_steps                   200 \
        --logging_steps                10 \
        --seed                         42 \
        2>&1 | tee "${LOG_DIR}/rl_process_only_from_100.log"
fi

############################################################
# Step 3: Eval all RL checkpoints
############################################################
echo ""
echo "############################################################"
echo "# Step 3: Eval all RL checkpoints   $(date '+%F %T')"
echo "############################################################"

clean_broken_ckpt "$RL_DIR"

python eval_compositional.py \
    --auto_discover "$RL_DIR" \
    --batch_size 8 \
    --max_new_tokens 128 \
    --out_dir "$EVAL_DIR" \
    2>&1 | tee "${LOG_DIR}/eval_process_only_from_100.log"

############################################################
# Step 4: Drift 总结 (兼容大小写 key)
############################################################
echo ""
echo "############################################################"
echo "# Step 4: Drift summary   $(date '+%F %T')"
echo "############################################################"

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
        print(f"  (directory not found: {eval_dir})")
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

show("SFT-100 baseline",         "${SFT100_EVAL_DIR}")
show("RL process_only from-100", "${EVAL_DIR}")

print()
print("=" * 78)
print("怎么读这张表 (期望的 drift 模式):")
print("  随 checkpoint 步数增加:")
print("    BridgeR / BridgeF1 :  保持 90+ 或微涨   ← 因为 bridge 在 reward 里")
print("    EM   / F1          :  单调下降          ← 因为 answer 不在 reward 里")
print("    ChainEM            :  下降 (取决于 EM)")
print("=" * 78)
EOF

echo ""
echo "############################################################"
echo "# DONE   $(date '+%F %T')"
echo "############################################################"
echo "Outputs:"
echo "  RL ckpts : $RL_DIR"
echo "  Eval     : $EVAL_DIR"
echo "  Logs     : $LOG_DIR"
echo ""
echo "查看训练 reward 轨迹:"
echo "  python -c \"import json; [print(json.dumps({k:v for k,v in json.loads(l).items() if k in ['step','reward_mean','bridge','answer_f1','answer_em']}, ensure_ascii=False)) for l in open('${RL_DIR}/train_log.jsonl')]\""