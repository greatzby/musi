#!/bin/bash
set -e

# ============================================================
# Continue / resume run after OOM
#
# 关键改动 vs 上一版:
#   1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  (修内存碎片)
#   2. token_logprobs 已改为 fused cross_entropy (在 .py 里, 节省 ~10 GiB)
#   3. --eval_steps 99999  关掉中途 eval (按你要求)
#   4. BS=1 GA=8           (effective batch 不变, peak mem 减半)
# ============================================================

# ★ 修内存碎片化:必须在 torchrun 之前 export
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REWARD_MODE=chain_binary
RL_TEMP=0.9
LR=5e-7
HARD_FILE=hard_data_from_50_sampled/train_2hop_mixed.jsonl

# ---------- 自动找 from-50 的 resume 点 ----------
FROM50_DIR=checkpoints/qwen2.5-3b-grpo-from-50-v3
if   [ -d "${FROM50_DIR}/best" ];           then RESUME_CKPT=${FROM50_DIR}/best
elif [ -d "${FROM50_DIR}/checkpoint-100" ]; then RESUME_CKPT=${FROM50_DIR}/checkpoint-100
else
    echo "[ERROR] cannot find resume ckpt under ${FROM50_DIR}/"
    ls -la ${FROM50_DIR}/ 2>/dev/null || true
    exit 1
fi
echo "[info] resuming from-50 from: ${RESUME_CKPT}"

RESUME_REF=checkpoints/qwen2.5-3b-2hop-sft-early/checkpoint-50


############################################
# Step A: continue from-50 (OOM-fixed, no mid-train eval)
############################################
OUT_DIR_50=checkpoints/qwen2.5-3b-grpo-from-50-v3-cont
if [ -d "${OUT_DIR_50}/final" ]; then
    echo "[skip] ${OUT_DIR_50}/final already exists"
else
    echo ""
    echo "############################################"
    echo "# Continue GRPO from-50 (300 more steps)"
    echo "############################################"
    torchrun --nproc_per_node=2 rl_grpo_2hop_new.py \
        --model_name_or_path     ${RESUME_CKPT} \
        --ref_model_name_or_path ${RESUME_REF} \
        --train_file             ${HARD_FILE} \
        --eval_file              prepared_data_2hop/eval_2hop.jsonl \
        --output_dir             ${OUT_DIR_50} \
        --include_bridge_count \
        --reward_mode            ${REWARD_MODE} \
        --bridge_gate_floor      0.0 \
        --max_steps              300 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --num_generations        4 \
        --learning_rate          ${LR} \
        --kl_beta                0.02 \
        --temperature            ${RL_TEMP} \
        --top_p                  0.95 \
        --eval_steps             99999 \
        --eval_subset_size       200 \
        --save_steps             100 \
        --logging_steps          10 \
        --seed                   42
fi


############################################
# Step B: 还没跑的 SFT ckpt (225 / 450 / 675)
############################################
REMAINING_CKPTS=(225 450 675)
for CKPT in "${REMAINING_CKPTS[@]}"; do
    SFT_DIR=checkpoints/qwen2.5-3b-2hop-sft/checkpoint-${CKPT}
    OUT_DIR=checkpoints/qwen2.5-3b-grpo-from-${CKPT}-v3

    if [ -d "${OUT_DIR}/final" ]; then
        echo "[skip] ${OUT_DIR}/final already exists"
        continue
    fi
    if [ ! -d "${SFT_DIR}" ]; then
        echo "[skip] SFT ckpt missing: ${SFT_DIR}"
        continue
    fi

    echo ""
    echo "############################################"
    echo "# GRPO from SFT ckpt-${CKPT}"
    echo "############################################"
    torchrun --nproc_per_node=2 rl_grpo_2hop_new.py \
        --model_name_or_path     ${SFT_DIR} \
        --ref_model_name_or_path ${SFT_DIR} \
        --train_file             ${HARD_FILE} \
        --eval_file              prepared_data_2hop/eval_2hop.jsonl \
        --output_dir             ${OUT_DIR} \
        --include_bridge_count \
        --reward_mode            ${REWARD_MODE} \
        --bridge_gate_floor      0.0 \
        --max_steps              400 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --num_generations        4 \
        --learning_rate          ${LR} \
        --kl_beta                0.02 \
        --temperature            ${RL_TEMP} \
        --top_p                  0.95 \
        --eval_steps             99999 \
        --eval_subset_size       200 \
        --save_steps             100 \
        --logging_steps          10 \
        --seed                   42
done


############################################
# Step C: Eval all (oracle hop count) — 训练全部跑完再做
############################################
declare -A OUT_DIR_MAP
OUT_DIR_MAP[50]=${OUT_DIR_50}
OUT_DIR_MAP[225]=checkpoints/qwen2.5-3b-grpo-from-225-v3
OUT_DIR_MAP[450]=checkpoints/qwen2.5-3b-grpo-from-450-v3
OUT_DIR_MAP[675]=checkpoints/qwen2.5-3b-grpo-from-675-v3

for CKPT in 50 225 450 675; do
    OUT_DIR=${OUT_DIR_MAP[$CKPT]}
    if [ ! -d "${OUT_DIR}" ]; then
        echo "[skip eval] ${OUT_DIR} not found"
        continue
    fi
    echo ""
    echo "############################################"
    echo "# Eval (oracle hop) RL from ckpt-${CKPT}"
    echo "############################################"
    python eval_compositional.py \
        --auto_discover ${OUT_DIR} \
        --include_bridge_count \
        --batch_size 8 \
        --max_new_tokens 128 \
        --out_dir eval_results_grpo_from_${CKPT}_v3_oracle
done

echo ""
echo "All done!"