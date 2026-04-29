#!/bin/bash
set -e

# ===== 实验配置 =====
EARLY_CKPT=50               # 用哪个早期 ckpt 做 mining
CKPTS=(50 225 450 675)      # 在哪些 SFT ckpt 上做 RL

# Mining 参数 (sampling-based)
NUM_SAMPLES=4               # 每个样本采几次 (与 RL num_generations 对齐)
MINING_TEMP=0.9
REWARD_THRESHOLD=0.85       # mean_reward < 此值 → hard
TARGET_SIZE=5000            # 至少凑够 5000 条
MIN_EASY_RATIO=0.2          # 最终至少 20% easy 样本

# RL 参数
REWARD_MODE=chain_binary
MAX_STEPS=400
RL_TEMP=0.9
LR=5e-7

# 路径
EARLY_SFT_DIR=checkpoints/qwen2.5-3b-2hop-sft-early/checkpoint-${EARLY_CKPT}
HARD_DIR=hard_data_from_${EARLY_CKPT}_sampled
HARD_FILE=${HARD_DIR}/train_2hop_mixed.jsonl


############################################
# Step 0: Train ckpt-${EARLY_CKPT} SFT (only if missing)
############################################
if [ ! -d "${EARLY_SFT_DIR}" ]; then
    echo ""
    echo "############################################"
    echo "# Train early SFT (${EARLY_CKPT} steps only)"
    echo "############################################"
    torchrun --nproc_per_node=2 sft_train_2hop.py \
        --output_dir       checkpoints/qwen2.5-3b-2hop-sft-early \
        --max_steps        ${EARLY_CKPT} \
        --save_strategy    steps \
        --save_steps       ${EARLY_CKPT} \
        --save_total_limit 1
else
    echo "[skip] early SFT already exists at ${EARLY_SFT_DIR}"
fi


############################################
# Step 1: Mine ONCE from early ckpt
############################################
if [ ! -f "${HARD_FILE}" ]; then
    echo ""
    echo "############################################"
    echo "# Mine hard examples from ckpt-${EARLY_CKPT} (sampling-based)"
    echo "############################################"
    mkdir -p ${HARD_DIR}
    python mine_hard_examples.py \
        --model_dir            ${EARLY_SFT_DIR} \
        --input_file           prepared_data_2hop/train_2hop.jsonl \
        --output_file          ${HARD_FILE} \
        --reward_mode          ${REWARD_MODE} \
        --include_bridge_count \
        --num_samples          ${NUM_SAMPLES} \
        --temperature          ${MINING_TEMP} \
        --top_p                0.95 \
        --reward_threshold     ${REWARD_THRESHOLD} \
        --target_size          ${TARGET_SIZE} \
        --min_easy_ratio       ${MIN_EASY_RATIO} \
        --batch_size           8 \
        --max_new_tokens       128 \
        --max_input_len        4096 \
        --seed                 42
else
    echo "[skip mining] ${HARD_FILE} already exists"
fi


############################################
# Step 2: GRPO from each SFT checkpoint (用同一份 mining data)
############################################
for CKPT in "${CKPTS[@]}"; do
    if [ "${CKPT}" = "${EARLY_CKPT}" ]; then
        SFT_DIR=${EARLY_SFT_DIR}
    else
        SFT_DIR=checkpoints/qwen2.5-3b-2hop-sft/checkpoint-${CKPT}
    fi
    OUT_DIR=checkpoints/qwen2.5-3b-grpo-from-${CKPT}-v3

    if [ -d "${OUT_DIR}/final" ]; then
        echo "[skip RL] ${OUT_DIR}/final already exists"
        continue
    fi

    if [ ! -d "${SFT_DIR}" ]; then
        echo "[skip RL] SFT ckpt missing: ${SFT_DIR}"
        continue
    fi

    echo ""
    echo "############################################"
    echo "# GRPO from SFT ckpt-${CKPT}  (data: ${HARD_FILE})"
    echo "############################################"
    torchrun --nproc_per_node=2 rl_grpo_2hop.py \
        --model_name_or_path     ${SFT_DIR} \
        --ref_model_name_or_path ${SFT_DIR} \
        --train_file             ${HARD_FILE} \
        --eval_file              prepared_data_2hop/eval_2hop.jsonl \
        --output_dir             ${OUT_DIR} \
        --include_bridge_count \
        --reward_mode            ${REWARD_MODE} \
        --bridge_gate_floor      0.0 \
        --max_steps              ${MAX_STEPS} \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --num_generations        4 \
        --learning_rate          ${LR} \
        --kl_beta                0.02 \
        --temperature            ${RL_TEMP} \
        --top_p                  0.95 \
        --eval_steps             100 \
        --eval_subset_size       200 \
        --save_steps             100 \
        --logging_steps          10 \
        --seed                   42
done


############################################
# Step 3: Eval each RL output (oracle hop count)
############################################
for CKPT in "${CKPTS[@]}"; do
    OUT_DIR=checkpoints/qwen2.5-3b-grpo-from-${CKPT}-v3
    if [ ! -d "${OUT_DIR}" ]; then
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