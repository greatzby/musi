#!/bin/bash
# run_step2_mine.sh - 用 checkpoint-50 在全量 train 上挖 hard subset
set -e

# ============ 配置 ============
SFT_CKPT="checkpoints/qwen2.5-3b-2hop-sft-light/checkpoint-50"
INPUT_FILE="prepared_data_2hop/train_2hop.jsonl"
OUTPUT_FILE="prepared_data_2hop/train_2hop_mined.jsonl"

# Mining 策略
NUM_SAMPLES=4              # 与 RL 的 num_generations 一致
TEMPERATURE=0.9            # 与 RL 一致
TOP_P=0.95
REWARD_THRESHOLD=0.85      # mean_reward < 0.85 → hard
TARGET_SIZE=6000           # 期望最终数据集大小
MIN_EASY_RATIO=0.2         # easy 占比下限 20%

BATCH_SIZE=4               # 每次 4 prompts × 4 samples = 16 序列
# ==============================

mkdir -p logs

echo "=========================================================="
echo " STEP 2: Hard Sample Mining"
echo " Model:  ${SFT_CKPT}"
echo " Input:  ${INPUT_FILE}"
echo " Output: ${OUTPUT_FILE}"
echo " Started: $(date)"
echo "=========================================================="

python mine_hard_examples.py \
    --model_dir       ${SFT_CKPT} \
    --input_file      ${INPUT_FILE} \
    --output_file     ${OUTPUT_FILE} \
    --reward_mode     chain_binary \
    --num_samples     ${NUM_SAMPLES} \
    --temperature     ${TEMPERATURE} \
    --top_p           ${TOP_P} \
    --reward_threshold ${REWARD_THRESHOLD} \
    --target_size     ${TARGET_SIZE} \
    --min_easy_ratio  ${MIN_EASY_RATIO} \
    --batch_size      ${BATCH_SIZE} \
    --max_new_tokens  128 \
    --max_input_len   2048 \
    --seed            42 \
    2>&1 | tee logs/mine_train_2hop.log

echo ""
echo "=========================================================="
echo " DONE at $(date)"
echo "=========================================================="
echo ""
echo " Output stats:"
ls -lh ${OUTPUT_FILE}
echo ""
wc -l ${OUTPUT_FILE} prepared_data_2hop/train_2hop_mined_debug_hard.jsonl prepared_data_2hop/train_2hop_mined_debug_easy.jsonl