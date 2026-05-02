#!/bin/bash
# run_step1_sft.sh - SFT-light 训练 + 全量评估
set -e

# ============ 配置（按需修改）============
N_GPUS=2                                              # 你有几张卡
EXP_NAME="qwen2.5-3b-2hop-sft-light"
SFT_OUT_DIR="checkpoints/${EXP_NAME}"
EVAL_OUT_DIR="eval_results_${EXP_NAME}"
EVAL_BATCH_SIZE=16                                    # 显存够可以加大到 32
# =========================================

mkdir -p logs

echo "=========================================================="
echo " STEP 1.A: SFT-light training"
echo " Output:  ${SFT_OUT_DIR}"
echo " Started: $(date)"
echo "=========================================================="

torchrun --nproc_per_node=${N_GPUS} sft_train_2hop_light.py 2>&1 | tee logs/sft_${EXP_NAME}.log

echo ""
echo "=========================================================="
echo " STEP 1.B: Full evaluation on all checkpoints"
echo " Output:  ${EVAL_OUT_DIR}"
echo " Started: $(date)"
echo "=========================================================="

python eval_compositional.py \
    --auto_discover ${SFT_OUT_DIR} \
    --out_dir ${EVAL_OUT_DIR} \
    --batch_size ${EVAL_BATCH_SIZE} \
    --splits 2hop 3hop_linear 4hop_linear \
    2>&1 | tee logs/eval_${EXP_NAME}.log

echo ""
echo "=========================================================="
echo " ALL DONE at $(date)"
echo "=========================================================="
echo ""
echo " Cross-checkpoint summary:"
echo "----------------------------------------------------------"
cat ${EVAL_OUT_DIR}/all_checkpoints_summary.json | python -m json.tool
echo ""
echo " Now check the table above and pick the checkpoint with"
echo " 2hop EM in [55, 65] range. Report it back."