#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="unsloth/Llama-3.2-3B"
DATA_DIR="prepared_data_2hop"

ANSWER_DIR="checkpoints/llama32-3b-musique-answer-only-gold"
BRIDGE_DIR="checkpoints/llama32-3b-musique-bridge-aware-gold"

echo "============================================================"
echo "1. Preparing MuSiQue"
echo "============================================================"

python prepare_musique_2hop.py \
  --musique_dir ./data \
  --output_dir "${DATA_DIR}" \
  --seed 42

echo "============================================================"
echo "2. Training answer-only SFT"
echo "============================================================"

python sft_llama32_musique.py \
  --model_name "${MODEL}" \
  --train_file "${DATA_DIR}/train_2hop.jsonl" \
  --output_dir "${ANSWER_DIR}" \
  --target_style answer_only \
  --context_mode gold \
  --prompt_style anchored \
  --max_length 2048 \
  --max_steps 200 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --dtype bf16 \
  --attn_implementation sdpa \
  --gradient_checkpointing \
  --seed 42

echo "============================================================"
echo "3. Training bridge-aware SFT"
echo "============================================================"

python sft_llama32_musique.py \
  --model_name "${MODEL}" \
  --train_file "${DATA_DIR}/train_2hop.jsonl" \
  --output_dir "${BRIDGE_DIR}" \
  --target_style bridge_aware \
  --context_mode gold \
  --prompt_style anchored \
  --max_length 2048 \
  --max_steps 200 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --dtype bf16 \
  --attn_implementation sdpa \
  --gradient_checkpointing \
  --seed 42

echo "============================================================"
echo "4. Main evaluation without oracle bridge count"
echo "============================================================"

python eval_llama32_musique.py \
  --model_dirs \
    "${ANSWER_DIR}/final" \
    "${BRIDGE_DIR}/final" \
  --eval_dir "${DATA_DIR}" \
  --out_dir eval_results/llama32-main \
  --splits 2hop 3hop_linear 4hop_linear \
  --prompt_variant standard \
  --context_mode auto \
  --batch_size 8 \
  --max_input_length 4096 \
  --max_new_tokens 128 \
  --dtype bf16 \
  --attn_implementation sdpa

echo "============================================================"
echo "5. Bridge-aware minimal-prompt diagnostic"
echo "============================================================"

python eval_llama32_musique.py \
  --model_dirs "${BRIDGE_DIR}/final" \
  --eval_dir "${DATA_DIR}" \
  --out_dir eval_results/llama32-prompt-ablation \
  --splits 2hop 3hop_linear 4hop_linear \
  --prompt_variant minimal \
  --context_mode auto \
  --batch_size 8 \
  --max_input_length 4096 \
  --max_new_tokens 128 \
  --dtype bf16 \
  --attn_implementation sdpa

echo "============================================================"
echo "All Llama experiments completed"
echo "============================================================"