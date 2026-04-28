# sft_train.py
import json
import os
from dataclasses import dataclass
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

# ================== 配置 ==================
MODEL_NAME   = "Qwen/Qwen2.5-3B"
TRAIN_FILE   = "prepared_data/atomic_train.jsonl"
OUTPUT_DIR   = "checkpoints/qwen2.5-3b-atomic-sft"
MAX_LEN      = 1024
SYSTEM_PROMPT = (
    "You answer questions based strictly on the given passage. "
    "Output only a short, direct answer (a few words), with no explanation."
)
# =========================================


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_messages(example: Dict) -> List[Dict]:
    user_content = (
        f"Passage Title: {example['context_title']}\n"
        f"Passage: {example['context']}\n\n"
        f"Question: {example['question']}"
    )
    return [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": example["answer"]},
    ]


def encode(example: Dict, tokenizer, max_length: int = MAX_LEN) -> Dict:
    """对单条样本 tokenize，并把 prompt 部分的 label 设成 -100（不算 loss）"""
    messages = build_messages(example)

    # 完整对话（含 assistant 回答）
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    # 只到 assistant prefix（不含答案），用来定位 mask 边界
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )

    full_ids   = tokenizer(full_text,   add_special_tokens=False, truncation=True,
                           max_length=max_length)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_len = min(len(prompt_ids), len(full_ids))

    labels = [-100] * prompt_len + full_ids[prompt_len:]
    # 兜底对齐
    labels = labels[:len(full_ids)]
    while len(labels) < len(full_ids):
        labels.append(-100)

    return {
        "input_ids":      full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels":         labels,
    }


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            pad = max_len - len(b["input_ids"])
            out["input_ids"].append(     b["input_ids"]      + [self.pad_token_id] * pad)
            out["attention_mask"].append(b["attention_mask"] + [0] * pad)
            out["labels"].append(        b["labels"]         + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def main():
    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- model ---
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # 没装 flash-attn 就改 "sdpa"
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()  # 省显存（虽然这次其实不缺）
    model.config.use_cache = False

    # --- data ---
    raw = load_jsonl(TRAIN_FILE)
    print(f"Loaded {len(raw)} atomic training examples")

    ds = Dataset.from_list(raw)
    ds = ds.map(
        lambda ex: encode(ex, tokenizer),
        remove_columns=ds.column_names,
        num_proc=4,
        desc="tokenizing",
    )
    # 过滤掉异常长的（截断后 label 全 -100 的）
    ds = ds.filter(lambda ex: any(l != -100 for l in ex["labels"]))
    print(f"After filtering: {len(ds)} examples")

    # --- training args ---
    args = TrainingArguments(
        output_dir              = OUTPUT_DIR,
        num_train_epochs        = 3,
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,   # 有效 batch = 8*4*2 = 64
        learning_rate           = 2e-5,
        lr_scheduler_type       = "cosine",
        warmup_ratio            = 0.03,
        weight_decay            = 0.0,
        max_grad_norm           = 1.0,
        bf16                    = True,
        logging_steps           = 20,
        save_strategy           = "epoch",
        save_total_limit        = 3,
        report_to               = "none",  # 想要 wandb 改成 "wandb"
        ddp_find_unused_parameters = False,
        dataloader_num_workers  = 4,
        seed                    = 42,
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = ds,
        data_collator   = PadCollator(pad_token_id=tokenizer.pad_token_id),
        tokenizer       = tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done. Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()