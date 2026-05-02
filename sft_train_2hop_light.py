"""
sft_train_2hop_light.py

故意欠训的 SFT, 用于给 GRPO 留出提升空间。
关键差别 vs sft_train_2hop.py:
  - max_steps=150 (而不是 3 epoch / 675 step)
  - save_steps=25 (每 25 步存一次, 共 6 个 checkpoint)
  - warmup_ratio=0.05 (因为总步数变少, 略增 warmup 比例)
"""

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
MODEL_NAME    = "Qwen/Qwen2.5-3B"
TRAIN_FILE    = "prepared_data_2hop/train_2hop.jsonl"
OUTPUT_DIR    = "checkpoints/qwen2.5-3b-2hop-sft-light"
MAX_LEN       = 2048

SYSTEM_PROMPT = (
    "You answer multi-step questions based strictly on the given passages.\n"
    "For each question:\n"
    "1. Identify each intermediate entity needed to reach the final answer (a \"bridge\"). "
    "List them in order, one per line, as 'Bridge 1: ...', 'Bridge 2: ...', etc.\n"
    "2. On the final line, output 'Answer: <short final answer>'.\n"
    "Use only information from the passages. Keep answers short (a few words)."
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


def format_user_content(example: Dict) -> str:
    parts = []
    for p in example["context_paragraphs"]:
        parts.append(f"Passage Title: {p['title']}\nPassage: {p['text']}")
    ctx = "\n\n".join(parts)
    return f"{ctx}\n\nQuestion: {example['question']}"


def format_assistant_target(example: Dict) -> str:
    lines = []
    for i, b in enumerate(example["bridges"], start=1):
        lines.append(f"Bridge {i}: {b}")
    lines.append(f"Answer: {example['final_answer']}")
    return "\n".join(lines)


def build_messages(example: Dict) -> List[Dict]:
    return [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": format_user_content(example)},
        {"role": "assistant", "content": format_assistant_target(example)},
    ]


def encode(example: Dict, tokenizer, max_length: int = MAX_LEN) -> Dict:
    messages = build_messages(example)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_ids   = tokenizer(full_text,   add_special_tokens=False, truncation=True,
                           max_length=max_length)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    raw = load_jsonl(TRAIN_FILE)
    print(f"Loaded {len(raw)} 2-hop training examples")

    ds = Dataset.from_list(raw)
    ds = ds.map(
        lambda ex: encode(ex, tokenizer),
        remove_columns=ds.column_names,
        num_proc=4,
        desc="tokenizing",
    )
    ds = ds.filter(lambda ex: any(l != -100 for l in ex["labels"]))
    print(f"After filtering: {len(ds)} examples")

    lens = [len(ex["input_ids"]) for ex in ds.select(range(min(100, len(ds))))]
    print(f"Sample input length: min={min(lens)}, max={max(lens)}, mean={sum(lens)/len(lens):.0f}")

    # ============ 关键改动 ============
    args = TrainingArguments(
        output_dir              = OUTPUT_DIR,
        max_steps               = 150,        # 故意欠训
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        learning_rate           = 2e-5,
        lr_scheduler_type       = "cosine",
        warmup_ratio            = 0.05,
        weight_decay            = 0.0,
        max_grad_norm           = 1.0,
        bf16                    = True,
        logging_steps           = 10,
        save_strategy           = "steps",
        save_steps              = 25,         # 25, 50, 75, 100, 125, 150
        save_total_limit        = 10,         # 保留全部
        report_to               = "none",
        ddp_find_unused_parameters = False,
        dataloader_num_workers  = 4,
        seed                    = 42,
    )
    # ==================================

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