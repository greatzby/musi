"""
sft_train_2hop.py  (v2: CLI args for max_steps / output_dir / save_strategy)

默认行为和原版完全一致；新增 CLI 覆盖以便训练早期 ckpt-50。
"""

import argparse
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

# ================== 默认配置 ==================
MODEL_NAME    = "Qwen/Qwen2.5-3B"
TRAIN_FILE    = "prepared_data_2hop/train_2hop.jsonl"
OUTPUT_DIR    = "checkpoints/qwen2.5-3b-2hop-sft"
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


def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",       default=MODEL_NAME)
    p.add_argument("--train_file",       default=TRAIN_FILE)
    p.add_argument("--output_dir",       default=OUTPUT_DIR)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--max_steps",        type=int,   default=-1,
                   help=">0 时覆盖 num_train_epochs")
    p.add_argument("--save_strategy",    default="epoch",
                   choices=["epoch", "steps", "no"])
    p.add_argument("--save_steps",       type=int,   default=500)
    p.add_argument("--save_total_limit", type=int,   default=3)
    p.add_argument("--learning_rate",    type=float, default=2e-5)
    p.add_argument("--seed",             type=int,   default=42)
    return p.parse_args()


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_user_content(example):
    parts = []
    for p in example["context_paragraphs"]:
        parts.append(f"Passage Title: {p['title']}\nPassage: {p['text']}")
    ctx = "\n\n".join(parts)
    return f"{ctx}\n\nQuestion: {example['question']}"


def format_assistant_target(example):
    lines = []
    for i, b in enumerate(example["bridges"], start=1):
        lines.append(f"Bridge {i}: {b}")
    lines.append(f"Answer: {example['final_answer']}")
    return "\n".join(lines)


def build_messages(example):
    return [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": format_user_content(example)},
        {"role": "assistant", "content": format_assistant_target(example)},
    ]


def encode(example, tokenizer, max_length=MAX_LEN):
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
    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            pad = max_len - len(b["input_ids"])
            out["input_ids"].append(     b["input_ids"]      + [self.pad_token_id] * pad)
            out["attention_mask"].append(b["attention_mask"] + [0] * pad)
            out["labels"].append(        b["labels"]         + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def main():
    cli = parse_cli()

    tokenizer = AutoTokenizer.from_pretrained(cli.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cli.model_name,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    raw = load_jsonl(cli.train_file)
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

    args = TrainingArguments(
        output_dir              = cli.output_dir,
        num_train_epochs        = cli.num_train_epochs,
        max_steps               = cli.max_steps if cli.max_steps > 0 else -1,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        learning_rate           = cli.learning_rate,
        lr_scheduler_type       = "cosine",
        warmup_ratio            = 0.03,
        weight_decay            = 0.0,
        max_grad_norm           = 1.0,
        bf16                    = True,
        logging_steps           = 20,
        save_strategy           = cli.save_strategy,
        save_steps              = cli.save_steps,
        save_total_limit        = cli.save_total_limit,
        report_to               = "none",
        ddp_find_unused_parameters = False,
        dataloader_num_workers  = 4,
        seed                    = cli.seed,
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = ds,
        data_collator   = PadCollator(pad_token_id=tokenizer.pad_token_id),
        tokenizer       = tokenizer,
    )

    trainer.train()
    trainer.save_model(cli.output_dir)
    tokenizer.save_pretrained(cli.output_dir)
    print(f"Done. Saved to {cli.output_dir}")


if __name__ == "__main__":
    main()