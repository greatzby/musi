#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Matched MuSiQue SFT for unsloth/Llama-3.2-3B.

Supported target styles:
  1. answer_only:
       Answer: <final answer>

  2. bridge_aware:
       Bridge 1: <bridge>
       Bridge 2: <bridge>
       ...
       Answer: <final answer>

Important:
  - Both styles use exactly the same training examples.
  - Logging occurs every 10 steps by default.
  - No intermediate checkpoint is saved.
  - Only OUTPUT_DIR/final is saved after training.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


LLAMA3_CHAT_TEMPLATE = r"""
{{- bos_token }}
{%- for message in messages %}
{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
{{- message['content'] | trim }}
{{- '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""


ANSWER_ONLY_ANCHORED_PROMPT = (
    "Answer the question based strictly on the given passages.\n"
    "Return exactly one short line in this format:\n"
    "Answer: <short final answer>\n"
    "Do not provide explanations or intermediate reasoning."
)

BRIDGE_AWARE_ANCHORED_PROMPT = (
    "Answer the multi-step question based strictly on the given passages.\n"
    "First identify the intermediate bridge answers needed to reach the "
    "final answer. List the bridges in dependency order, one per line:\n"
    "Bridge 1: <first intermediate answer>\n"
    "Bridge 2: <second intermediate answer>\n"
    "Continue this numbering if more bridges are needed.\n"
    "On the final line output:\n"
    "Answer: <short final answer>\n"
    "Do not add explanations outside these lines."
)

MINIMAL_PROMPT = (
    "Answer the question based strictly on the given passages. "
    "Keep the response concise."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/Llama-3.2-3B",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="prepared_data_2hop/train_2hop.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--target_style",
        choices=["answer_only", "bridge_aware"],
        required=True,
    )
    parser.add_argument(
        "--context_mode",
        choices=["gold", "all"],
        default="gold",
    )
    parser.add_argument(
        "--prompt_style",
        choices=["anchored", "minimal"],
        default="anchored",
    )

    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["sdpa", "eager", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Set --lora_r 0 for full fine-tuning.
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    data: List[Dict] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16

    if name == "fp16":
        return torch.float16

    return torch.float32


def install_chat_template_if_needed(tokenizer) -> bool:
    if tokenizer.chat_template:
        return False

    tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    return True


def ensure_distinct_pad_token(tokenizer) -> Tuple[bool, int]:
    eos_id = tokenizer.eos_token_id

    if (
        tokenizer.pad_token_id is not None
        and tokenizer.pad_token_id != eos_id
    ):
        return False, 0

    preferred_pad = "<|finetune_right_pad_id|>"
    vocabulary = tokenizer.get_vocab()

    if preferred_pad in vocabulary:
        tokenizer.pad_token = preferred_pad

        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            raise RuntimeError(
                "PAD remains equal to EOS after selecting "
                "<|finetune_right_pad_id|>."
            )

        return True, 0

    number_added = tokenizer.add_special_tokens(
        {"pad_token": "<|pad|>"}
    )

    if tokenizer.pad_token_id is None:
        raise RuntimeError("Could not create a PAD token.")

    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise RuntimeError("PAD and EOS must be distinct.")

    return True, int(number_added)


def get_stop_token_id(tokenizer) -> int:
    vocabulary = tokenizer.get_vocab()

    if "<|eot_id|>" in vocabulary:
        return int(vocabulary["<|eot_id|>"])

    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)

    raise RuntimeError(
        "Tokenizer has neither <|eot_id|> nor eos_token_id."
    )


def get_system_prompt(
    target_style: str,
    prompt_style: str,
) -> str:
    if prompt_style == "minimal":
        return MINIMAL_PROMPT

    if target_style == "answer_only":
        return ANSWER_ONLY_ANCHORED_PROMPT

    return BRIDGE_AWARE_ANCHORED_PROMPT


def get_context_paragraphs(
    example: Dict,
    context_mode: str,
) -> List[Dict]:
    if context_mode == "gold":
        paragraphs = example.get(
            "gold_context_paragraphs",
            example.get("context_paragraphs", []),
        )
    else:
        paragraphs = example.get(
            "all_context_paragraphs",
            example.get("context_paragraphs", []),
        )

    return list(paragraphs)


def paragraph_to_text(
    paragraph: Dict,
    paragraph_number: int,
) -> str:
    title = str(paragraph.get("title", ""))
    text = str(
        paragraph.get(
            "text",
            paragraph.get("paragraph_text", ""),
        )
    )

    return (
        f"Passage {paragraph_number}\n"
        f"Passage Title: {title}\n"
        f"Passage: {text}"
    )


def format_user_content(
    paragraph_blocks: List[str],
    question: str,
) -> str:
    if paragraph_blocks:
        context = "\n\n".join(paragraph_blocks)
    else:
        context = "(No passages provided.)"

    return f"{context}\n\nQuestion: {question}"


def format_target(
    example: Dict,
    target_style: str,
) -> str:
    final_answer = str(example["final_answer"])

    if target_style == "answer_only":
        return f"Answer: {final_answer}"

    lines = [
        f"Bridge {index}: {bridge}"
        for index, bridge in enumerate(
            example.get("bridges", []),
            start=1,
        )
    ]

    lines.append(f"Answer: {final_answer}")
    return "\n".join(lines)


def render_prompt_ids(
    tokenizer,
    system_prompt: str,
    paragraph_blocks: List[str],
    question: str,
) -> List[int]:
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": format_user_content(
                paragraph_blocks,
                question,
            ),
        },
    ]

    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    return list(map(int, token_ids))


def fit_prompt_to_budget(
    tokenizer,
    system_prompt: str,
    paragraph_blocks: List[str],
    question: str,
    maximum_prompt_tokens: int,
) -> Tuple[List[int], bool]:
    full_prompt = render_prompt_ids(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        paragraph_blocks=paragraph_blocks,
        question=question,
    )

    if len(full_prompt) <= maximum_prompt_tokens:
        return full_prompt, False

    # Preserve every paragraph by applying approximately the same
    # token cap to every paragraph block.
    block_token_ids = [
        tokenizer(
            block,
            add_special_tokens=False,
        )["input_ids"]
        for block in paragraph_blocks
    ]

    maximum_block_length = max(
        [len(ids) for ids in block_token_ids] or [0]
    )

    low = 0
    high = maximum_block_length
    best_prompt: Optional[List[int]] = None

    while low <= high:
        cap = (low + high) // 2

        cropped_blocks: List[str] = []

        for block_ids in block_token_ids:
            cropped_ids = block_ids[:cap]

            if not cropped_ids:
                continue

            cropped_blocks.append(
                tokenizer.decode(
                    cropped_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )

        candidate = render_prompt_ids(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            paragraph_blocks=cropped_blocks,
            question=question,
        )

        if len(candidate) <= maximum_prompt_tokens:
            best_prompt = candidate
            low = cap + 1
        else:
            high = cap - 1

    if best_prompt is not None:
        return best_prompt, True

    # Extreme fallback: preserve BOS and the end containing the question.
    minimal_prompt = render_prompt_ids(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        paragraph_blocks=[],
        question=question,
    )

    if len(minimal_prompt) <= maximum_prompt_tokens:
        return minimal_prompt, True

    if maximum_prompt_tokens < 2:
        raise ValueError(
            "maximum_prompt_tokens is too small."
        )

    if (
        tokenizer.bos_token_id is not None
        and minimal_prompt[0] == tokenizer.bos_token_id
    ):
        minimal_prompt = (
            [minimal_prompt[0]]
            + minimal_prompt[
                -(maximum_prompt_tokens - 1):
            ]
        )
    else:
        minimal_prompt = minimal_prompt[
            -maximum_prompt_tokens:
        ]

    return minimal_prompt, True


def encode_example(
    example: Dict,
    tokenizer,
    target_style: str,
    context_mode: str,
    prompt_style: str,
    max_length: int,
    stop_token_id: int,
) -> Dict:
    target_text = format_target(
        example,
        target_style,
    )

    target_ids = tokenizer(
        target_text,
        add_special_tokens=False,
    )["input_ids"]

    target_ids = list(map(int, target_ids))
    target_ids.append(int(stop_token_id))

    maximum_prompt_tokens = max_length - len(target_ids)

    if maximum_prompt_tokens < 32:
        raise ValueError(
            f"Target is too long for max_length={max_length}: "
            f"target tokens={len(target_ids)}"
        )

    paragraphs = get_context_paragraphs(
        example,
        context_mode,
    )

    paragraph_blocks = [
        paragraph_to_text(
            paragraph,
            paragraph_number=index,
        )
        for index, paragraph in enumerate(
            paragraphs,
            start=1,
        )
    ]

    prompt_ids, was_truncated = fit_prompt_to_budget(
        tokenizer=tokenizer,
        system_prompt=get_system_prompt(
            target_style,
            prompt_style,
        ),
        paragraph_blocks=paragraph_blocks,
        question=str(example["question"]),
        maximum_prompt_tokens=maximum_prompt_tokens,
    )

    input_ids = prompt_ids + target_ids
    labels = (
        [-100] * len(prompt_ids)
        + target_ids
    )

    if len(input_ids) > max_length:
        raise RuntimeError(
            "Internal truncation error: encoded sequence "
            f"has {len(input_ids)} tokens, max={max_length}."
        )

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "length": len(input_ids),
        "truncated": int(was_truncated),
    }


@dataclass
class SupervisedDataCollator:
    pad_token_id: int

    def __call__(
        self,
        batch: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        maximum_length = max(
            len(example["input_ids"])
            for example in batch
        )

        input_ids: List[List[int]] = []
        attention_masks: List[List[int]] = []
        labels: List[List[int]] = []

        for example in batch:
            padding_length = (
                maximum_length
                - len(example["input_ids"])
            )

            input_ids.append(
                example["input_ids"]
                + [self.pad_token_id] * padding_length
            )

            attention_masks.append(
                example["attention_mask"]
                + [0] * padding_length
            )

            labels.append(
                example["labels"]
                + [-100] * padding_length
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }


def build_trainer(
    model,
    tokenizer,
    training_args,
    train_dataset,
    data_collator,
) -> Trainer:
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }

    signature = inspect.signature(
        Trainer.__init__
    )

    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer

    return Trainer(**kwargs)


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if args.dtype == "bf16":
        if (
            torch.cuda.is_available()
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "The selected GPU does not support BF16. "
                "Use --dtype fp16."
            )

    train_path = Path(
        args.train_file
    ).expanduser().resolve()

    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    final_dir = output_dir / "final"

    if final_dir.exists():
        raise FileExistsError(
            f"Final model directory already exists: {final_dir}\n"
            "Use a new --output_dir or remove the old directory."
        )

    print("=" * 80)
    print("Llama-3.2-3B MuSiQue SFT")
    print(f"Model          : {args.model_name}")
    print(f"Target style   : {args.target_style}")
    print(f"Context mode   : {args.context_mode}")
    print(f"Prompt style   : {args.prompt_style}")
    print(f"Training file  : {train_path}")
    print(f"Output         : {output_dir}")
    print(f"Max steps      : {args.max_steps}")
    print(f"Logging steps  : {args.logging_steps}")
    print(f"LoRA rank      : {args.lora_r}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )

    installed_template = (
        install_chat_template_if_needed(tokenizer)
    )

    pad_changed, tokens_added = ensure_distinct_pad_token(
        tokenizer
    )

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    stop_token_id = get_stop_token_id(tokenizer)

    print(f"Installed chat template: {installed_template}")
    print(f"PAD changed            : {pad_changed}")
    print(f"PAD token              : {tokenizer.pad_token!r}")
    print(f"PAD token id           : {tokenizer.pad_token_id}")
    print(f"EOS token              : {tokenizer.eos_token!r}")
    print(f"EOS token id           : {tokenizer.eos_token_id}")
    print(f"Stop token id          : {stop_token_id}")

    raw_examples = load_jsonl(train_path)
    print(f"Loaded {len(raw_examples)} examples")

    raw_dataset = Dataset.from_list(
        raw_examples
    )

    encoded_dataset = raw_dataset.map(
        lambda example: encode_example(
            example=example,
            tokenizer=tokenizer,
            target_style=args.target_style,
            context_mode=args.context_mode,
            prompt_style=args.prompt_style,
            max_length=args.max_length,
            stop_token_id=stop_token_id,
        ),
        remove_columns=raw_dataset.column_names,
        num_proc=args.num_proc,
        desc="Tokenizing",
    )

    encoded_dataset = encoded_dataset.filter(
        lambda example: any(
            label != -100
            for label in example["labels"]
        ),
        num_proc=args.num_proc,
        desc="Filtering empty targets",
    )

    lengths = encoded_dataset["length"]
    truncated = encoded_dataset["truncated"]

    print(f"Encoded examples : {len(encoded_dataset)}")
    print(f"Minimum length   : {min(lengths)}")
    print(f"Maximum length   : {max(lengths)}")
    print(
        "Mean length      : "
        f"{sum(lengths) / len(lengths):.1f}"
    )
    print(
        "Truncated        : "
        f"{sum(truncated)}/{len(truncated)}"
    )

    model_dtype = torch_dtype_from_name(
        args.dtype
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=model_dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )

    embedding_size = int(
        model.get_input_embeddings().weight.shape[0]
    )

    if (
        embedding_size != len(tokenizer)
        or tokens_added > 0
    ):
        print(
            f"Resizing token embeddings: "
            f"{embedding_size} -> {len(tokenizer)}"
        )
        model.resize_token_embeddings(
            len(tokenizer)
        )

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = (
            tokenizer.eos_token_id
        )

    model.config.use_cache = False

    if args.lora_r > 0:
        modules_to_save: Optional[List[str]] = None

        if tokens_added > 0:
            modules_to_save = [
                "embed_tokens",
                "lm_head",
            ]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            modules_to_save=modules_to_save,
        )

        model = get_peft_model(
            model,
            lora_config,
        )
        model.print_trainable_parameters()
    else:
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        print(
            f"Full fine-tuning enabled: "
            f"{trainable:,} trainable parameters"
        )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            }
        )

        if hasattr(
            model,
            "enable_input_require_grads",
        ):
            model.enable_input_require_grads()

    use_bf16 = args.dtype == "bf16"
    use_fp16 = args.dtype == "fp16"

    training_args = TrainingArguments(
        output_dir=str(output_dir),

        max_steps=args.max_steps,
        per_device_train_batch_size=(
            args.per_device_train_batch_size
        ),
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),

        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,

        bf16=use_bf16,
        fp16=use_fp16,
        tf32=True,

        gradient_checkpointing=(
            args.gradient_checkpointing
        ),

        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,

        # No intermediate checkpoints.
        save_strategy="no",

        report_to="none",
        dataloader_num_workers=(
            args.dataloader_num_workers
        ),
        dataloader_pin_memory=True,

        seed=args.seed,
        data_seed=args.seed,

        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=encoded_dataset,
        data_collator=SupervisedDataCollator(
            pad_token_id=int(
                tokenizer.pad_token_id
            )
        ),
    )

    train_result = trainer.train()

    # Save only the final model/adapter.
    final_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)

    experiment_config = {
        **vars(args),
        "train_file_resolved": str(train_path),
        "final_dir": str(final_dir),
        "number_of_examples": len(
            encoded_dataset
        ),
        "minimum_length": int(min(lengths)),
        "maximum_length": int(max(lengths)),
        "mean_length": float(
            sum(lengths) / len(lengths)
        ),
        "number_truncated": int(
            sum(truncated)
        ),
        "effective_batch_size_per_process": int(
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
        ),
        "pad_token_id": int(
            tokenizer.pad_token_id
        ),
        "eos_token_id": (
            int(tokenizer.eos_token_id)
            if tokenizer.eos_token_id is not None
            else None
        ),
        "stop_token_id": int(stop_token_id),
        "installed_chat_template": bool(
            installed_template
        ),
        "train_metrics": {
            key: float(value)
            if isinstance(value, (int, float))
            else value
            for key, value in train_result.metrics.items()
        },
    }

    for directory in [output_dir, final_dir]:
        with (
            directory / "experiment_config.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(
                experiment_config,
                file,
                indent=2,
                ensure_ascii=False,
            )

    trainer.save_state()

    print("\n" + "=" * 80)
    print("Training complete")
    print(f"Final model: {final_dir}")
    print(
        "No intermediate checkpoint-* directories "
        "were created."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()