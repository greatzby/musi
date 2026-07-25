#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-GPU-safe MuSiQue SFT for unsloth/Llama-3.2-3B.

Supported target styles:
  1. answer_only
       Answer: <final answer>

  2. bridge_aware
       Bridge 1: <bridge>
       Bridge 2: <bridge>
       ...
       Answer: <final answer>

Important properties:
  - Supports single-GPU Python launch and multi-GPU torchrun launch.
  - Intended multi-GPU command:
        torchrun --standalone --nproc_per_node=2 sft_llama32_musique.py ...
  - No DataLoader subprocesses by default.
  - Logs every 10 optimizer steps by default.
  - Does not save intermediate checkpoints.
  - Saves only OUTPUT_DIR/final.
  - Final saving is rank-safe under DDP.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import transformers
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from peft import LoraConfig, TaskType, get_peft_model
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

    # Model and paths
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

    # Experiment settings
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

    # Sequence and training
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
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

    # Data processing
    parser.add_argument(
        "--num_proc",
        type=int,
        default=0,
        help=(
            "Number of Dataset.map subprocesses. "
            "0 or 1 means no multiprocessing."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Keep at 0 for maximum cluster stability.",
    )
    parser.add_argument(
        "--dataloader_pin_memory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    # Precision/runtime
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=[
            "sdpa",
            "eager",
            "flash_attention_2",
        ],
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
    parser.add_argument(
        "--optim",
        choices=[
            "adamw_torch",
            "adamw_torch_fused",
        ],
        default="adamw_torch",
    )

    # LoRA; lora_r=0 means full fine-tuning.
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
    )

    # DDP
    parser.add_argument(
        "--ddp_timeout",
        type=int,
        default=1800,
    )
    parser.add_argument(
        "--ddp_bucket_cap_mb",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return get_rank() == 0


def main_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


def load_jsonl(path: Path) -> List[Dict]:
    examples: List[Dict] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                examples.append(json.loads(line))

    return examples


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
    eos_token_id = tokenizer.eos_token_id

    if (
        tokenizer.pad_token_id is not None
        and tokenizer.pad_token_id != eos_token_id
    ):
        return False, 0

    preferred_pad_token = "<|finetune_right_pad_id|>"
    vocabulary = tokenizer.get_vocab()

    if preferred_pad_token in vocabulary:
        tokenizer.pad_token = preferred_pad_token

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
        raise RuntimeError(
            "Failed to create a PAD token."
        )

    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise RuntimeError(
            "PAD and EOS token IDs must be distinct."
        )

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
        return list(
            example.get(
                "gold_context_paragraphs",
                example.get(
                    "context_paragraphs",
                    [],
                ),
            )
        )

    return list(
        example.get(
            "all_context_paragraphs",
            example.get(
                "context_paragraphs",
                [],
            ),
        )
    )


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
    context = (
        "\n\n".join(paragraph_blocks)
        if paragraph_blocks
        else "(No passages provided.)"
    )

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
    full_prompt_ids = render_prompt_ids(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        paragraph_blocks=paragraph_blocks,
        question=question,
    )

    if len(full_prompt_ids) <= maximum_prompt_tokens:
        return full_prompt_ids, False

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
    best_prompt_ids: Optional[List[int]] = None

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

        candidate_ids = render_prompt_ids(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            paragraph_blocks=cropped_blocks,
            question=question,
        )

        if len(candidate_ids) <= maximum_prompt_tokens:
            best_prompt_ids = candidate_ids
            low = cap + 1
        else:
            high = cap - 1

    if best_prompt_ids is not None:
        return best_prompt_ids, True

    minimal_prompt_ids = render_prompt_ids(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        paragraph_blocks=[],
        question=question,
    )

    if len(minimal_prompt_ids) <= maximum_prompt_tokens:
        return minimal_prompt_ids, True

    if maximum_prompt_tokens < 2:
        raise ValueError(
            "maximum_prompt_tokens is too small."
        )

    bos_token_id = tokenizer.bos_token_id

    if (
        bos_token_id is not None
        and minimal_prompt_ids
        and minimal_prompt_ids[0] == bos_token_id
    ):
        minimal_prompt_ids = (
            [minimal_prompt_ids[0]]
            + minimal_prompt_ids[
                -(maximum_prompt_tokens - 1):
            ]
        )
    else:
        minimal_prompt_ids = minimal_prompt_ids[
            -maximum_prompt_tokens:
        ]

    return minimal_prompt_ids, True


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
        example=example,
        target_style=target_style,
    )

    target_ids = tokenizer(
        target_text,
        add_special_tokens=False,
    )["input_ids"]

    target_ids = list(map(int, target_ids))
    target_ids.append(int(stop_token_id))

    maximum_prompt_tokens = (
        max_length - len(target_ids)
    )

    if maximum_prompt_tokens < 32:
        raise ValueError(
            f"Target is too long for max_length={max_length}. "
            f"Target length={len(target_ids)}."
        )

    paragraphs = get_context_paragraphs(
        example=example,
        context_mode=context_mode,
    )

    paragraph_blocks = [
        paragraph_to_text(
            paragraph=paragraph,
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
            target_style=target_style,
            prompt_style=prompt_style,
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
            "Internal tokenization error: "
            f"{len(input_ids)} > {max_length}."
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

        batch_input_ids: List[List[int]] = []
        batch_attention_masks: List[List[int]] = []
        batch_labels: List[List[int]] = []

        for example in batch:
            padding_length = (
                maximum_length
                - len(example["input_ids"])
            )

            batch_input_ids.append(
                example["input_ids"]
                + [self.pad_token_id] * padding_length
            )

            batch_attention_masks.append(
                example["attention_mask"]
                + [0] * padding_length
            )

            batch_labels.append(
                example["labels"]
                + [-100] * padding_length
            )

        return {
            "input_ids": torch.tensor(
                batch_input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                batch_attention_masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                batch_labels,
                dtype=torch.long,
            ),
        }


def build_trainer(
    model,
    tokenizer,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    data_collator: SupervisedDataCollator,
) -> Trainer:
    trainer_arguments = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }

    trainer_signature = inspect.signature(
        Trainer.__init__
    )

    if (
        "processing_class"
        in trainer_signature.parameters
    ):
        trainer_arguments["processing_class"] = tokenizer
    else:
        trainer_arguments["tokenizer"] = tokenizer

    return Trainer(**trainer_arguments)


def make_json_safe(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(item)
            for item in value
        ]

    return value


def main() -> None:
    args = parse_args()

    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()

    if not is_main_process():
        disable_progress_bar()

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    set_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this training script."
        )

    if (
        args.dtype == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "The selected GPU does not support BF16. "
            "Use --dtype fp16."
        )

    train_path = Path(
        args.train_file
    ).expanduser().resolve()

    if not train_path.exists():
        raise FileNotFoundError(train_path)

    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()

    final_dir = output_dir / "final"

    # It is okay for output_dir to exist after a failed run,
    # but do not overwrite a completed final model.
    if final_dir.exists():
        raise FileExistsError(
            f"Final model already exists: {final_dir}\n"
            "Use a different output directory or remove it explicitly."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    effective_global_batch_size = (
        args.per_device_train_batch_size
        * args.gradient_accumulation_steps
        * world_size
    )

    main_print("=" * 88)
    main_print("Llama-3.2-3B MuSiQue SFT")
    main_print(f"Transformers       : {transformers.__version__}")
    main_print(f"PyTorch            : {torch.__version__}")
    main_print(f"Model              : {args.model_name}")
    main_print(f"Target style       : {args.target_style}")
    main_print(f"Context mode       : {args.context_mode}")
    main_print(f"Prompt style       : {args.prompt_style}")
    main_print(f"Training file      : {train_path}")
    main_print(f"Output             : {output_dir}")
    main_print(f"World size         : {world_size}")
    main_print(f"Per-device batch   : {args.per_device_train_batch_size}")
    main_print(f"Gradient accum.    : {args.gradient_accumulation_steps}")
    main_print(f"Global batch       : {effective_global_batch_size}")
    main_print(f"Max steps          : {args.max_steps}")
    main_print(f"Logging steps      : {args.logging_steps}")
    main_print(f"Attention          : {args.attn_implementation}")
    main_print(f"Optimizer          : {args.optim}")
    main_print(f"DataLoader workers : {args.dataloader_num_workers}")
    main_print(f"LoRA rank          : {args.lora_r}")
    main_print("=" * 88)

    print(
        f"[rank={rank} local_rank={local_rank}] "
        f"CUDA device={torch.cuda.current_device()}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )

    installed_chat_template = (
        install_chat_template_if_needed(tokenizer)
    )

    pad_changed, tokens_added = (
        ensure_distinct_pad_token(tokenizer)
    )

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    stop_token_id = get_stop_token_id(tokenizer)

    main_print(
        f"Installed chat template: "
        f"{installed_chat_template}"
    )
    main_print(f"PAD changed            : {pad_changed}")
    main_print(f"PAD token              : {tokenizer.pad_token!r}")
    main_print(f"PAD token id           : {tokenizer.pad_token_id}")
    main_print(f"EOS token              : {tokenizer.eos_token!r}")
    main_print(f"EOS token id           : {tokenizer.eos_token_id}")
    main_print(f"Stop token id          : {stop_token_id}")

    raw_examples = load_jsonl(train_path)
    main_print(f"Loaded {len(raw_examples)} examples")

    raw_dataset = Dataset.from_list(raw_examples)

    # num_proc=None means no Dataset.map subprocess.
    map_num_proc = (
        args.num_proc
        if args.num_proc > 1
        else None
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
        num_proc=map_num_proc,
        desc=(
            "Tokenizing"
            if is_main_process()
            else None
        ),
    )

    encoded_dataset = encoded_dataset.filter(
        lambda example: any(
            label != -100
            for label in example["labels"]
        ),
        num_proc=map_num_proc,
        desc=(
            "Filtering empty targets"
            if is_main_process()
            else None
        ),
    )

    lengths = encoded_dataset["length"]
    truncated_flags = encoded_dataset["truncated"]

    main_print(f"Encoded examples : {len(encoded_dataset)}")
    main_print(f"Minimum length   : {min(lengths)}")
    main_print(f"Maximum length   : {max(lengths)}")
    main_print(
        "Mean length      : "
        f"{sum(lengths) / len(lengths):.1f}"
    )
    main_print(
        "Truncated        : "
        f"{sum(truncated_flags)}/{len(truncated_flags)}"
    )

    model_dtype = torch_dtype_from_name(
        args.dtype
    )

    # Transformers 4.57 uses dtype rather than torch_dtype.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=model_dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )

    original_embedding_size = int(
        model.get_input_embeddings().weight.shape[0]
    )

    if (
        original_embedding_size != len(tokenizer)
        or tokens_added > 0
    ):
        main_print(
            "Resizing token embeddings: "
            f"{original_embedding_size} -> {len(tokenizer)}"
        )

        model.resize_token_embeddings(
            len(tokenizer)
        )

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )
    model.config.use_cache = False

    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = (
            tokenizer.eos_token_id
        )

    if (
        hasattr(model, "generation_config")
        and model.generation_config is not None
    ):
        model.generation_config.pad_token_id = (
            tokenizer.pad_token_id
        )

        if tokenizer.eos_token_id is not None:
            model.generation_config.eos_token_id = (
                tokenizer.eos_token_id
            )

    if args.lora_r > 0:
        modules_to_save: Optional[List[str]] = None

        if tokens_added > 0:
            modules_to_save = [
                "embed_tokens",
                "lm_head",
            ]

        lora_configuration = LoraConfig(
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
            lora_configuration,
        )

        if is_main_process():
            model.print_trainable_parameters()
    else:
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        main_print(
            "Full fine-tuning enabled: "
            f"{trainable_parameters:,} trainable parameters"
        )

    # Required for PEFT + gradient checkpointing when base
    # embeddings are frozen.
    if (
        args.gradient_checkpointing
        and hasattr(
            model,
            "enable_input_require_grads",
        )
    ):
        model.enable_input_require_grads()

    use_bf16 = args.dtype == "bf16"
    use_fp16 = args.dtype == "fp16"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,

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

        optim=args.optim,

        bf16=use_bf16,
        fp16=use_fp16,
        tf32=True,

        gradient_checkpointing=(
            args.gradient_checkpointing
        ),
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False}
            if args.gradient_checkpointing
            else None
        ),

        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,

        # No checkpoint-10/checkpoint-50/etc.
        save_strategy="no",
        save_safetensors=True,

        report_to="none",

        dataloader_num_workers=(
            args.dataloader_num_workers
        ),
        dataloader_pin_memory=(
            args.dataloader_pin_memory
        ),
        dataloader_persistent_workers=False,

        # DDP-safe settings for gradient checkpointing.
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,
        ddp_bucket_cap_mb=args.ddp_bucket_cap_mb,
        ddp_timeout=args.ddp_timeout,

        seed=args.seed,
        data_seed=args.seed,

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

    # Wait until all ranks finish training.
    trainer.accelerator.wait_for_everyone()

    # Trainer.save_model is distributed-aware and only writes
    # from the process that should save.
    trainer.save_model(str(final_dir))

    trainer.accelerator.wait_for_everyone()

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)

        experiment_config = {
            **vars(args),
            "train_file_resolved": str(train_path),
            "output_dir_resolved": str(output_dir),
            "final_dir": str(final_dir),
            "world_size": int(world_size),
            "effective_global_batch_size": int(
                effective_global_batch_size
            ),
            "number_of_examples": int(
                len(encoded_dataset)
            ),
            "minimum_length": int(min(lengths)),
            "maximum_length": int(max(lengths)),
            "mean_length": float(
                sum(lengths) / len(lengths)
            ),
            "number_truncated": int(
                sum(truncated_flags)
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
                installed_chat_template
            ),
            "train_metrics": make_json_safe(
                train_result.metrics
            ),
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
        }

        for directory in [
            output_dir,
            final_dir,
        ]:
            with (
                directory / "experiment_config.json"
            ).open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    experiment_config,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

        print("\n" + "=" * 88)
        print("Training complete")
        print(f"Final model: {final_dir}")
        print(
            "No intermediate checkpoint-* "
            "directories were created."
        )
        print("=" * 88)

    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()