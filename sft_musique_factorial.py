#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
from transformers.trainer_utils import get_last_checkpoint

from musique_common import (
    build_full_messages,
    build_prompt_messages,
    build_system_prompt,
    effective_training_epochs,
    ensure_pad_token,
    install_chat_template_if_needed,
    load_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--base_key",
        type=str,
        required=True,
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
        choices=[
            "answer_only",
            "bridge_aware",
        ],
        required=True,
    )
    parser.add_argument(
        "--prompt_style",
        choices=[
            "canonical",
            "anchored",
        ],
        required=True,
    )
    parser.add_argument(
        "--context_mode",
        choices=[
            "gold",
            "all",
        ],
        default="gold",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=2000,
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
        "--warmup_steps",
        type=int,
        default=200,
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
        "--save_steps",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=8,
    )

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

    parser.add_argument(
        "--dtype",
        choices=[
            "bf16",
            "fp16",
            "fp32",
        ],
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
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--num_proc",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="auto",
        help=(
            "'auto', 'none', or a checkpoint directory."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
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


def torch_dtype_from_name(
    name: str,
) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16

    if name == "fp16":
        return torch.float16

    return torch.float32


def encode_example(
    example: Dict,
    tokenizer,
    target_style: str,
    prompt_style: str,
    context_mode: str,
    max_length: int,
) -> Dict:
    prompt_messages = build_prompt_messages(
        example=example,
        context_mode=context_mode,
        prompt_style=prompt_style,
    )

    full_messages = build_full_messages(
        example=example,
        context_mode=context_mode,
        prompt_style=prompt_style,
        target_style=target_style,
    )

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    prompt_ids = list(map(int, prompt_ids))
    full_ids = list(map(int, full_ids))

    if (
        len(full_ids) < len(prompt_ids)
        or full_ids[:len(prompt_ids)] != prompt_ids
    ):
        example_id = example.get(
            "id",
            "unknown",
        )

        raise RuntimeError(
            "Chat-template prefix mismatch for "
            f"example {example_id}. The tokenized "
            "generation prompt is not a prefix of the "
            "complete training conversation."
        )

    if len(full_ids) > max_length:
        example_id = example.get(
            "id",
            "unknown",
        )

        raise ValueError(
            f"Example {example_id} has "
            f"{len(full_ids)} tokens, exceeding "
            f"max_length={max_length}. Increase "
            "--max_length rather than silently "
            "truncating the question or answer."
        )

    labels = (
        [-100] * len(prompt_ids)
        + full_ids[len(prompt_ids):]
    )

    if not any(
        label != -100
        for label in labels
    ):
        raise RuntimeError(
            "Encoded example has no supervised tokens."
        )

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "length": len(full_ids),
        "prompt_length": len(prompt_ids),
        "target_length": (
            len(full_ids) - len(prompt_ids)
        ),
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
                + [self.pad_token_id]
                * padding_length
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
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }

    signature = inspect.signature(
        Trainer.__init__
    )

    if (
        "processing_class"
        in signature.parameters
    ):
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer

    return Trainer(**kwargs)


def make_json_safe(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()

        return value.detach().cpu().tolist()

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


def resolve_resume_checkpoint(
    output_dir: Path,
    requested: str,
) -> Optional[str]:
    normalized = requested.strip().lower()

    if normalized in {
        "",
        "none",
        "false",
        "no",
    }:
        return None

    if normalized == "auto":
        checkpoint = get_last_checkpoint(
            str(output_dir)
        )

        return checkpoint

    checkpoint_path = Path(
        requested
    ).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    return str(checkpoint_path)


def write_json(
    path: Path,
    value: Dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            value,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    args = parse_args()

    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()

    if not is_main_process():
        disable_progress_bar()

    random.seed(args.seed + rank)
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    torch.cuda.set_device(local_rank)

    if (
        args.dtype == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "The selected GPU does not support BF16."
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision(
            "high"
        )
    except Exception:
        pass

    train_path = Path(
        args.train_file
    ).expanduser().resolve()

    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()

    final_dir = output_dir / "final"

    if not train_path.exists():
        raise FileNotFoundError(train_path)

    if final_dir.exists():
        main_print(
            f"Completed model already exists: "
            f"{final_dir}\nSkipping training."
        )
        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    global_batch_size = (
        args.per_device_train_batch_size
        * args.gradient_accumulation_steps
        * world_size
    )

    main_print("=" * 92)
    main_print("MuSiQue controlled SFT")
    main_print("=" * 92)
    main_print(f"Base key            : {args.base_key}")
    main_print(f"Model               : {args.model_name}")
    main_print(f"Target style        : {args.target_style}")
    main_print(f"Prompt style        : {args.prompt_style}")
    main_print(f"Context mode        : {args.context_mode}")
    main_print(f"Training file       : {train_path}")
    main_print(f"Output              : {output_dir}")
    main_print(f"World size          : {world_size}")
    main_print(
        f"Per-device batch    : "
        f"{args.per_device_train_batch_size}"
    )
    main_print(
        f"Gradient accumulation: "
        f"{args.gradient_accumulation_steps}"
    )
    main_print(
        f"Global batch        : "
        f"{global_batch_size}"
    )
    main_print(
        f"Maximum steps       : "
        f"{args.max_steps}"
    )
    main_print(
        f"Learning rate       : "
        f"{args.learning_rate}"
    )
    main_print(
        f"Warmup steps        : "
        f"{args.warmup_steps}"
    )
    main_print("=" * 92)

    print(
        f"[rank={rank} local_rank={local_rank}] "
        f"CUDA device={torch.cuda.current_device()}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    chat_template_status = (
        install_chat_template_if_needed(
            tokenizer
        )
    )

    number_added_tokens = ensure_pad_token(
        tokenizer
    )

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    main_print(
        f"Chat template       : "
        f"{chat_template_status}"
    )
    main_print(
        f"PAD token           : "
        f"{tokenizer.pad_token!r}"
    )
    main_print(
        f"PAD token ID        : "
        f"{tokenizer.pad_token_id}"
    )
    main_print(
        f"EOS token           : "
        f"{tokenizer.eos_token!r}"
    )
    main_print(
        f"EOS token ID        : "
        f"{tokenizer.eos_token_id}"
    )
    main_print(
        f"New vocabulary tokens: "
        f"{number_added_tokens}"
    )

    raw_examples = load_jsonl(
        train_path
    )

    main_print(
        f"Loaded examples     : "
        f"{len(raw_examples)}"
    )

    bridge_distribution = Counter(
        len(example.get("bridges", []))
        for example in raw_examples
    )

    main_print(
        "Bridge-count distribution: "
        f"{dict(sorted(bridge_distribution.items()))}"
    )

    raw_dataset = Dataset.from_list(
        raw_examples
    )

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
            prompt_style=args.prompt_style,
            context_mode=args.context_mode,
            max_length=args.max_length,
        ),
        remove_columns=raw_dataset.column_names,
        num_proc=map_num_proc,
        desc=(
            "Tokenizing"
            if is_main_process()
            else None
        ),
    )

    lengths = encoded_dataset["length"]
    prompt_lengths = (
        encoded_dataset["prompt_length"]
    )
    target_lengths = (
        encoded_dataset["target_length"]
    )

    estimated_epochs = effective_training_epochs(
        max_steps=args.max_steps,
        global_batch_size=global_batch_size,
        number_examples=len(encoded_dataset),
    )

    main_print(
        f"Encoded examples    : "
        f"{len(encoded_dataset)}"
    )
    main_print(
        f"Minimum length      : "
        f"{min(lengths)}"
    )
    main_print(
        f"Maximum length      : "
        f"{max(lengths)}"
    )
    main_print(
        "Mean total length   : "
        f"{sum(lengths) / len(lengths):.1f}"
    )
    main_print(
        "Mean prompt length  : "
        f"{sum(prompt_lengths) / len(prompt_lengths):.1f}"
    )
    main_print(
        "Mean target length  : "
        f"{sum(target_lengths) / len(target_lengths):.1f}"
    )
    main_print(
        "Approximate epochs  : "
        f"{estimated_epochs:.3f}"
    )

    first_example = encoded_dataset[0]

    first_prompt_ids = first_example[
        "input_ids"
    ][:first_example["prompt_length"]]

    first_target_ids = first_example[
        "input_ids"
    ][first_example["prompt_length"]:]

    decoded_prompt = tokenizer.decode(
        first_prompt_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    decoded_target = tokenizer.decode(
        first_target_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    main_print(
        "\nFirst prompt ending:\n"
        + decoded_prompt[-1800:]
    )
    main_print(
        "\nFirst supervised target:\n"
        + decoded_target
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch_dtype_from_name(
            args.dtype
        ),
        attn_implementation=(
            args.attn_implementation
        ),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    )

    original_vocabulary_size = int(
        model.get_input_embeddings()
        .weight.shape[0]
    )

    if (
        original_vocabulary_size
        != len(tokenizer)
    ):
        main_print(
            "Resizing token embeddings: "
            f"{original_vocabulary_size} "
            f"-> {len(tokenizer)}"
        )

        model.resize_token_embeddings(
            len(tokenizer)
        )

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )
    model.config.use_cache = False

    modules_to_save: Optional[List[str]] = None

    if number_added_tokens > 0:
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

    if (
        args.gradient_checkpointing
        and hasattr(
            model,
            "enable_input_require_grads",
        )
    ):
        model.enable_input_require_grads()

    if is_main_process():
        model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        run_name=output_dir.name,

        max_steps=args.max_steps,

        per_device_train_batch_size=(
            args.per_device_train_batch_size
        ),
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),

        learning_rate=args.learning_rate,
        lr_scheduler_type=(
            args.lr_scheduler_type
        ),
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        optim="adamw_torch",

        bf16=args.dtype == "bf16",
        fp16=args.dtype == "fp16",
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
        log_on_each_node=False,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=(
            args.save_total_limit
        ),
        save_safetensors=True,
        save_on_each_node=False,

        report_to="none",

        dataloader_num_workers=(
            args.dataloader_num_workers
        ),
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,

        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,
        ddp_bucket_cap_mb=(
            args.ddp_bucket_cap_mb
        ),
        ddp_timeout=args.ddp_timeout,

        seed=args.seed,
        data_seed=args.seed,

        remove_unused_columns=False,
        disable_tqdm=not is_main_process(),
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

    resume_checkpoint = (
        resolve_resume_checkpoint(
            output_dir=output_dir,
            requested=(
                args.resume_from_checkpoint
            ),
        )
    )

    main_print(
        f"Resume checkpoint   : "
        f"{resume_checkpoint}"
    )

    experiment_config = {
        **vars(args),
        "run_name": output_dir.name,
        "train_file_resolved": str(
            train_path
        ),
        "output_dir_resolved": str(
            output_dir
        ),
        "final_dir": str(final_dir),
        "world_size": world_size,
        "effective_global_batch_size": (
            global_batch_size
        ),
        "number_of_examples": len(
            encoded_dataset
        ),
        "approximate_epochs": (
            estimated_epochs
        ),
        "minimum_length": min(lengths),
        "maximum_length": max(lengths),
        "mean_length": (
            sum(lengths) / len(lengths)
        ),
        "mean_prompt_length": (
            sum(prompt_lengths)
            / len(prompt_lengths)
        ),
        "mean_target_length": (
            sum(target_lengths)
            / len(target_lengths)
        ),
        "bridge_count_distribution": {
            str(key): int(value)
            for key, value
            in bridge_distribution.items()
        },
        "chat_template_status": (
            chat_template_status
        ),
        "pad_token_id": (
            tokenizer.pad_token_id
        ),
        "eos_token_id": (
            tokenizer.eos_token_id
        ),
        "system_prompt": (
            build_system_prompt(
                args.prompt_style
            )
        ),
        "prompt_protocol_version": (
            "goal_anchor_factorial_v1"
        ),
        "versions": {
            "torch": torch.__version__,
            "transformers": (
                transformers.__version__
            ),
        },
    }

    if is_main_process():
        write_json(
            output_dir
            / "experiment_config.json",
            experiment_config,
        )

    train_result = trainer.train(
        resume_from_checkpoint=(
            resume_checkpoint
        )
    )

    trainer.accelerator.wait_for_everyone()

    trainer.save_model(
        str(final_dir)
    )

    trainer.accelerator.wait_for_everyone()

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(
            final_dir
        )

        experiment_config[
            "train_metrics"
        ] = make_json_safe(
            train_result.metrics
        )

        write_json(
            output_dir
            / "experiment_config.json",
            experiment_config,
        )

        write_json(
            final_dir
            / "experiment_config.json",
            experiment_config,
        )

        print("\n" + "=" * 92)
        print("Training complete")
        print(f"Final model: {final_dir}")
        print("=" * 92)

    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()