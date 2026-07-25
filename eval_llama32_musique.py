#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate answer-only and bridge-aware Llama SFT models on MuSiQue.

Prompt variants:
  standard:
    Explicit output schema, but no oracle bridge count.

  minimal:
    Minimal instruction without explicit schema.

  bridge_count:
    Explicit schema plus the gold number of bridges.
    This is an oracle diagnostic and must not be used as the main result.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


EVAL_FILES = {
    "2hop": "eval_2hop.jsonl",
    "3hop_linear": "eval_3hop_linear.jsonl",
    "4hop_linear": "eval_4hop_linear.jsonl",
}


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


RE_BRIDGE = re.compile(
    r"^\s*Bridge\s*\d+\s*[:：]\s*(.+?)\s*$",
    re.IGNORECASE,
)
RE_ANSWER = re.compile(
    r"^\s*Answer\s*[:：]\s*(.+?)\s*$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dirs",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="prepared_data_2hop",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="eval_results",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=[
            "2hop",
            "3hop_linear",
            "4hop_linear",
        ],
    )

    parser.add_argument(
        "--target_style",
        choices=["auto", "answer_only", "bridge_aware"],
        default="auto",
    )
    parser.add_argument(
        "--context_mode",
        choices=["auto", "gold", "all"],
        default="auto",
    )
    parser.add_argument(
        "--prompt_variant",
        choices=["standard", "minimal", "bridge_count"],
        default="standard",
    )

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
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
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser.parse_args()


def torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16

    if name == "fp16":
        return torch.float16

    return torch.float32


def load_jsonl(path: Path) -> List[Dict]:
    data: List[Dict] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def load_experiment_config(
    model_dir: Path,
) -> Dict:
    candidates = [
        model_dir / "experiment_config.json",
        model_dir.parent / "experiment_config.json",
    ]

    for path in candidates:
        if path.exists():
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                return json.load(file)

    return {}


def infer_run_name(model_dir: Path) -> str:
    if model_dir.name == "final":
        return model_dir.parent.name

    return model_dir.name


def normalize_answer(text: Optional[str]) -> str:
    if text is None:
        return ""

    def remove_articles(value: str) -> str:
        return re.sub(
            r"\b(a|an|the)\b",
            " ",
            value,
        )

    def remove_punctuation(value: str) -> str:
        punctuation = set(string.punctuation)
        return "".join(
            character
            for character in value
            if character not in punctuation
        )

    def normalize_whitespace(value: str) -> str:
        return " ".join(value.split())

    return normalize_whitespace(
        remove_articles(
            remove_punctuation(
                text.lower()
            )
        )
    )


def exact_match(
    prediction: str,
    gold_answers: List[str],
) -> float:
    normalized_prediction = normalize_answer(
        prediction
    )

    return float(
        any(
            normalized_prediction
            == normalize_answer(gold)
            for gold in gold_answers
            if gold
        )
    )


def pair_f1(
    prediction: str,
    gold: str,
) -> float:
    prediction_tokens = normalize_answer(
        prediction
    ).split()

    gold_tokens = normalize_answer(
        gold
    ).split()

    if not prediction_tokens or not gold_tokens:
        return float(
            prediction_tokens == gold_tokens
        )

    common = (
        Counter(prediction_tokens)
        & Counter(gold_tokens)
    )

    number_same = sum(common.values())

    if number_same == 0:
        return 0.0

    precision = (
        number_same
        / len(prediction_tokens)
    )
    recall = (
        number_same
        / len(gold_tokens)
    )

    return (
        2.0
        * precision
        * recall
        / (precision + recall)
    )


def answer_f1(
    prediction: str,
    gold_answers: List[str],
) -> float:
    valid_golds = [
        gold for gold in gold_answers if gold
    ]

    if not valid_golds:
        return 0.0

    return max(
        pair_f1(prediction, gold)
        for gold in valid_golds
    )


def parse_output(
    text: str,
) -> Tuple[List[str], str, bool]:
    bridges: List[str] = []
    answer = ""
    found_answer_label = False
    last_nonempty_line = ""

    for raw_line in text.strip().splitlines():
        line = raw_line.strip()

        if not line:
            continue

        last_nonempty_line = line

        bridge_match = RE_BRIDGE.match(line)

        if bridge_match:
            bridges.append(
                bridge_match.group(1).strip()
            )
            continue

        answer_match = RE_ANSWER.match(line)

        if answer_match:
            answer = answer_match.group(1).strip()
            found_answer_label = True

    if not found_answer_label and last_nonempty_line:
        answer = last_nonempty_line

    return (
        bridges,
        answer,
        found_answer_label,
    )


def bridge_matches(
    prediction: str,
    gold: str,
) -> bool:
    normalized_prediction = normalize_answer(
        prediction
    )
    normalized_gold = normalize_answer(gold)

    if not normalized_prediction or not normalized_gold:
        return False

    if normalized_prediction == normalized_gold:
        return True

    if (
        normalized_gold in normalized_prediction
        or normalized_prediction in normalized_gold
    ):
        return True

    return pair_f1(
        normalized_prediction,
        normalized_gold,
    ) >= 0.7


def compute_bridge_metrics(
    predicted_bridges: List[str],
    gold_bridges: List[str],
) -> Dict[str, float]:
    if not gold_bridges:
        return {
            "bridge_recall": 1.0,
            "bridge_f1": 1.0,
            "bridge_hits": 0,
            "number_gold": 0,
            "number_predicted": len(
                predicted_bridges
            ),
            "bridge_count_correct": float(
                len(predicted_bridges) == 0
            ),
        }

    if not predicted_bridges:
        return {
            "bridge_recall": 0.0,
            "bridge_f1": 0.0,
            "bridge_hits": 0,
            "number_gold": len(gold_bridges),
            "number_predicted": 0,
            "bridge_count_correct": 0.0,
        }

    hits = sum(
        any(
            bridge_matches(
                predicted_bridge,
                gold_bridge,
            )
            for predicted_bridge
            in predicted_bridges
        )
        for gold_bridge in gold_bridges
    )

    per_gold_f1 = [
        max(
            pair_f1(
                predicted_bridge,
                gold_bridge,
            )
            for predicted_bridge
            in predicted_bridges
        )
        for gold_bridge in gold_bridges
    ]

    return {
        "bridge_recall": (
            hits / len(gold_bridges)
        ),
        "bridge_f1": (
            sum(per_gold_f1)
            / len(per_gold_f1)
        ),
        "bridge_hits": hits,
        "number_gold": len(gold_bridges),
        "number_predicted": len(
            predicted_bridges
        ),
        "bridge_count_correct": float(
            len(predicted_bridges)
            == len(gold_bridges)
        ),
    }


def get_gold_answers(
    example: Dict,
) -> List[str]:
    answers: List[str] = []

    if example.get("final_answer"):
        answers.append(
            str(example["final_answer"])
        )

    if example.get("answer"):
        answers.append(
            str(example["answer"])
        )

    for alias in example.get(
        "answer_aliases",
        [],
    ):
        if alias:
            answers.append(str(alias))

    # Stable de-duplication.
    unique_answers = list(dict.fromkeys(answers))

    return unique_answers or [""]


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


def format_user_content(
    example: Dict,
    context_mode: str,
    prompt_variant: str,
    target_style: str,
) -> str:
    paragraphs = get_context_paragraphs(
        example,
        context_mode,
    )

    blocks: List[str] = []

    for index, paragraph in enumerate(
        paragraphs,
        start=1,
    ):
        title = str(
            paragraph.get("title", "")
        )
        text = str(
            paragraph.get(
                "text",
                paragraph.get(
                    "paragraph_text",
                    "",
                ),
            )
        )

        blocks.append(
            f"Passage {index}\n"
            f"Passage Title: {title}\n"
            f"Passage: {text}"
        )

    context = "\n\n".join(blocks)
    question = str(example["question"])

    oracle_hint = ""

    if prompt_variant == "bridge_count":
        if target_style != "bridge_aware":
            raise ValueError(
                "bridge_count is only valid for "
                "bridge_aware models."
            )

        number_bridges = len(
            example.get("bridges", [])
        )

        oracle_hint = (
            "\n\nThis question requires exactly "
            f"{number_bridges} intermediate bridge "
            f"answer(s). Output exactly {number_bridges} "
            "Bridge line(s), followed by the Answer line."
        )

    return (
        f"{context}\n\n"
        f"Question: {question}"
        f"{oracle_hint}"
    )


def get_system_prompt(
    target_style: str,
    prompt_variant: str,
) -> str:
    if prompt_variant == "minimal":
        return MINIMAL_PROMPT

    if target_style == "answer_only":
        return ANSWER_ONLY_ANCHORED_PROMPT

    return BRIDGE_AWARE_ANCHORED_PROMPT


def get_eos_token_ids(tokenizer) -> List[int]:
    token_ids: List[int] = []
    vocabulary = tokenizer.get_vocab()

    if "<|eot_id|>" in vocabulary:
        token_ids.append(
            int(vocabulary["<|eot_id|>"])
        )

    if tokenizer.eos_token_id is not None:
        eos_id = int(tokenizer.eos_token_id)

        if eos_id not in token_ids:
            token_ids.append(eos_id)

    if not token_ids:
        raise RuntimeError(
            "No generation stop token was found."
        )

    return token_ids


def load_model_and_tokenizer(
    model_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
    trust_remote_code: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token_id is None:
        vocabulary = tokenizer.get_vocab()

        if "<|finetune_right_pad_id|>" in vocabulary:
            tokenizer.pad_token = (
                "<|finetune_right_pad_id|>"
            )
        else:
            raise RuntimeError(
                f"No PAD token in tokenizer: {model_dir}"
            )

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    adapter_config_path = (
        model_dir / "adapter_config.json"
    )

    if adapter_config_path.exists():
        peft_config = PeftConfig.from_pretrained(
            model_dir
        )

        base_model = (
            AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                torch_dtype=dtype,
                attn_implementation=(
                    attn_implementation
                ),
                trust_remote_code=trust_remote_code,
            )
        )

        embedding_size = int(
            base_model.get_input_embeddings()
            .weight.shape[0]
        )

        if embedding_size != len(tokenizer):
            base_model.resize_token_embeddings(
                len(tokenizer)
            )

        model = PeftModel.from_pretrained(
            base_model,
            model_dir,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )
    model.config.use_cache = True

    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = (
            tokenizer.pad_token_id
        )

    model = model.to(device)
    model.eval()

    return model, tokenizer


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    rendered_prompts: List[str],
    device: torch.device,
    max_input_length: int,
    max_new_tokens: int,
) -> List[str]:
    inputs = tokenizer(
        rendered_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
        add_special_tokens=False,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    prompt_width = int(
        inputs["input_ids"].shape[1]
    )

    eos_token_ids = get_eos_token_ids(
        tokenizer
    )

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=(
            eos_token_ids[0]
            if len(eos_token_ids) == 1
            else eos_token_ids
        ),
        use_cache=True,
    )

    new_tokens = generated[:, prompt_width:]

    return [
        tokenizer.decode(
            row,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        for row in new_tokens
    ]


def evaluate_split(
    model,
    tokenizer,
    examples: List[Dict],
    target_style: str,
    context_mode: str,
    prompt_variant: str,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> Tuple[Dict, List[Dict]]:
    rendered_prompts: List[str] = []

    for example in examples:
        messages = [
            {
                "role": "system",
                "content": get_system_prompt(
                    target_style,
                    prompt_variant,
                ),
            },
            {
                "role": "user",
                "content": format_user_content(
                    example=example,
                    context_mode=context_mode,
                    prompt_variant=prompt_variant,
                    target_style=target_style,
                ),
            },
        ]

        rendered_prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    # Sort by approximate length for more efficient batches.
    order = sorted(
        range(len(rendered_prompts)),
        key=lambda index: len(
            rendered_prompts[index]
        ),
    )

    sorted_prompts = [
        rendered_prompts[index]
        for index in order
    ]

    sorted_outputs: List[str] = []

    for start in tqdm(
        range(0, len(sorted_prompts), batch_size),
        desc="generation",
    ):
        batch_prompts = sorted_prompts[
            start:start + batch_size
        ]

        sorted_outputs.extend(
            generate_batch(
                model=model,
                tokenizer=tokenizer,
                rendered_prompts=batch_prompts,
                device=device,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
            )
        )

    outputs: List[Optional[str]] = [
        None
    ] * len(examples)

    for sorted_index, original_index in enumerate(
        order
    ):
        outputs[original_index] = (
            sorted_outputs[sorted_index]
        )

    total_em = 0.0
    total_f1 = 0.0
    total_format = 0.0

    total_bridge_recall = 0.0
    total_bridge_f1 = 0.0
    total_bridge_count = 0.0
    total_chain_em = 0.0

    detailed: List[Dict] = []

    for example, raw_output in zip(
        examples,
        outputs,
    ):
        assert raw_output is not None

        (
            predicted_bridges,
            predicted_answer,
            found_answer_label,
        ) = parse_output(raw_output)

        gold_answers = get_gold_answers(
            example
        )
        gold_bridges = [
            str(bridge)
            for bridge in example.get(
                "bridges",
                [],
            )
        ]

        em = exact_match(
            predicted_answer,
            gold_answers,
        )
        f1 = answer_f1(
            predicted_answer,
            gold_answers,
        )

        total_em += em
        total_f1 += f1
        total_format += float(
            found_answer_label
        )

        bridge_metrics = (
            compute_bridge_metrics(
                predicted_bridges,
                gold_bridges,
            )
        )

        if target_style == "bridge_aware":
            total_bridge_recall += (
                bridge_metrics["bridge_recall"]
            )
            total_bridge_f1 += (
                bridge_metrics["bridge_f1"]
            )
            total_bridge_count += (
                bridge_metrics[
                    "bridge_count_correct"
                ]
            )

            all_bridges_hit = (
                bridge_metrics["bridge_hits"]
                == bridge_metrics["number_gold"]
            )

            chain_em = float(
                em == 1.0
                and all_bridges_hit
            )

            total_chain_em += chain_em
        else:
            chain_em = None

        detailed.append(
            {
                "id": example.get("id", ""),
                "hop": example.get("hop"),
                "question": example.get(
                    "question",
                    "",
                ),
                "target_style": target_style,
                "context_mode": context_mode,
                "prompt_variant": prompt_variant,
                "gold_bridges": gold_bridges,
                "gold_answers": gold_answers,
                "raw_output": raw_output,
                "predicted_bridges": (
                    predicted_bridges
                ),
                "predicted_answer": (
                    predicted_answer
                ),
                "found_answer_label": (
                    found_answer_label
                ),
                "em": em,
                "f1": f1,
                "bridge_recall": (
                    bridge_metrics[
                        "bridge_recall"
                    ]
                    if target_style
                    == "bridge_aware"
                    else None
                ),
                "bridge_f1": (
                    bridge_metrics["bridge_f1"]
                    if target_style
                    == "bridge_aware"
                    else None
                ),
                "bridge_count_correct": (
                    bridge_metrics[
                        "bridge_count_correct"
                    ]
                    if target_style
                    == "bridge_aware"
                    else None
                ),
                "chain_em": chain_em,
            }
        )

    number_examples = len(examples)

    if number_examples == 0:
        raise ValueError(
            "The evaluation split is empty."
        )

    metrics = {
        "n": number_examples,
        "em": (
            100.0
            * total_em
            / number_examples
        ),
        "f1": (
            100.0
            * total_f1
            / number_examples
        ),
        "answer_format_rate": (
            100.0
            * total_format
            / number_examples
        ),
        "bridge_recall": (
            100.0
            * total_bridge_recall
            / number_examples
            if target_style == "bridge_aware"
            else None
        ),
        "bridge_f1": (
            100.0
            * total_bridge_f1
            / number_examples
            if target_style == "bridge_aware"
            else None
        ),
        "bridge_count_accuracy": (
            100.0
            * total_bridge_count
            / number_examples
            if target_style == "bridge_aware"
            else None
        ),
        "chain_em": (
            100.0
            * total_chain_em
            / number_examples
            if target_style == "bridge_aware"
            else None
        ),
    }

    return metrics, detailed


def free_model(model, tokenizer) -> None:
    del model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_metrics(
    run_name: str,
    split: str,
    metrics: Dict,
) -> None:
    base = (
        f"{run_name} / {split}: "
        f"N={metrics['n']} "
        f"EM={metrics['em']:.2f} "
        f"F1={metrics['f1']:.2f} "
        f"Format={metrics['answer_format_rate']:.2f}"
    )

    if metrics["bridge_recall"] is not None:
        base += (
            f" BridgeR={metrics['bridge_recall']:.2f}"
            f" BridgeF1={metrics['bridge_f1']:.2f}"
            f" BridgeCount={metrics['bridge_count_accuracy']:.2f}"
            f" ChainEM={metrics['chain_em']:.2f}"
        )

    print(base)


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    dtype = torch_dtype_from_name(
        args.dtype
    )

    eval_dir = Path(
        args.eval_dir
    ).expanduser().resolve()

    output_dir = Path(
        args.out_dir
    ).expanduser().resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results: Dict[str, Dict] = {}

    for model_dir_string in args.model_dirs:
        model_dir = Path(
            model_dir_string
        ).expanduser().resolve()

        if not model_dir.exists():
            raise FileNotFoundError(model_dir)

        run_name = infer_run_name(model_dir)
        experiment_config = (
            load_experiment_config(model_dir)
        )

        if args.target_style == "auto":
            target_style = experiment_config.get(
                "target_style"
            )
        else:
            target_style = args.target_style

        if target_style not in {
            "answer_only",
            "bridge_aware",
        }:
            raise ValueError(
                f"Could not infer target_style for {model_dir}. "
                "Pass --target_style explicitly."
            )

        if args.context_mode == "auto":
            context_mode = experiment_config.get(
                "context_mode",
                "gold",
            )
        else:
            context_mode = args.context_mode

        if (
            args.prompt_variant == "bridge_count"
            and target_style != "bridge_aware"
        ):
            raise ValueError(
                "The bridge_count oracle can only be "
                "used with bridge_aware models."
            )

        run_output_dir = (
            output_dir
            / run_name
            / args.prompt_variant
            / context_mode
        )
        run_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("\n" + "#" * 90)
        print(f"Run            : {run_name}")
        print(f"Model directory: {model_dir}")
        print(f"Target style   : {target_style}")
        print(f"Context mode   : {context_mode}")
        print(f"Prompt variant : {args.prompt_variant}")
        print(f"Output         : {run_output_dir}")
        print("#" * 90)

        model, tokenizer = load_model_and_tokenizer(
            model_dir=model_dir,
            device=device,
            dtype=dtype,
            attn_implementation=(
                args.attn_implementation
            ),
            trust_remote_code=(
                args.trust_remote_code
            ),
        )

        run_summary: Dict[str, Dict] = {}

        for split in args.splits:
            if split not in EVAL_FILES:
                print(f"[skip] unknown split: {split}")
                continue

            split_path = (
                eval_dir / EVAL_FILES[split]
            )

            if not split_path.exists():
                print(
                    f"[skip] file not found: "
                    f"{split_path}"
                )
                continue

            examples = load_jsonl(split_path)

            if args.limit is not None:
                examples = examples[
                    :args.limit
                ]

            print(
                f"\nEvaluating {split}: "
                f"{len(examples)} examples"
            )

            metrics, detailed = evaluate_split(
                model=model,
                tokenizer=tokenizer,
                examples=examples,
                target_style=target_style,
                context_mode=context_mode,
                prompt_variant=(
                    args.prompt_variant
                ),
                batch_size=args.batch_size,
                max_input_length=(
                    args.max_input_length
                ),
                max_new_tokens=(
                    args.max_new_tokens
                ),
                device=device,
            )

            print_metrics(
                run_name,
                split,
                metrics,
            )

            run_summary[split] = metrics

            prediction_path = (
                run_output_dir
                / f"{split}_predictions.jsonl"
            )

            with prediction_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                for item in detailed:
                    file.write(
                        json.dumps(
                            item,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        with (
            run_output_dir / "summary.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "run_name": run_name,
                    "model_dir": str(model_dir),
                    "target_style": target_style,
                    "context_mode": context_mode,
                    "prompt_variant": (
                        args.prompt_variant
                    ),
                    "results": run_summary,
                },
                file,
                indent=2,
                ensure_ascii=False,
            )

        all_results[run_name] = {
            "target_style": target_style,
            "context_mode": context_mode,
            "prompt_variant": (
                args.prompt_variant
            ),
            "results": run_summary,
        }

        free_model(model, tokenizer)

    with (
        output_dir
        / f"comparison_{args.prompt_variant}.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 100)
    print("FINAL ANSWER COMPARISON")
    print("=" * 100)

    for split in args.splits:
        print(f"\n--- {split} ---")
        print(
            f"{'Run':<42}"
            f"{'N':>7}"
            f"{'EM':>10}"
            f"{'F1':>10}"
            f"{'Format':>10}"
        )
        print("-" * 79)

        for run_name, run_data in all_results.items():
            metrics = run_data["results"].get(
                split
            )

            if metrics is None:
                continue

            print(
                f"{run_name:<42}"
                f"{metrics['n']:>7}"
                f"{metrics['em']:>10.2f}"
                f"{metrics['f1']:>10.2f}"
                f"{metrics['answer_format_rate']:>10.2f}"
            )

    print("=" * 100)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()