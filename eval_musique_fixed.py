#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fixed MuSiQue evaluation for the two-base factorial SFT experiment.

Main correction:
    The output protocol says that the final Answer line contains the
    requested answer. Therefore the main metric uses the LAST explicit
    "Answer:" line, matching the previous evaluator.

Modes:
    reparse:
        Re-score previously saved raw generations. No GPU generation is
        performed. This is the fastest way to diagnose the parser issue.

    generate:
        Reload every final LoRA adapter, regenerate all outputs, and score
        them using the corrected parser.

The script discovers:
    CHECKPOINT_ROOT/*/final

It evaluates all discovered runs on:
    2hop
    3hop_linear
    4hop_linear
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import statistics
import string
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# ============================================================
# Dataset files
# ============================================================

EVAL_FILES = {
    "2hop": "eval_2hop.jsonl",
    "3hop_linear": "eval_3hop_linear.jsonl",
    "4hop_linear": "eval_4hop_linear.jsonl",
}


# ============================================================
# Prompt protocol: must match training exactly
# ============================================================

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


QWEN_CHAT_TEMPLATE = r"""
{%- for message in messages %}
{{- '<|im_start|>' + message['role'] + '\n' }}
{{- message['content'] | trim }}
{{- '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- endif %}
"""


COMMON_OUTPUT_SCHEMA = (
    "Response format:\n"
    "- You may output zero or more intermediate bridge lines in dependency "
    "order, using 'Bridge 1: ...', 'Bridge 2: ...', and so on.\n"
    "- The final non-empty line must be exactly "
    "'Answer: <short final answer>'.\n"
    "- Do not output explanations outside these lines."
)


CANONICAL_INSTRUCTION = (
    "Use only information from the supplied passages to answer the question."
)


ANCHORING_ADDITION = (
    "Treat the question labeled 'Question:' as the exact target goal. "
    "Keep conditioning on that same target while producing any intermediate "
    "bridges. Intermediate bridge answers are only supporting steps and must "
    "not replace the requested final answer. Before stopping, verify that the "
    "final Answer line directly answers the exact target question."
)


# ============================================================
# Output parsing
# ============================================================

RE_BRIDGE = re.compile(
    r"^\s*Bridge\s*(\d+)\s*[:：]\s*(.*?)\s*$",
    re.IGNORECASE,
)

RE_ANSWER = re.compile(
    r"^\s*Answer\s*[:：]\s*(.*?)\s*$",
    re.IGNORECASE,
)

RE_TRAILING_SCHEMA = re.compile(
    r"\s+(?:Bridge\s*\d+|Answer)\s*[:：]",
    re.IGNORECASE,
)


MAIN_METRICS = [
    "em",
    "f1",
    "bridge_recall",
    "bridge_f1",
    "bridge_count_accuracy",
    "chain_em",
    "mean_predicted_bridges",
]


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["reparse", "generate"],
        required=True,
    )

    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default="checkpoints/musique-two-base-factorial-1000",
    )

    parser.add_argument(
        "--source_results_dir",
        type=str,
        default="eval_results/musique-two-base-factorial-1000",
        help=(
            "Used only by reparse mode. This directory must contain "
            "<run_name>/<split>_predictions.jsonl."
        ),
    )

    parser.add_argument(
        "--eval_dir",
        type=str,
        default="prepared_data_2hop",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
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
        "--batch_size",
        type=int,
        default=8,
    )

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
        choices=[
            "sdpa",
            "eager",
            "flash_attention_2",
        ],
        default="sdpa",
    )

    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    return parser.parse_args()


# ============================================================
# Basic I/O
# ============================================================

def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def write_json(
    path: Path,
    value,
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


def write_jsonl(
    path: Path,
    rows: List[Dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_csv(
    path: Path,
    rows: List[Dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    fields: List[str] = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Experiment discovery
# ============================================================

def discover_final_models(
    checkpoint_root: Path,
) -> List[Path]:
    final_dirs: List[Path] = []

    if not checkpoint_root.exists():
        raise FileNotFoundError(
            checkpoint_root
        )

    for candidate in sorted(
        checkpoint_root.glob("*/final")
    ):
        if (
            candidate.is_dir()
            and (
                candidate
                / "adapter_config.json"
            ).exists()
        ):
            final_dirs.append(
                candidate.resolve()
            )

    if not final_dirs:
        raise ValueError(
            "No final LoRA adapters were found under "
            f"{checkpoint_root}"
        )

    return final_dirs


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

    raise FileNotFoundError(
        f"No experiment_config.json found for {model_dir}"
    )


# ============================================================
# Prompt construction
# ============================================================

def install_chat_template_if_needed(
    tokenizer,
) -> str:
    if tokenizer.chat_template:
        return "existing"

    vocabulary = tokenizer.get_vocab()

    if (
        "<|im_start|>" in vocabulary
        and "<|im_end|>" in vocabulary
    ):
        tokenizer.chat_template = (
            QWEN_CHAT_TEMPLATE
        )
        return "installed_qwen"

    if (
        "<|start_header_id|>" in vocabulary
        and "<|end_header_id|>" in vocabulary
        and "<|eot_id|>" in vocabulary
    ):
        tokenizer.chat_template = (
            LLAMA3_CHAT_TEMPLATE
        )
        return "installed_llama"

    raise RuntimeError(
        "Tokenizer has no supported chat template."
    )


def ensure_pad_token(
    tokenizer,
) -> int:
    if tokenizer.pad_token_id is not None:
        return 0

    vocabulary = tokenizer.get_vocab()

    for token in [
        "<|finetune_right_pad_id|>",
        "<|endoftext|>",
        "<|end_of_text|>",
    ]:
        if token in vocabulary:
            tokenizer.pad_token = token
            return 0

    if tokenizer.eos_token is not None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )
        return 0

    return int(
        tokenizer.add_special_tokens(
            {"pad_token": "<|pad|>"}
        )
    )


def build_system_prompt(
    prompt_style: str,
) -> str:
    if prompt_style == "canonical":
        instruction = (
            CANONICAL_INSTRUCTION
        )

    elif prompt_style == "anchored":
        instruction = (
            f"{CANONICAL_INSTRUCTION}\n"
            f"{ANCHORING_ADDITION}"
        )

    else:
        raise ValueError(
            f"Unknown prompt style: {prompt_style}"
        )

    return (
        f"{instruction}\n\n"
        f"{COMMON_OUTPUT_SCHEMA}"
    )


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

    if context_mode == "all":
        return list(
            example.get(
                "all_context_paragraphs",
                example.get(
                    "context_paragraphs",
                    [],
                ),
            )
        )

    raise ValueError(
        f"Unknown context mode: {context_mode}"
    )


def format_user_content(
    example: Dict,
    context_mode: str,
    prompt_style: str,
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

    context = (
        "\n\n".join(blocks)
        if blocks
        else "(No passages provided.)"
    )

    question = str(
        example["question"]
    )

    result = (
        f"{context}\n\n"
        f"Question: {question}"
    )

    if prompt_style == "anchored":
        result += (
            "\n\n"
            "Target reminder: the final Answer line "
            "must answer exactly this question:\n"
            f"{question}"
        )

    return result


def build_prompt_messages(
    example: Dict,
    context_mode: str,
    prompt_style: str,
) -> List[Dict]:
    return [
        {
            "role": "system",
            "content": build_system_prompt(
                prompt_style
            ),
        },
        {
            "role": "user",
            "content": format_user_content(
                example,
                context_mode,
                prompt_style,
            ),
        },
    ]


# ============================================================
# Answer normalization and metrics
# ============================================================

def normalize_answer(
    text: Optional[str],
) -> str:
    if text is None:
        return ""

    def remove_articles(
        value: str,
    ) -> str:
        return re.sub(
            r"\b(a|an|the)\b",
            " ",
            value,
        )

    def remove_punctuation(
        value: str,
    ) -> str:
        punctuation = set(
            string.punctuation
        )

        return "".join(
            character
            for character in value
            if character not in punctuation
        )

    return " ".join(
        remove_articles(
            remove_punctuation(
                text.lower()
            )
        ).split()
    )


def pair_f1(
    prediction: str,
    gold: str,
) -> float:
    prediction_tokens = (
        normalize_answer(
            prediction
        ).split()
    )

    gold_tokens = (
        normalize_answer(
            gold
        ).split()
    )

    if (
        not prediction_tokens
        or not gold_tokens
    ):
        return float(
            prediction_tokens
            == gold_tokens
        )

    common = (
        Counter(prediction_tokens)
        & Counter(gold_tokens)
    )

    number_same = sum(
        common.values()
    )

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


def exact_match(
    prediction: str,
    gold_answers: List[str],
) -> float:
    normalized_prediction = (
        normalize_answer(
            prediction
        )
    )

    return float(
        any(
            normalized_prediction
            == normalize_answer(gold)
            for gold in gold_answers
            if gold
        )
    )


def answer_f1(
    prediction: str,
    gold_answers: List[str],
) -> float:
    valid_golds = [
        gold
        for gold in gold_answers
        if gold
    ]

    if not valid_golds:
        return 0.0

    return max(
        pair_f1(
            prediction,
            gold,
        )
        for gold in valid_golds
    )


# ============================================================
# Corrected output parser
# ============================================================

def clean_schema_value(
    value: str,
) -> str:
    """
    Remove accidental schema continuation on the same line.

    Example:
        "Amador County Answer: Amador County"
    becomes:
        "Amador County"
    """

    value = value.strip()

    trailing_match = (
        RE_TRAILING_SCHEMA.search(
            value
        )
    )

    if trailing_match:
        value = value[
            :trailing_match.start()
        ]

    return value.strip(
        " \t`*_-\u2022"
    )


def parse_output_fixed(
    text: str,
) -> Dict:
    bridges: List[str] = []
    answer_candidates: List[str] = []
    nonempty_lines: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        nonempty_lines.append(
            line
        )

        bridge_match = RE_BRIDGE.match(
            line
        )

        if bridge_match:
            value = clean_schema_value(
                bridge_match.group(2)
            )

            if value:
                bridges.append(
                    value
                )

            continue

        answer_match = RE_ANSWER.match(
            line
        )

        if answer_match:
            value = clean_schema_value(
                answer_match.group(1)
            )

            if value:
                answer_candidates.append(
                    value
                )

            continue

    fallback = (
        nonempty_lines[-1]
        if nonempty_lines
        else ""
    )

    first_answer = (
        answer_candidates[0]
        if answer_candidates
        else fallback
    )

    # Main correction:
    # use the final explicit Answer line.
    last_answer = (
        answer_candidates[-1]
        if answer_candidates
        else fallback
    )

    final_line_is_answer = bool(
        nonempty_lines
        and RE_ANSWER.match(
            nonempty_lines[-1]
        )
    )

    return {
        "bridges": bridges,
        "answer_candidates": (
            answer_candidates
        ),
        "first_answer": (
            first_answer
        ),
        "last_answer": (
            last_answer
        ),
        "answer_line_count": len(
            answer_candidates
        ),
        "has_answer_label": bool(
            answer_candidates
        ),
        "final_line_is_answer": (
            final_line_is_answer
        ),
        "last_nonempty_line": (
            fallback
        ),
    }


# ============================================================
# Bridge metrics
# ============================================================

def bridge_matches(
    prediction: str,
    gold: str,
) -> bool:
    normalized_prediction = (
        normalize_answer(
            prediction
        )
    )

    normalized_gold = (
        normalize_answer(
            gold
        )
    )

    if (
        not normalized_prediction
        or not normalized_gold
    ):
        return False

    if (
        normalized_prediction
        == normalized_gold
    ):
        return True

    if (
        normalized_gold
        in normalized_prediction
        or normalized_prediction
        in normalized_gold
    ):
        return True

    return (
        pair_f1(
            prediction,
            gold,
        )
        >= 0.7
    )


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
            "number_gold": len(
                gold_bridges
            ),
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
        "number_gold": len(
            gold_bridges
        ),
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

    for key in [
        "final_answer",
        "answer",
    ]:
        if example.get(key):
            answers.append(
                str(example[key])
            )

    for alias in example.get(
        "answer_aliases",
        [],
    ):
        if alias:
            answers.append(
                str(alias)
            )

    answers = list(
        dict.fromkeys(answers)
    )

    return answers or [""]


# ============================================================
# Score records
# ============================================================

def score_records(
    records: List[Dict],
) -> Tuple[Dict, List[Dict]]:
    totals = defaultdict(float)
    details: List[Dict] = []

    for record in records:
        raw_output = str(
            record.get(
                "raw_output",
                record.get(
                    "pred_raw",
                    "",
                ),
            )
        )

        gold_answers = record.get(
            "gold_answers",
            record.get(
                "gold_final",
                [""],
            ),
        )

        if isinstance(
            gold_answers,
            str,
        ):
            gold_answers = [
                gold_answers
            ]

        gold_answers = [
            str(value)
            for value in gold_answers
            if value is not None
        ] or [""]

        gold_bridges = [
            str(value)
            for value in record.get(
                "gold_bridges",
                [],
            )
        ]

        parsed = parse_output_fixed(
            raw_output
        )

        first_answer = parsed[
            "first_answer"
        ]

        last_answer = parsed[
            "last_answer"
        ]

        answer_candidates = parsed[
            "answer_candidates"
        ]

        main_em = exact_match(
            last_answer,
            gold_answers,
        )

        main_f1 = answer_f1(
            last_answer,
            gold_answers,
        )

        first_em = exact_match(
            first_answer,
            gold_answers,
        )

        first_f1 = answer_f1(
            first_answer,
            gold_answers,
        )

        oracle_candidates = (
            answer_candidates
            if answer_candidates
            else [last_answer]
        )

        any_answer_em = max(
            exact_match(
                candidate,
                gold_answers,
            )
            for candidate
            in oracle_candidates
        )

        any_answer_f1 = max(
            answer_f1(
                candidate,
                gold_answers,
            )
            for candidate
            in oracle_candidates
        )

        bridge_metrics = (
            compute_bridge_metrics(
                parsed["bridges"],
                gold_bridges,
            )
        )

        all_bridges_hit = (
            bridge_metrics[
                "bridge_hits"
            ]
            == bridge_metrics[
                "number_gold"
            ]
        )

        chain_em = float(
            main_em == 1.0
            and all_bridges_hit
        )

        totals["em"] += main_em
        totals["f1"] += main_f1
        totals["first_answer_em"] += (
            first_em
        )
        totals["first_answer_f1"] += (
            first_f1
        )
        totals["any_answer_em"] += (
            any_answer_em
        )
        totals["any_answer_f1"] += (
            any_answer_f1
        )

        totals["answer_format_rate"] += float(
            parsed["has_answer_label"]
        )

        totals["final_answer_line_rate"] += float(
            parsed[
                "final_line_is_answer"
            ]
        )

        totals["multiple_answer_rate"] += float(
            parsed[
                "answer_line_count"
            ]
            > 1
        )

        totals["mean_answer_lines"] += (
            parsed[
                "answer_line_count"
            ]
        )

        totals["bridge_recall"] += (
            bridge_metrics[
                "bridge_recall"
            ]
        )

        totals["bridge_f1"] += (
            bridge_metrics[
                "bridge_f1"
            ]
        )

        totals["bridge_count_accuracy"] += (
            bridge_metrics[
                "bridge_count_correct"
            ]
        )

        totals["chain_em"] += (
            chain_em
        )

        totals[
            "mean_predicted_bridges"
        ] += len(
            parsed["bridges"]
        )

        details.append(
            {
                **record,
                "raw_output": raw_output,
                "gold_answers": gold_answers,
                "gold_bridges": gold_bridges,

                "predicted_bridges": (
                    parsed["bridges"]
                ),
                "answer_candidates": (
                    answer_candidates
                ),
                "answer_line_count": (
                    parsed[
                        "answer_line_count"
                    ]
                ),

                "first_predicted_answer": (
                    first_answer
                ),
                "predicted_answer": (
                    last_answer
                ),

                "first_answer_em": (
                    first_em
                ),
                "first_answer_f1": (
                    first_f1
                ),

                "em": main_em,
                "f1": main_f1,

                "any_answer_em": (
                    any_answer_em
                ),
                "any_answer_f1": (
                    any_answer_f1
                ),

                "has_answer_label": (
                    parsed[
                        "has_answer_label"
                    ]
                ),
                "final_line_is_answer": (
                    parsed[
                        "final_line_is_answer"
                    ]
                ),

                "bridge_recall": (
                    bridge_metrics[
                        "bridge_recall"
                    ]
                ),
                "bridge_f1": (
                    bridge_metrics[
                        "bridge_f1"
                    ]
                ),
                "bridge_count_correct": (
                    bridge_metrics[
                        "bridge_count_correct"
                    ]
                ),

                "chain_em": (
                    chain_em
                ),
            }
        )

    number_examples = len(
        records
    )

    if number_examples == 0:
        raise ValueError(
            "Cannot score an empty record list."
        )

    percent_metrics = [
        "em",
        "f1",
        "first_answer_em",
        "first_answer_f1",
        "any_answer_em",
        "any_answer_f1",
        "answer_format_rate",
        "final_answer_line_rate",
        "multiple_answer_rate",
        "bridge_recall",
        "bridge_f1",
        "bridge_count_accuracy",
        "chain_em",
    ]

    metrics: Dict[str, float] = {
        "n": number_examples,
    }

    for metric in percent_metrics:
        metrics[metric] = (
            100.0
            * totals[metric]
            / number_examples
        )

    metrics["mean_answer_lines"] = (
        totals["mean_answer_lines"]
        / number_examples
    )

    metrics["mean_predicted_bridges"] = (
        totals[
            "mean_predicted_bridges"
        ]
        / number_examples
    )

    return metrics, details


# ============================================================
# Model loading and generation
# ============================================================

def torch_dtype_from_name(
    name: str,
) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16

    if name == "fp16":
        return torch.float16

    return torch.float32


def get_eos_token_ids(
    tokenizer,
) -> List[int]:
    token_ids: List[int] = []
    vocabulary = tokenizer.get_vocab()

    for token in [
        "<|eot_id|>",
        "<|im_end|>",
    ]:
        if token in vocabulary:
            token_id = int(
                vocabulary[token]
            )

            if token_id not in token_ids:
                token_ids.append(
                    token_id
                )

    if tokenizer.eos_token_id is not None:
        eos_id = int(
            tokenizer.eos_token_id
        )

        if eos_id not in token_ids:
            token_ids.append(
                eos_id
            )

    if not token_ids:
        raise RuntimeError(
            "No generation stop token found."
        )

    return token_ids


def load_model_and_tokenizer(
    model_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
    local_files_only: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        use_fast=True,
        trust_remote_code=True,
        local_files_only=(
            local_files_only
        ),
    )

    install_chat_template_if_needed(
        tokenizer
    )

    ensure_pad_token(
        tokenizer
    )

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    peft_config = (
        PeftConfig.from_pretrained(
            str(model_dir)
        )
    )

    base_model_name = (
        peft_config
        .base_model_name_or_path
    )

    if not base_model_name:
        raise RuntimeError(
            f"No base model recorded in {model_dir}"
        )

    base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            base_model_name,
            dtype=dtype,
            attn_implementation=(
                attn_implementation
            ),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=(
                local_files_only
            ),
        )
    )

    embedding_size = int(
        base_model
        .get_input_embeddings()
        .weight.shape[0]
    )

    if len(tokenizer) > embedding_size:
        print(
            "Expanding base embeddings: "
            f"{embedding_size} "
            f"-> {len(tokenizer)}"
        )

        base_model.resize_token_embeddings(
            len(tokenizer)
        )

    elif len(tokenizer) < embedding_size:
        print(
            "Keeping base vocabulary unchanged: "
            f"model={embedding_size}, "
            f"tokenizer={len(tokenizer)}"
        )

    model = PeftModel.from_pretrained(
        base_model,
        str(model_dir),
        is_trainable=False,
    )

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )
    model.config.use_cache = True

    if (
        hasattr(
            model,
            "generation_config",
        )
        and model.generation_config
        is not None
    ):
        model.generation_config.pad_token_id = (
            tokenizer.pad_token_id
        )
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None

    model = model.to(
        device
    )
    model.eval()

    return model, tokenizer


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int,
) -> List[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )

    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
    }

    prompt_width = int(
        inputs["input_ids"].shape[1]
    )

    eos_ids = get_eos_token_ids(
        tokenizer
    )

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=(
            tokenizer.pad_token_id
        ),
        eos_token_id=(
            eos_ids[0]
            if len(eos_ids) == 1
            else eos_ids
        ),
        use_cache=True,
    )

    new_tokens = generated[
        :,
        prompt_width:,
    ]

    return [
        tokenizer.decode(
            row,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        for row in new_tokens
    ]


def generate_records(
    model,
    tokenizer,
    examples: List[Dict],
    prompt_style: str,
    context_mode: str,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> List[Dict]:
    rendered_prompts: List[str] = []

    for example in examples:
        messages = build_prompt_messages(
            example,
            context_mode,
            prompt_style,
        )

        rendered_prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    prompt_lengths = [
        len(
            tokenizer(
                prompt,
                add_special_tokens=False,
            )["input_ids"]
        )
        for prompt in rendered_prompts
    ]

    maximum_prompt_length = max(
        prompt_lengths,
        default=0,
    )

    if (
        maximum_prompt_length
        > max_input_length
    ):
        raise ValueError(
            "Evaluation input exceeds budget: "
            f"{maximum_prompt_length} "
            f"> {max_input_length}"
        )

    order = sorted(
        range(len(rendered_prompts)),
        key=lambda index: (
            prompt_lengths[index]
        ),
    )

    sorted_prompts = [
        rendered_prompts[index]
        for index in order
    ]

    sorted_outputs: List[str] = []

    for start in tqdm(
        range(
            0,
            len(sorted_prompts),
            batch_size,
        ),
        desc="generation",
    ):
        sorted_outputs.extend(
            generate_batch(
                model,
                tokenizer,
                sorted_prompts[
                    start:start + batch_size
                ],
                device,
                max_new_tokens,
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

    records: List[Dict] = []

    for example, raw_output in zip(
        examples,
        outputs,
    ):
        assert raw_output is not None

        records.append(
            {
                "id": example.get(
                    "id",
                    "",
                ),
                "hop": example.get(
                    "hop"
                ),
                "question": example.get(
                    "question",
                    "",
                ),
                "gold_answers": (
                    get_gold_answers(
                        example
                    )
                ),
                "gold_bridges": [
                    str(value)
                    for value
                    in example.get(
                        "bridges",
                        [],
                    )
                ],
                "raw_output": raw_output,
            }
        )

    return records


def free_model(
    model,
    tokenizer,
) -> None:
    del model
    del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Aggregation
# ============================================================

def flatten_summaries(
    summaries: List[Dict],
) -> List[Dict]:
    rows: List[Dict] = []

    for summary in summaries:
        experiment = summary[
            "experiment"
        ]

        for split, metrics in (
            summary["results"].items()
        ):
            row = {
                "run_name": (
                    summary["run_name"]
                ),
                "model_dir": (
                    summary["model_dir"]
                ),
                "base_key": (
                    experiment.get(
                        "base_key",
                        "unknown",
                    )
                ),
                "model_name": (
                    experiment.get(
                        "model_name",
                        "unknown",
                    )
                ),
                "seed": experiment.get(
                    "seed",
                    42,
                ),
                "target_style": (
                    experiment[
                        "target_style"
                    ]
                ),
                "prompt_style": (
                    experiment[
                        "prompt_style"
                    ]
                ),
                "split": split,
            }

            row.update(metrics)
            rows.append(row)

    return rows


def build_contrasts(
    rows: List[Dict],
) -> List[Dict]:
    index: Dict[Tuple, Dict] = {}

    for row in rows:
        key = (
            row["base_key"],
            int(row["seed"]),
            row["target_style"],
            row["prompt_style"],
            row["split"],
        )

        index[key] = row

    combinations = sorted(
        {
            (
                row["base_key"],
                int(row["seed"]),
                row["split"],
            )
            for row in rows
        }
    )

    contrasts: List[Dict] = []

    for base_key, seed, split in combinations:
        for target_style in [
            "answer_only",
            "bridge_aware",
        ]:
            canonical = index.get(
                (
                    base_key,
                    seed,
                    target_style,
                    "canonical",
                    split,
                )
            )

            anchored = index.get(
                (
                    base_key,
                    seed,
                    target_style,
                    "anchored",
                    split,
                )
            )

            if (
                canonical is not None
                and anchored is not None
            ):
                contrast = {
                    "contrast_type": (
                        "prompt_anchoring"
                    ),
                    "base_key": base_key,
                    "seed": seed,
                    "split": split,
                    "fixed_factor": (
                        "target_style"
                    ),
                    "fixed_value": (
                        target_style
                    ),
                    "condition_a": (
                        "canonical"
                    ),
                    "condition_b": (
                        "anchored"
                    ),
                }

                for metric in MAIN_METRICS:
                    contrast[
                        f"delta_{metric}"
                    ] = (
                        float(
                            anchored[metric]
                        )
                        - float(
                            canonical[metric]
                        )
                    )

                contrasts.append(
                    contrast
                )

        for prompt_style in [
            "canonical",
            "anchored",
        ]:
            answer_only = index.get(
                (
                    base_key,
                    seed,
                    "answer_only",
                    prompt_style,
                    split,
                )
            )

            bridge_aware = index.get(
                (
                    base_key,
                    seed,
                    "bridge_aware",
                    prompt_style,
                    split,
                )
            )

            if (
                answer_only is not None
                and bridge_aware is not None
            ):
                contrast = {
                    "contrast_type": (
                        "bridge_supervision"
                    ),
                    "base_key": base_key,
                    "seed": seed,
                    "split": split,
                    "fixed_factor": (
                        "prompt_style"
                    ),
                    "fixed_value": (
                        prompt_style
                    ),
                    "condition_a": (
                        "answer_only"
                    ),
                    "condition_b": (
                        "bridge_aware"
                    ),
                }

                for metric in MAIN_METRICS:
                    contrast[
                        f"delta_{metric}"
                    ] = (
                        float(
                            bridge_aware[
                                metric
                            ]
                        )
                        - float(
                            answer_only[
                                metric
                            ]
                        )
                    )

                contrasts.append(
                    contrast
                )

    return contrasts


def print_metrics(
    run_name: str,
    split: str,
    metrics: Dict,
) -> None:
    print(
        f"{run_name} / {split}: "
        f"N={metrics['n']} "
        f"LastEM={metrics['em']:.2f} "
        f"LastF1={metrics['f1']:.2f} "
        f"FirstEM={metrics['first_answer_em']:.2f} "
        f"AnyAnswerEM={metrics['any_answer_em']:.2f} "
        f"MultiAnswer={metrics['multiple_answer_rate']:.2f} "
        f"FinalLine={metrics['final_answer_line_rate']:.2f} "
        f"BridgeR={metrics['bridge_recall']:.2f} "
        f"BridgeCount={metrics['bridge_count_accuracy']:.2f}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    checkpoint_root = Path(
        args.checkpoint_root
    ).expanduser().resolve()

    source_results_dir = Path(
        args.source_results_dir
    ).expanduser().resolve()

    eval_dir = Path(
        args.eval_dir
    ).expanduser().resolve()

    out_dir = Path(
        args.out_dir
    ).expanduser().resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_dirs = discover_final_models(
        checkpoint_root
    )

    print(
        f"Found {len(model_dirs)} final models:"
    )

    for model_dir in model_dirs:
        print(
            f"  {model_dir}"
        )

    if args.mode == "generate":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for generate mode."
            )

        device = torch.device(
            args.device
        )

        dtype = torch_dtype_from_name(
            args.dtype
        )

    summaries: List[Dict] = []
    failures: List[Dict] = []

    for model_dir in model_dirs:
        model = None
        tokenizer = None

        try:
            experiment = (
                load_experiment_config(
                    model_dir
                )
            )

            run_name = experiment.get(
                "run_name",
                model_dir.parent.name,
            )

            target_style = experiment[
                "target_style"
            ]

            prompt_style = experiment[
                "prompt_style"
            ]

            context_mode = experiment.get(
                "context_mode",
                "gold",
            )

            run_out_dir = (
                out_dir / run_name
            )

            run_out_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            print(
                "\n" + "#" * 100
            )
            print(
                f"Run          : {run_name}"
            )
            print(
                f"Mode         : {args.mode}"
            )
            print(
                f"Target style : {target_style}"
            )
            print(
                f"Prompt style : {prompt_style}"
            )
            print(
                f"Model dir    : {model_dir}"
            )
            print(
                "#" * 100
            )

            if args.mode == "generate":
                model, tokenizer = (
                    load_model_and_tokenizer(
                        model_dir,
                        device,
                        dtype,
                        args.attn_implementation,
                        args.local_files_only,
                    )
                )

            run_results: Dict[str, Dict] = {}

            for split in args.splits:
                if split not in EVAL_FILES:
                    continue

                if args.mode == "reparse":
                    source_path = (
                        source_results_dir
                        / run_name
                        / (
                            f"{split}_"
                            "predictions.jsonl"
                        )
                    )

                    if not source_path.exists():
                        raise FileNotFoundError(
                            source_path
                        )

                    records = load_jsonl(
                        source_path
                    )

                    if args.limit is not None:
                        records = records[
                            :args.limit
                        ]

                else:
                    split_path = (
                        eval_dir
                        / EVAL_FILES[split]
                    )

                    if not split_path.exists():
                        raise FileNotFoundError(
                            split_path
                        )

                    examples = load_jsonl(
                        split_path
                    )

                    if args.limit is not None:
                        examples = examples[
                            :args.limit
                        ]

                    print(
                        f"\nGenerating {split}: "
                        f"{len(examples)} examples"
                    )

                    records = generate_records(
                        model,
                        tokenizer,
                        examples,
                        prompt_style,
                        context_mode,
                        args.batch_size,
                        args.max_input_length,
                        args.max_new_tokens,
                        device,
                    )

                metrics, details = (
                    score_records(
                        records
                    )
                )

                print_metrics(
                    run_name,
                    split,
                    metrics,
                )

                run_results[
                    split
                ] = metrics

                write_jsonl(
                    run_out_dir
                    / (
                        f"{split}_"
                        "predictions_fixed.jsonl"
                    ),
                    details,
                )

            run_summary = {
                "run_name": run_name,
                "model_dir": str(
                    model_dir
                ),
                "mode": args.mode,
                "experiment": (
                    experiment
                ),
                "results": (
                    run_results
                ),
            }

            write_json(
                run_out_dir
                / "summary_fixed.json",
                run_summary,
            )

            summaries.append(
                run_summary
            )

        except Exception as error:
            failure = {
                "model_dir": str(
                    model_dir
                ),
                "error": repr(
                    error
                ),
                "traceback": (
                    traceback.format_exc()
                ),
            }

            failures.append(
                failure
            )

            print(
                failure["traceback"]
            )

        finally:
            if (
                model is not None
                and tokenizer is not None
            ):
                free_model(
                    model,
                    tokenizer,
                )

    rows = flatten_summaries(
        summaries
    )

    contrasts = build_contrasts(
        rows
    )

    write_json(
        out_dir / "all_results_fixed.json",
        {
            "mode": args.mode,
            "runs": summaries,
            "failures": failures,
        },
    )

    write_csv(
        out_dir / "all_results_fixed.csv",
        rows,
    )

    write_csv(
        out_dir / "contrasts_fixed.csv",
        contrasts,
    )

    write_json(
        out_dir / "failures.json",
        failures,
    )

    print(
        "\n" + "=" * 120
    )
    print(
        "CORRECTED FINAL RESULTS"
    )
    print(
        "=" * 120
    )

    for row in rows:
        print(
            f"{row['base_key']:<13} "
            f"{row['target_style']:<13} "
            f"{row['prompt_style']:<10} "
            f"{row['split']:<13} "
            f"LastEM={row['em']:>6.2f} "
            f"LastF1={row['f1']:>6.2f} "
            f"FirstEM={row['first_answer_em']:>6.2f} "
            f"AnyEM={row['any_answer_em']:>6.2f} "
            f"MultiAnswer={row['multiple_answer_rate']:>6.2f}"
        )

    print(
        "=" * 120
    )
    print(
        f"Results saved to: {out_dir}"
    )

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()