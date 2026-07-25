#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from musique_common import (
    build_prompt_messages,
    ensure_pad_token,
    infer_run_name,
    install_chat_template_if_needed,
    load_experiment_config,
    load_jsonl,
)


EVAL_FILES = {
    "2hop": "eval_2hop.jsonl",
    "3hop_linear": "eval_3hop_linear.jsonl",
    "4hop_linear": "eval_4hop_linear.jsonl",
}


RE_BRIDGE = re.compile(
    r"^\s*Bridge\s*(\d+)\s*[:：]\s*(.+?)\s*$",
    re.IGNORECASE,
)

RE_ANSWER = re.compile(
    r"^\s*Answer\s*[:：]\s*(.+?)\s*$",
    re.IGNORECASE,
)


METRIC_NAMES = [
    "em",
    "f1",
    "answer_format_rate",
    "bridge_recall",
    "bridge_f1",
    "bridge_count_accuracy",
    "chain_em",
    "mean_predicted_bridges",
]


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
        default="eval_results/musique_factorial",
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
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser.parse_args()


def torch_dtype_from_name(
    name: str,
) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16

    if name == "fp16":
        return torch.float16

    return torch.float32


def normalize_answer(
    text: Optional[str],
) -> str:
    if text is None:
        return ""

    def remove_articles(value: str) -> str:
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

    def normalize_whitespace(
        value: str,
    ) -> str:
        return " ".join(
            value.split()
        )

    return normalize_whitespace(
        remove_articles(
            remove_punctuation(
                text.lower()
            )
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
        normalize_answer(prediction)
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


def parse_output(
    text: str,
) -> Tuple[List[str], str, bool]:
    indexed_bridges: List[
        Tuple[int, str]
    ] = []

    answer = ""
    found_answer_label = False
    last_nonempty_line = ""

    for raw_line in text.strip().splitlines():
        line = raw_line.strip()

        if not line:
            continue

        last_nonempty_line = line

        bridge_match = RE_BRIDGE.match(
            line
        )

        if bridge_match:
            bridge_index = int(
                bridge_match.group(1)
            )

            bridge_value = (
                bridge_match.group(2)
                .strip()
            )

            indexed_bridges.append(
                (
                    bridge_index,
                    bridge_value,
                )
            )

            continue

        answer_match = RE_ANSWER.match(
            line
        )

        if (
            answer_match
            and not found_answer_label
        ):
            answer = (
                answer_match.group(1)
                .strip()
            )

            found_answer_label = True

    indexed_bridges.sort(
        key=lambda item: item[0]
    )

    bridges = [
        value
        for _, value
        in indexed_bridges
    ]

    if (
        not found_answer_label
        and last_nonempty_line
    ):
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
    normalized_prediction = (
        normalize_answer(prediction)
    )

    normalized_gold = (
        normalize_answer(gold)
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
        for gold_bridge
        in gold_bridges
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
            answers.append(
                str(alias)
            )

    answers = list(
        dict.fromkeys(answers)
    )

    return answers or [""]


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
            token_ids.append(eos_id)

    if not token_ids:
        raise RuntimeError(
            "No generation stopping token "
            "was found."
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
        model_dir,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    install_chat_template_if_needed(
        tokenizer
    )

    ensure_pad_token(tokenizer)

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    adapter_config_path = (
        model_dir
        / "adapter_config.json"
    )

    if adapter_config_path.exists():
        peft_config = (
            PeftConfig.from_pretrained(
                model_dir
            )
        )

        base_model_name = (
            peft_config
            .base_model_name_or_path
        )

        if not base_model_name:
            raise RuntimeError(
                "adapter_config.json has no "
                "base_model_name_or_path."
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

        if embedding_size != len(tokenizer):
            base_model.resize_token_embeddings(
                len(tokenizer)
            )

        model = PeftModel.from_pretrained(
            base_model,
            model_dir,
            is_trainable=False,
        )
    else:
        model = (
            AutoModelForCausalLM
            .from_pretrained(
                model_dir,
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

    model = model.to(device)
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

    eos_token_ids = (
        get_eos_token_ids(
            tokenizer
        )
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
            eos_token_ids[0]
            if len(eos_token_ids) == 1
            else eos_token_ids
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


def evaluate_split(
    model,
    tokenizer,
    examples: List[Dict],
    target_style: str,
    prompt_style: str,
    context_mode: str,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> Tuple[Dict, List[Dict]]:
    rendered_prompts: List[str] = []

    for example in examples:
        messages = build_prompt_messages(
            example=example,
            context_mode=context_mode,
            prompt_style=prompt_style,
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

    maximum_found = max(
        prompt_lengths,
        default=0,
    )

    if maximum_found > max_input_length:
        raise ValueError(
            "Evaluation prompt exceeds the "
            "configured input budget: "
            f"{maximum_found} > "
            f"{max_input_length}. Increase "
            "--max_input_length."
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
        batch_prompts = sorted_prompts[
            start:start + batch_size
        ]

        sorted_outputs.extend(
            generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                device=device,
                max_new_tokens=(
                    max_new_tokens
                ),
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
    total_predicted_bridges = 0.0

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

        bridge_metrics = (
            compute_bridge_metrics(
                predicted_bridges,
                gold_bridges,
            )
        )

        all_bridges_hit = (
            bridge_metrics["bridge_hits"]
            == bridge_metrics["number_gold"]
        )

        chain_em = float(
            em == 1.0
            and all_bridges_hit
        )

        total_em += em
        total_f1 += f1
        total_format += float(
            found_answer_label
        )
        total_bridge_recall += (
            bridge_metrics[
                "bridge_recall"
            ]
        )
        total_bridge_f1 += (
            bridge_metrics[
                "bridge_f1"
            ]
        )
        total_bridge_count += (
            bridge_metrics[
                "bridge_count_correct"
            ]
        )
        total_chain_em += chain_em
        total_predicted_bridges += len(
            predicted_bridges
        )

        detailed.append(
            {
                "id": example.get("id", ""),
                "hop": example.get("hop"),
                "question": example.get(
                    "question",
                    "",
                ),
                "target_style": target_style,
                "prompt_style": prompt_style,
                "context_mode": context_mode,
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
                "chain_em": chain_em,
            }
        )

    number_examples = len(examples)

    if number_examples == 0:
        raise ValueError(
            "Evaluation split is empty."
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
        ),
        "bridge_f1": (
            100.0
            * total_bridge_f1
            / number_examples
        ),
        "bridge_count_accuracy": (
            100.0
            * total_bridge_count
            / number_examples
        ),
        "chain_em": (
            100.0
            * total_chain_em
            / number_examples
        ),
        "mean_predicted_bridges": (
            total_predicted_bridges
            / number_examples
        ),
        "maximum_prompt_tokens": (
            maximum_found
        ),
    }

    return metrics, detailed


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

    fieldnames: List[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def print_metrics(
    run_name: str,
    split: str,
    metrics: Dict,
) -> None:
    print(
        f"{run_name} / {split}: "
        f"N={metrics['n']} "
        f"EM={metrics['em']:.2f} "
        f"F1={metrics['f1']:.2f} "
        f"Format={metrics['answer_format_rate']:.2f} "
        f"BridgeR={metrics['bridge_recall']:.2f} "
        f"BridgeF1={metrics['bridge_f1']:.2f} "
        f"BridgeCount={metrics['bridge_count_accuracy']:.2f} "
        f"ChainEM={metrics['chain_em']:.2f} "
        f"PredBridges={metrics['mean_predicted_bridges']:.2f}"
    )


def free_model(
    model,
    tokenizer,
) -> None:
    if model is not None:
        del model

    if tokenizer is not None:
        del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def flatten_run_summaries(
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
                "run_name": summary[
                    "run_name"
                ],
                "model_dir": summary[
                    "model_dir"
                ],
                "base_key": experiment.get(
                    "base_key",
                    experiment.get(
                        "model_name",
                        "unknown",
                    ),
                ),
                "model_name": experiment.get(
                    "model_name",
                    "unknown",
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
                "context_mode": (
                    experiment.get(
                        "context_mode",
                        "gold",
                    )
                ),
                "split": split,
            }

            row.update(metrics)
            rows.append(row)

    return rows


def aggregate_result_rows(
    rows: List[Dict],
) -> List[Dict]:
    groups = defaultdict(list)

    for row in rows:
        key = (
            row["base_key"],
            row["model_name"],
            row["target_style"],
            row["prompt_style"],
            row["context_mode"],
            row["split"],
        )

        groups[key].append(row)

    aggregated: List[Dict] = []

    for key, group_rows in sorted(
        groups.items()
    ):
        (
            base_key,
            model_name,
            target_style,
            prompt_style,
            context_mode,
            split,
        ) = key

        output = {
            "base_key": base_key,
            "model_name": model_name,
            "target_style": target_style,
            "prompt_style": prompt_style,
            "context_mode": context_mode,
            "split": split,
            "number_runs": len(
                group_rows
            ),
        }

        for metric in METRIC_NAMES:
            values = [
                float(row[metric])
                for row in group_rows
                if row.get(metric)
                is not None
            ]

            if not values:
                output[
                    f"{metric}_mean"
                ] = None
                output[
                    f"{metric}_std"
                ] = None
                continue

            output[
                f"{metric}_mean"
            ] = statistics.mean(
                values
            )

            output[
                f"{metric}_std"
            ] = (
                statistics.pstdev(
                    values
                )
                if len(values) > 1
                else 0.0
            )

        aggregated.append(output)

    return aggregated


def build_contrasts(
    rows: List[Dict],
) -> List[Dict]:
    index = {}

    for row in rows:
        key = (
            row["base_key"],
            int(row["seed"]),
            row["target_style"],
            row["prompt_style"],
            row["split"],
        )

        index[key] = row

    base_seed_split = sorted(
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

    for (
        base_key,
        seed,
        split,
    ) in base_seed_split:
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

                for metric in METRIC_NAMES:
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

                for metric in METRIC_NAMES:
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


def aggregate_contrasts(
    contrasts: List[Dict],
) -> List[Dict]:
    groups = defaultdict(list)

    for row in contrasts:
        key = (
            row["contrast_type"],
            row["base_key"],
            row["split"],
            row["fixed_factor"],
            row["fixed_value"],
            row["condition_a"],
            row["condition_b"],
        )

        groups[key].append(row)

    aggregated: List[Dict] = []

    for key, group_rows in sorted(
        groups.items()
    ):
        (
            contrast_type,
            base_key,
            split,
            fixed_factor,
            fixed_value,
            condition_a,
            condition_b,
        ) = key

        output = {
            "contrast_type": (
                contrast_type
            ),
            "base_key": base_key,
            "split": split,
            "fixed_factor": fixed_factor,
            "fixed_value": fixed_value,
            "condition_a": condition_a,
            "condition_b": condition_b,
            "number_seeds": len(
                group_rows
            ),
        }

        for metric in METRIC_NAMES:
            field = f"delta_{metric}"

            values = [
                float(row[field])
                for row in group_rows
            ]

            output[
                f"{field}_mean"
            ] = statistics.mean(
                values
            )

            output[
                f"{field}_std"
            ] = (
                statistics.pstdev(
                    values
                )
                if len(values) > 1
                else 0.0
            )

        aggregated.append(output)

    return aggregated


def print_final_tables(
    rows: List[Dict],
    contrasts: List[Dict],
) -> None:
    print("\n" + "=" * 132)
    print("FINAL MODEL RESULTS")
    print("=" * 132)

    for split in sorted(
        {row["split"] for row in rows}
    ):
        print(f"\n--- {split} ---")

        print(
            f"{'Base':<14}"
            f"{'Target':<15}"
            f"{'Prompt':<12}"
            f"{'Seed':>7}"
            f"{'EM':>9}"
            f"{'F1':>9}"
            f"{'BridgeR':>11}"
            f"{'BridgeCount':>14}"
            f"{'ChainEM':>11}"
        )

        print("-" * 107)

        split_rows = sorted(
            [
                row
                for row in rows
                if row["split"] == split
            ],
            key=lambda row: (
                row["base_key"],
                row["target_style"],
                row["prompt_style"],
                int(row["seed"]),
            ),
        )

        for row in split_rows:
            print(
                f"{row['base_key']:<14}"
                f"{row['target_style']:<15}"
                f"{row['prompt_style']:<12}"
                f"{int(row['seed']):>7}"
                f"{row['em']:>9.2f}"
                f"{row['f1']:>9.2f}"
                f"{row['bridge_recall']:>11.2f}"
                f"{row['bridge_count_accuracy']:>14.2f}"
                f"{row['chain_em']:>11.2f}"
            )

    print("\n" + "=" * 132)
    print("PAIRED CONTRASTS")
    print(
        "Positive delta means condition_b "
        "outperformed condition_a."
    )
    print("=" * 132)

    for row in contrasts:
        print(
            f"{row['contrast_type']:<20} "
            f"base={row['base_key']:<12} "
            f"split={row['split']:<13} "
            f"{row['fixed_factor']}="
            f"{row['fixed_value']:<13} "
            f"{row['condition_b']} - "
            f"{row['condition_a']}: "
            f"ΔEM={row['delta_em']:+.2f}, "
            f"ΔF1={row['delta_f1']:+.2f}, "
            f"ΔBridgeR="
            f"{row['delta_bridge_recall']:+.2f}"
        )


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        args.device
    )

    dtype = torch_dtype_from_name(
        args.dtype
    )

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

    all_summaries: List[Dict] = []
    failures: List[Dict] = []

    for model_dir_string in args.model_dirs:
        model_dir = Path(
            model_dir_string
        ).expanduser().resolve()

        model = None
        tokenizer = None

        try:
            if not model_dir.exists():
                raise FileNotFoundError(
                    model_dir
                )

            experiment = (
                load_experiment_config(
                    model_dir
                )
            )

            required_fields = [
                "target_style",
                "prompt_style",
                "context_mode",
                "model_name",
            ]

            missing_fields = [
                field
                for field in required_fields
                if field not in experiment
            ]

            if missing_fields:
                raise ValueError(
                    "Missing experiment config "
                    f"fields for {model_dir}: "
                    f"{missing_fields}"
                )

            run_name = experiment.get(
                "run_name",
                infer_run_name(model_dir),
            )

            run_out_dir = (
                out_dir / run_name
            )

            run_out_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            summary_path = (
                run_out_dir
                / "summary.json"
            )

            existing_results: Dict = {}

            if (
                args.resume
                and summary_path.exists()
            ):
                with summary_path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    existing_summary = (
                        json.load(file)
                    )

                existing_results = dict(
                    existing_summary.get(
                        "results",
                        {},
                    )
                )

            requested_splits = [
                split
                for split in args.splits
                if split in EVAL_FILES
            ]

            missing_splits = [
                split
                for split in requested_splits
                if split
                not in existing_results
            ]

            print("\n" + "#" * 100)
            print(f"Run          : {run_name}")
            print(f"Model dir    : {model_dir}")
            print(
                f"Base          : "
                f"{experiment.get('base_key')}"
            )
            print(
                f"Target style  : "
                f"{experiment['target_style']}"
            )
            print(
                f"Prompt style  : "
                f"{experiment['prompt_style']}"
            )
            print(
                f"Context mode  : "
                f"{experiment['context_mode']}"
            )
            print(
                f"Missing splits: "
                f"{missing_splits}"
            )
            print("#" * 100)

            if missing_splits:
                model, tokenizer = (
                    load_model_and_tokenizer(
                        model_dir=model_dir,
                        device=device,
                        dtype=dtype,
                        attn_implementation=(
                            args.attn_implementation
                        ),
                        local_files_only=(
                            args.local_files_only
                        ),
                    )
                )

            for split in missing_splits:
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
                    f"\nEvaluating {split}: "
                    f"{len(examples)} examples"
                )

                metrics, detailed = (
                    evaluate_split(
                        model=model,
                        tokenizer=tokenizer,
                        examples=examples,
                        target_style=(
                            experiment[
                                "target_style"
                            ]
                        ),
                        prompt_style=(
                            experiment[
                                "prompt_style"
                            ]
                        ),
                        context_mode=(
                            experiment[
                                "context_mode"
                            ]
                        ),
                        batch_size=(
                            args.batch_size
                        ),
                        max_input_length=(
                            args.max_input_length
                        ),
                        max_new_tokens=(
                            args.max_new_tokens
                        ),
                        device=device,
                    )
                )

                print_metrics(
                    run_name=run_name,
                    split=split,
                    metrics=metrics,
                )

                existing_results[
                    split
                ] = metrics

                write_jsonl(
                    run_out_dir
                    / (
                        f"{split}_"
                        "predictions.jsonl"
                    ),
                    detailed,
                )

                partial_summary = {
                    "run_name": run_name,
                    "model_dir": str(
                        model_dir
                    ),
                    "experiment": (
                        experiment
                    ),
                    "results": (
                        existing_results
                    ),
                }

                write_json(
                    summary_path,
                    partial_summary,
                )

            run_summary = {
                "run_name": run_name,
                "model_dir": str(model_dir),
                "experiment": experiment,
                "results": existing_results,
            }

            write_json(
                summary_path,
                run_summary,
            )

            all_summaries.append(
                run_summary
            )

        except Exception as error:
            failure = {
                "model_dir": (
                    str(model_dir)
                ),
                "error": repr(error),
                "traceback": (
                    traceback.format_exc()
                ),
            }

            failures.append(failure)

            print("\n" + "!" * 100)
            print(
                f"EVALUATION FAILED: "
                f"{model_dir}"
            )
            print(failure["traceback"])
            print("!" * 100)

        finally:
            free_model(
                model,
                tokenizer,
            )

    flat_rows = flatten_run_summaries(
        all_summaries
    )

    aggregate_rows = (
        aggregate_result_rows(
            flat_rows
        )
    )

    contrasts = build_contrasts(
        flat_rows
    )

    aggregate_contrast_rows = (
        aggregate_contrasts(
            contrasts
        )
    )

    write_json(
        out_dir / "all_results.json",
        {
            "runs": all_summaries,
            "failures": failures,
        },
    )

    write_csv(
        out_dir / "all_results.csv",
        flat_rows,
    )

    write_csv(
        out_dir / "aggregate_results.csv",
        aggregate_rows,
    )

    write_csv(
        out_dir / "contrasts.csv",
        contrasts,
    )

    write_csv(
        out_dir
        / "aggregate_contrasts.csv",
        aggregate_contrast_rows,
    )

    write_json(
        out_dir / "failures.json",
        failures,
    )

    print_final_tables(
        rows=flat_rows,
        contrasts=contrasts,
    )

    print("\nResults saved to:")
    print(f"  {out_dir}")
    print(
        "Main tables:\n"
        f"  {out_dir / 'all_results.csv'}\n"
        f"  {out_dir / 'aggregate_results.csv'}\n"
        f"  {out_dir / 'contrasts.csv'}\n"
        f"  {out_dir / 'aggregate_contrasts.csv'}"
    )

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()