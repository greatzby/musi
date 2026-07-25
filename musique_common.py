#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


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


# 该输出格式在 answer-only / bridge-aware 之间完全相同。
# 唯一不同之处是训练 target 中是否真的包含 Bridge 行。
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


ANCHORED_INSTRUCTION = (
    "Use only information from the supplied passages to answer the exact "
    "target question.\n"
    "The question labeled 'Question:' is the goal. Keep that exact goal in "
    "focus while producing any intermediate bridges. Intermediate bridge "
    "answers are only steps and must not replace the requested final answer.\n"
    "Before stopping, verify that the final Answer line directly answers the "
    "exact target question."
)


def load_jsonl(path: Path) -> List[Dict]:
    examples: List[Dict] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                examples.append(json.loads(line))

    return examples


def install_chat_template_if_needed(tokenizer) -> str:
    """
    Return:
        existing
        installed_qwen
        installed_llama
    """
    if tokenizer.chat_template:
        return "existing"

    vocabulary = tokenizer.get_vocab()

    if (
        "<|im_start|>" in vocabulary
        and "<|im_end|>" in vocabulary
    ):
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE
        return "installed_qwen"

    if (
        "<|start_header_id|>" in vocabulary
        and "<|end_header_id|>" in vocabulary
        and "<|eot_id|>" in vocabulary
    ):
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
        return "installed_llama"

    raise RuntimeError(
        "Tokenizer has no chat template and its vocabulary does not "
        "look like either Qwen ChatML or Llama-3. Please install an "
        "appropriate chat template explicitly."
    )


def ensure_pad_token(tokenizer) -> int:
    """
    Ensure that a pad token exists.

    Returns:
        Number of newly added vocabulary tokens.
    """
    if tokenizer.pad_token_id is not None:
        return 0

    vocabulary = tokenizer.get_vocab()

    preferred_tokens = [
        "<|finetune_right_pad_id|>",
        "<|endoftext|>",
        "<|end_of_text|>",
    ]

    for token in preferred_tokens:
        if token in vocabulary:
            tokenizer.pad_token = token
            return 0

    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return 0

    return int(
        tokenizer.add_special_tokens(
            {"pad_token": "<|pad|>"}
        )
    )


def build_system_prompt(prompt_style: str) -> str:
    if prompt_style == "canonical":
        instruction = CANONICAL_INSTRUCTION
    elif prompt_style == "anchored":
        instruction = ANCHORED_INSTRUCTION
    else:
        raise ValueError(
            f"Unknown prompt_style: {prompt_style}"
        )

    return f"{instruction}\n\n{COMMON_OUTPUT_SCHEMA}"


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
        f"Unknown context_mode: {context_mode}"
    )


def paragraph_to_text(
    paragraph: Dict,
    paragraph_number: int,
) -> str:
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

    return (
        f"Passage {paragraph_number}\n"
        f"Passage Title: {title}\n"
        f"Passage: {text}"
    )


def format_user_content(
    example: Dict,
    context_mode: str,
    prompt_style: str,
) -> str:
    paragraphs = get_context_paragraphs(
        example=example,
        context_mode=context_mode,
    )

    blocks = [
        paragraph_to_text(
            paragraph=paragraph,
            paragraph_number=index,
        )
        for index, paragraph in enumerate(
            paragraphs,
            start=1,
        )
    ]

    context = (
        "\n\n".join(blocks)
        if blocks
        else "(No passages provided.)"
    )

    question = str(example["question"])

    content = (
        f"{context}\n\n"
        f"Question: {question}"
    )

    if prompt_style == "anchored":
        content += (
            "\n\n"
            "Target reminder: the final Answer line must answer "
            "exactly this question:\n"
            f"{question}"
        )

    return content


def format_target(
    example: Dict,
    target_style: str,
) -> str:
    final_answer = str(
        example["final_answer"]
    )

    if target_style == "answer_only":
        return f"Answer: {final_answer}"

    if target_style == "bridge_aware":
        lines = [
            f"Bridge {index}: {bridge}"
            for index, bridge in enumerate(
                example.get("bridges", []),
                start=1,
            )
        ]

        lines.append(
            f"Answer: {final_answer}"
        )

        return "\n".join(lines)

    raise ValueError(
        f"Unknown target_style: {target_style}"
    )


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
                example=example,
                context_mode=context_mode,
                prompt_style=prompt_style,
            ),
        },
    ]


def build_full_messages(
    example: Dict,
    context_mode: str,
    prompt_style: str,
    target_style: str,
) -> List[Dict]:
    return (
        build_prompt_messages(
            example=example,
            context_mode=context_mode,
            prompt_style=prompt_style,
        )
        + [
            {
                "role": "assistant",
                "content": format_target(
                    example=example,
                    target_style=target_style,
                ),
            }
        ]
    )


def load_experiment_config(
    model_dir: Path,
) -> Dict:
    candidates = [
        model_dir / "experiment_config.json",
        model_dir.parent / "experiment_config.json",
        model_dir.parent.parent / "experiment_config.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            with candidate.open(
                "r",
                encoding="utf-8",
            ) as file:
                return json.load(file)

    return {}


def infer_run_name(model_dir: Path) -> str:
    if model_dir.name == "final":
        return model_dir.parent.name

    return model_dir.name


def effective_training_epochs(
    max_steps: int,
    global_batch_size: int,
    number_examples: int,
) -> float:
    if number_examples <= 0:
        return 0.0

    return (
        max_steps
        * global_batch_size
        / number_examples
    )