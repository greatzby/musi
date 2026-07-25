#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare MuSiQue-Ans for matched answer-only and bridge-aware experiments.

Training:
  - original 2-hop examples from train

Evaluation:
  - 2-hop from dev
  - 3hop1 linear from dev
  - 4hop1 linear from dev

Each processed example contains:
  - context_paragraphs: gold supporting paragraphs only
  - all_context_paragraphs: every original paragraph
  - bridges
  - final_answer
  - answer_aliases
  - gold_chain

The same files are used for both answer-only and bridge-aware SFT.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--musique_dir",
        type=str,
        default="./data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./prepared_data_2hop",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    items: List[Dict] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    return items


def dump_jsonl(items: List[Dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )

    print(f"Wrote {len(items):>6} examples -> {path}")


def parse_hop_id(example: Dict) -> Tuple[int, str]:
    match = re.match(
        r"(\d+)hop(\d*)__",
        str(example["id"]),
    )

    if match is None:
        raise ValueError(
            f"Cannot parse hop from id: {example['id']}"
        )

    return int(match.group(1)), match.group(2)


def is_linear_chain(example: Dict) -> bool:
    hop, variant = parse_hop_id(example)

    if hop == 2:
        return True

    return variant == "1"


def resolve_references(
    subquestion: str,
    previous_answers: List[str],
) -> str:
    def replacement(match: re.Match) -> str:
        index = int(match.group(1)) - 1

        if 0 <= index < len(previous_answers):
            return previous_answers[index]

        return match.group(0)

    return re.sub(
        r"#(\d+)",
        replacement,
        subquestion,
    )


def get_supporting_paragraph(
    example: Dict,
    paragraph_index: int,
) -> Optional[Dict]:
    for paragraph in example.get("paragraphs", []):
        if int(paragraph["idx"]) == int(paragraph_index):
            return paragraph

    return None


def normalize_paragraph(paragraph: Dict) -> Dict:
    return {
        "idx": int(paragraph["idx"]),
        "title": str(paragraph.get("title", "")),
        "text": str(
            paragraph.get(
                "paragraph_text",
                paragraph.get("text", ""),
            )
        ),
    }


def build_processed_example(
    example: Dict,
) -> Optional[Dict]:
    hop, variant = parse_hop_id(example)
    decomposition = example.get(
        "question_decomposition",
        [],
    )

    if len(decomposition) != hop:
        return None

    gold_paragraphs: List[Dict] = []
    seen_indices = set()

    for step in decomposition:
        support_index = int(
            step["paragraph_support_idx"]
        )

        paragraph = get_supporting_paragraph(
            example,
            support_index,
        )

        if paragraph is None:
            return None

        if support_index not in seen_indices:
            gold_paragraphs.append(
                normalize_paragraph(paragraph)
            )
            seen_indices.add(support_index)

    all_paragraphs = [
        normalize_paragraph(paragraph)
        for paragraph in example.get("paragraphs", [])
    ]

    previous_answers: List[str] = []
    gold_chain: List[Dict] = []

    for step in decomposition:
        raw_question = str(step["question"])
        step_answer = str(step["answer"])

        gold_chain.append(
            {
                "step_idx": int(step["id"]),
                "subq_raw": raw_question,
                "subq": resolve_references(
                    raw_question,
                    previous_answers,
                ),
                "answer": step_answer,
                "support_idx": int(
                    step["paragraph_support_idx"]
                ),
            }
        )

        previous_answers.append(step_answer)

    bridges = [
        str(step["answer"])
        for step in decomposition[:-1]
    ]

    final_answer = str(
        example.get(
            "answer",
            decomposition[-1]["answer"],
        )
    )

    return {
        "id": str(example["id"]),
        "hop": int(hop),
        "hop_variant": variant,
        "is_linear": bool(is_linear_chain(example)),
        "question": str(example["question"]),

        # Backward-compatible name: gold supporting context.
        "context_paragraphs": gold_paragraphs,
        "gold_context_paragraphs": gold_paragraphs,

        # Standard setting with original distractors.
        "all_context_paragraphs": all_paragraphs,

        "supporting_paragraph_indices": sorted(
            int(index) for index in seen_indices
        ),
        "bridges": bridges,
        "final_answer": final_answer,
        "answer_aliases": [
            str(alias)
            for alias in example.get(
                "answer_aliases",
                [],
            )
            if alias
        ],
        "gold_chain": gold_chain,
    }


def print_distribution(
    name: str,
    examples: List[Dict],
) -> None:
    counts = defaultdict(int)

    for example in examples:
        hop, variant = parse_hop_id(example)
        key = f"{hop}hop{variant}" if variant else f"{hop}hop"
        counts[key] += 1

    print(f"{name}: {dict(sorted(counts.items()))}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    musique_dir = Path(
        args.musique_dir
    ).expanduser().resolve()

    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = (
        musique_dir
        / "musique_ans_v1.0_train.jsonl"
    )
    dev_path = (
        musique_dir
        / "musique_ans_v1.0_dev.jsonl"
    )

    if not train_path.exists():
        raise FileNotFoundError(train_path)

    if not dev_path.exists():
        raise FileNotFoundError(dev_path)

    print("Loading raw MuSiQue-Ans...")
    train_raw = load_jsonl(train_path)
    dev_raw = load_jsonl(dev_path)

    print(f"Train raw: {len(train_raw)}")
    print(f"Dev raw  : {len(dev_raw)}")

    print("\nStructure distributions:")
    print_distribution("train", train_raw)
    print_distribution("dev", dev_raw)

    # Training: original 2-hop only.
    train_2hop: List[Dict] = []
    skipped_train = 0

    for example in train_raw:
        hop, _ = parse_hop_id(example)

        if hop != 2:
            continue

        processed = build_processed_example(example)

        if processed is None:
            skipped_train += 1
            continue

        train_2hop.append(processed)

    random.Random(args.seed).shuffle(train_2hop)

    # Evaluation.
    dev_processed: List[Dict] = []
    skipped_dev = 0

    for example in dev_raw:
        processed = build_processed_example(example)

        if processed is None:
            skipped_dev += 1
            continue

        dev_processed.append(processed)

    eval_2hop = [
        example
        for example in dev_processed
        if example["hop"] == 2
    ]

    eval_3hop_linear = [
        example
        for example in dev_processed
        if (
            example["hop"] == 3
            and example["is_linear"]
        )
    ]

    eval_4hop_linear = [
        example
        for example in dev_processed
        if (
            example["hop"] == 4
            and example["is_linear"]
        )
    ]

    dump_jsonl(
        train_2hop,
        output_dir / "train_2hop.jsonl",
    )
    dump_jsonl(
        eval_2hop,
        output_dir / "eval_2hop.jsonl",
    )
    dump_jsonl(
        eval_3hop_linear,
        output_dir / "eval_3hop_linear.jsonl",
    )
    dump_jsonl(
        eval_4hop_linear,
        output_dir / "eval_4hop_linear.jsonl",
    )

    metadata = {
        "seed": args.seed,
        "train_raw": len(train_raw),
        "dev_raw": len(dev_raw),
        "train_2hop": len(train_2hop),
        "eval_2hop": len(eval_2hop),
        "eval_3hop_linear": len(
            eval_3hop_linear
        ),
        "eval_4hop_linear": len(
            eval_4hop_linear
        ),
        "skipped_train": skipped_train,
        "skipped_dev": skipped_dev,
    }

    with (
        output_dir / "metadata.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("Preparation complete")
    print(f"Train 2-hop       : {len(train_2hop)}")
    print(f"Eval 2-hop        : {len(eval_2hop)}")
    print(f"Eval 3-hop linear : {len(eval_3hop_linear)}")
    print(f"Eval 4-hop linear : {len(eval_4hop_linear)}")
    print(f"Skipped train/dev : {skipped_train}/{skipped_dev}")
    print(f"Output directory  : {output_dir}")
    print("=" * 70)

    if train_2hop:
        example = train_2hop[0]
        print("\nExample:")
        print("ID:", example["id"])
        print("Question:", example["question"])
        print("Bridges:", example["bridges"])
        print("Answer:", example["final_answer"])
        print(
            "Gold/all paragraphs:",
            len(example["gold_context_paragraphs"]),
            len(example["all_context_paragraphs"]),
        )


if __name__ == "__main__":
    main()