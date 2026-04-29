"""
mine_hard_examples.py

跑一遍 SFT checkpoint 在训练集上的 reward,
扔掉 reward >= threshold 的"easy"样本,
保留 hard examples 用于 GRPO.

用法:
  python mine_hard_examples.py \
      --model_dir checkpoints/qwen2.5-3b-2hop-sft/checkpoint-225 \
      --input_file prepared_data_2hop/train_2hop.jsonl \
      --output_file hard_data_from_225/train_2hop_hard.jsonl \
      --reward_threshold 0.95 \
      --reward_mode chain_binary \
      --include_bridge_count
"""

import argparse
import json
import os
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ============================================================
# Prompt (与 rl_grpo_2hop.py 严格一致)
# ============================================================

SYSTEM_PROMPT = (
    "You answer multi-step questions based strictly on the given passages.\n"
    "For each question:\n"
    "1. Identify each intermediate entity needed to reach the final answer "
    "(a \"bridge\"). List them in order, one per line, as "
    "'Bridge 1: ...', 'Bridge 2: ...', etc.\n"
    "2. On the final line, output 'Answer: <short final answer>'.\n"
    "Use only information from the passages. Keep answers short (a few words)."
)


def format_user_content(example: Dict, include_bridge_count: bool = False) -> str:
    parts = []
    for p in example["context_paragraphs"]:
        title = p.get("title", "")
        text = p.get("text", p.get("paragraph_text", ""))
        parts.append(f"Passage Title: {title}\nPassage: {text}")
    ctx = "\n\n".join(parts)
    question = example["question"]
    if include_bridge_count:
        n_bridge = len(example.get("bridges", []))
        count_hint = (
            f"\n\nThis question requires exactly {n_bridge} intermediate "
            f"bridge answer(s). Output exactly {n_bridge} bridge line(s), "
            f"then the final answer line."
        )
    else:
        count_hint = ""
    return f"{ctx}\n\nQuestion: {question}{count_hint}"


# ============================================================
# QA metrics (与 rl_grpo_2hop.py 一致)
# ============================================================

def normalize_answer(s):
    if s is None:
        return ""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _f1_pair(pred, gold):
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_toks)
    recall = num_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def f1_score(pred, golds):
    golds = [g for g in golds if g]
    if not golds:
        return 0.0
    return max(_f1_pair(pred, g) for g in golds)


def em_score(pred, golds):
    p = normalize_answer(pred)
    golds = [g for g in golds if g]
    if not golds:
        return 0.0
    return float(any(p == normalize_answer(g) for g in golds))


def get_final_golds(example):
    golds = []
    if example.get("final_answer"):
        golds.append(example["final_answer"])
    if example.get("answer"):
        golds.append(example["answer"])
    if example.get("answer_aliases"):
        golds.extend(example["answer_aliases"])
    return [g for g in golds if g] or [""]


# ============================================================
# Output parsing
# ============================================================

RE_BRIDGE = re.compile(r"^\s*Bridge\s*(\d+)?\s*[:：]\s*(.+?)\s*$", re.IGNORECASE)
RE_ANSWER = re.compile(r"^\s*Answer\s*[:：]\s*(.+?)\s*$", re.IGNORECASE)


def clean_field(s):
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r"\s*(<\|im_end\|>|<\|endoftext\|>)\s*$", "", s)
    return s.strip().strip("\"'").strip()


@dataclass
class ParsedOutput:
    bridges: List[str]
    answer: str
    has_bridge_line: bool
    has_answer_line: bool
    bridge_before_answer: bool
    raw_nonempty_lines: int


def parse_output(text):
    bridges, events, answer = [], [], ""
    last_nonempty = ""
    n_nonempty = 0
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        n_nonempty += 1
        last_nonempty = line
        mb = RE_BRIDGE.match(line)
        if mb:
            bridges.append(clean_field(mb.group(2)))
            events.append("bridge")
            continue
        ma = RE_ANSWER.match(line)
        if ma:
            if not answer:
                answer = clean_field(ma.group(1))
            events.append("answer")
            continue
    if not answer and last_nonempty:
        answer = clean_field(last_nonempty)
    has_bridge = len(bridges) > 0
    has_answer = "answer" in events
    bridge_before_answer = False
    if has_bridge and has_answer:
        bridge_before_answer = events.index("bridge") < events.index("answer")
    return ParsedOutput(bridges, answer, has_bridge, has_answer,
                        bridge_before_answer, n_nonempty)


# ============================================================
# Reward (与 rl_grpo_2hop.py 一致, 包含 chain_binary)
# ============================================================

def compute_bridge_score(pred_bridges, gold_bridges):
    if not gold_bridges:
        return 1.0
    if not pred_bridges:
        return 0.0
    scores = []
    for i, gold in enumerate(gold_bridges):
        ordered = _f1_pair(pred_bridges[i], gold) if i < len(pred_bridges) else 0.0
        any_pos = max(_f1_pair(p, gold) for p in pred_bridges)
        scores.append(0.8 * ordered + 0.2 * any_pos)
    return sum(scores) / len(scores)


def compute_format_score(parsed, n_gold_bridges):
    score = 0.0
    if parsed.has_bridge_line:
        score += 0.25
    if parsed.has_answer_line:
        score += 0.25
    if parsed.bridge_before_answer:
        score += 0.20
    if len(parsed.bridges) >= n_gold_bridges:
        score += 0.15
    ans_words = parsed.answer.split()
    if 0 < len(ans_words) <= 12:
        score += 0.10
    if parsed.raw_nonempty_lines <= max(n_gold_bridges + 3, 4):
        score += 0.05
    return min(score, 1.0)


def compute_reward(text, example, reward_mode,
                   w_format=0.10, w_bridge=0.45, w_final=0.45,
                   bridge_gate_floor=0.0):
    parsed = parse_output(text)
    gold_bridges = example.get("bridges", [])
    final_golds = get_final_golds(example)

    bridge_score = compute_bridge_score(parsed.bridges, gold_bridges)
    answer_f1 = f1_score(parsed.answer, final_golds)
    answer_em = em_score(parsed.answer, final_golds)
    format_score = compute_format_score(parsed, len(gold_bridges))

    if reward_mode == "chain_binary":
        bridge_threshold = 0.5
        bridge_pass = float(bridge_score >= bridge_threshold)
        chain_correct = float(answer_em == 1.0 and bridge_pass == 1.0)
        if chain_correct == 1.0:
            reward = 1.0
        else:
            partial = (
                0.10 * format_score
                + 0.45 * bridge_score
                + 0.30 * answer_f1 * bridge_score
            )
            reward = min(partial, 0.85)
    elif reward_mode == "process_final":
        final_effective = answer_f1 * (
            bridge_gate_floor + (1.0 - bridge_gate_floor) * bridge_score
        )
        denom = max(w_format + w_bridge + w_final, 1e-8)
        reward = (
            w_format * format_score
            + w_bridge * bridge_score
            + w_final * final_effective
        ) / denom
    elif reward_mode == "final_only":
        reward = 0.15 * format_score + 0.85 * answer_f1
    else:
        raise ValueError(f"Unsupported reward_mode for mining: {reward_mode}")

    reward = float(max(0.0, min(1.0, reward)))
    return reward, {
        "bridge_score": bridge_score,
        "answer_f1": answer_f1,
        "answer_em": answer_em,
        "format_score": format_score,
        "n_pred_bridge": len(parsed.bridges),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--reward_threshold", type=float, default=0.95,
                        help="reward < threshold 的样本会保留")
    parser.add_argument("--reward_mode", default="chain_binary",
                        choices=["chain_binary", "process_final", "final_only"])
    parser.add_argument("--include_bridge_count", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_input_len", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--keep_prob_for_easy", type=float, default=0.0,
                        help="给 easy 样本保留概率, 比如 0.1 = 10% easy 样本也保留")
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).cuda()
    model.eval()

    examples = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if args.limit:
        examples = examples[: args.limit]
    print(f"Loaded {len(examples)} examples from {args.input_file}")

    prompts = []
    for ex in examples:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user_content(ex, args.include_bridge_count)},
        ]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        ))

    # 按长度排序加速 padding
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    sorted_prompts = [prompts[i] for i in order]
    sorted_examples = [examples[i] for i in order]

    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    sorted_rewards, sorted_outputs, sorted_metas = [], [], []

    for i in tqdm(range(0, len(sorted_prompts), args.batch_size), desc="Mining"):
        batch_prompts = sorted_prompts[i : i + args.batch_size]
        batch_examples = sorted_examples[i : i + args.batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_len,
            add_special_tokens=False,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_id,
                use_cache=True,
            )

        prompt_len = inputs["input_ids"].shape[1]
        new = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)

        for ex, txt in zip(batch_examples, new):
            txt = txt.strip()
            r, meta = compute_reward(txt, ex, reward_mode=args.reward_mode)
            sorted_rewards.append(r)
            sorted_outputs.append(txt)
            sorted_metas.append(meta)

    # 还原顺序
    rewards = [None] * len(examples)
    outputs = [None] * len(examples)
    metas = [None] * len(examples)
    for rank, orig_idx in enumerate(order):
        rewards[orig_idx] = sorted_rewards[rank]
        outputs[orig_idx] = sorted_outputs[rank]
        metas[orig_idx] = sorted_metas[rank]

    # 划分 hard / easy
    import random
    random.seed(42)
    hard_examples, easy_examples = [], []
    for ex, r, out_text, meta in zip(examples, rewards, outputs, metas):
        record = dict(ex)
        record["_mine_reward"] = r
        record["_mine_output"] = out_text
        record["_mine_meta"] = meta
        if r < args.reward_threshold:
            hard_examples.append(record)
        elif random.random() < args.keep_prob_for_easy:
            hard_examples.append(record)  # 少量 easy 也保留, 防止灾难性遗忘
        else:
            easy_examples.append(record)

    # 统计
    print(f"\n{'='*60}")
    print(f"Mining Result (reward_mode={args.reward_mode})")
    print(f"{'='*60}")
    print(f"Total examples       : {len(examples)}")
    print(f"Hard (r < {args.reward_threshold:.2f})    : "
          f"{len(hard_examples)}  ({len(hard_examples)/len(examples)*100:.1f}%)")
    print(f"Easy (r >= {args.reward_threshold:.2f})   : "
          f"{len(easy_examples)}  ({len(easy_examples)/len(examples)*100:.1f}%)")
    print(f"Mean reward          : {sum(rewards)/len(rewards):.4f}")

    # 直方图
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001]
    hist = [0] * (len(bins) - 1)
    for r in rewards:
        for k in range(len(bins) - 1):
            if bins[k] <= r < bins[k + 1]:
                hist[k] += 1
                break
    max_h = max(hist) if max(hist) > 0 else 1
    print("\nReward distribution:")
    for k in range(len(bins) - 1):
        bar = "#" * int(hist[k] / max_h * 50)
        print(f"  [{bins[k]:.2f}, {bins[k+1]:.2f}): {hist[k]:6d}  {bar}")

    # 写文件
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for ex in hard_examples:
            ex_clean = {k: v for k, v in ex.items() if not k.startswith("_mine_")}
            f.write(json.dumps(ex_clean, ensure_ascii=False) + "\n")
    print(f"\n✓ Hard examples → {args.output_file}")

    debug_dir = os.path.dirname(args.output_file) or "."
    debug_file = os.path.join(
        debug_dir,
        os.path.basename(args.output_file).replace(".jsonl", "_easy_sample.jsonl")
    )
    with open(debug_file, "w", encoding="utf-8") as f:
        for ex in easy_examples[:200]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"✓ Easy sample (first 200, with reward info) → {debug_file}")


if __name__ == "__main__":
    main()