"""
mine_hard_examples.py  (v2: sampling-based + easy-sample mixing)

关键改进:
- 每个 example 用 sampling 采 N 次 (与 RL 的 num_generations 一致)
- 用 mean_reward 分类 hard / easy (而不是 greedy 单次)
- 混入 easy 样本: 保证 min_easy_ratio + 凑够 target_size
"""

import argparse
import json
import os
import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

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


def format_user_content(example, include_bridge_count=False):
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
# QA metrics
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
# Reward (与 rl_grpo_2hop.py 严格一致)
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


def compute_reward(text, example, reward_mode):
    parsed = parse_output(text)
    gold_bridges = example.get("bridges", [])
    final_golds = get_final_golds(example)

    bridge_score = compute_bridge_score(parsed.bridges, gold_bridges)
    answer_f1 = f1_score(parsed.answer, final_golds)
    answer_em = em_score(parsed.answer, final_golds)
    format_score = compute_format_score(parsed, len(gold_bridges))

    if reward_mode == "chain_binary":
        bridge_pass = float(bridge_score >= 0.5)
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
        final_effective = answer_f1 * bridge_score
        reward = 0.10 * format_score + 0.45 * bridge_score + 0.45 * final_effective
    elif reward_mode == "final_only":
        reward = 0.15 * format_score + 0.85 * answer_f1
    else:
        raise ValueError(f"Unsupported reward_mode: {reward_mode}")

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

    # Reward
    parser.add_argument("--reward_mode", default="chain_binary",
                        choices=["chain_binary", "process_final", "final_only"])
    parser.add_argument("--include_bridge_count", action="store_true")

    # Sampling (mining 时用 sampling)
    parser.add_argument("--num_samples", type=int, default=4,
                        help="每个样本采几次")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)

    # Hard / Easy 分类
    parser.add_argument("--reward_threshold", type=float, default=0.85,
                        help="mean_reward < threshold → hard")

    # Easy 混合策略
    parser.add_argument("--target_size", type=int, default=5000,
                        help="希望最终数据集大小. 0 = 不强制")
    parser.add_argument("--min_easy_ratio", type=float, default=0.2,
                        help="最终 easy 样本占比下限 (0~1)")

    # Misc
    parser.add_argument("--batch_size", type=int, default=8,
                        help="prompt 数 (每个 prompt 会被采样 num_samples 次)")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_input_len", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Load model ----
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

    # ---- Load data ----
    examples = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if args.limit:
        examples = examples[: args.limit]
    print(f"Loaded {len(examples)} examples from {args.input_file}")

    # ---- Build prompts ----
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

    # storage 按 sorted 顺序
    sorted_rewards_per_ex = [[] for _ in range(len(prompts))]
    sorted_outputs_per_ex = [[] for _ in range(len(prompts))]
    sorted_metas_per_ex = [[] for _ in range(len(prompts))]

    desc = f"Mining (N={args.num_samples}, T={args.temperature})"
    for i in tqdm(range(0, len(sorted_prompts), args.batch_size), desc=desc):
        batch_prompts = sorted_prompts[i : i + args.batch_size]
        batch_examples = sorted_examples[i : i + args.batch_size]
        n_in_batch = len(batch_prompts)

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
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.num_samples,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_id,
                use_cache=True,
            )

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = out[:, prompt_len:]
        # generate 输出顺序是: ex0_s0, ex0_s1, ..., ex0_s(N-1), ex1_s0, ...
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for j in range(n_in_batch):
            ex = batch_examples[j]
            for k in range(args.num_samples):
                idx = j * args.num_samples + k
                txt = texts[idx].strip()
                r, meta = compute_reward(txt, ex, reward_mode=args.reward_mode)
                sorted_rewards_per_ex[i + j].append(r)
                sorted_outputs_per_ex[i + j].append(txt)
                sorted_metas_per_ex[i + j].append(meta)

    # 还原原顺序
    rewards_per_ex = [None] * len(examples)
    outputs_per_ex = [None] * len(examples)
    metas_per_ex = [None] * len(examples)
    for sorted_i, orig_i in enumerate(order):
        rewards_per_ex[orig_i] = sorted_rewards_per_ex[sorted_i]
        outputs_per_ex[orig_i] = sorted_outputs_per_ex[sorted_i]
        metas_per_ex[orig_i] = sorted_metas_per_ex[sorted_i]

    # 聚合
    mean_rewards = [sum(rs) / len(rs) for rs in rewards_per_ex]
    success_rates = [
        sum(1 for r in rs if r >= 0.95) / len(rs) for rs in rewards_per_ex
    ]
    group_stds = [
        (sum((r - sum(rs)/len(rs))**2 for r in rs) / len(rs)) ** 0.5
        for rs in rewards_per_ex
    ]

    # 分类
    hard_indices = [i for i, mr in enumerate(mean_rewards) if mr < args.reward_threshold]
    easy_indices = [i for i, mr in enumerate(mean_rewards) if mr >= args.reward_threshold]

    # 计算要补多少 easy
    n_hard = len(hard_indices)
    if args.target_size > 0:
        n_for_target = max(0, args.target_size - n_hard)
    else:
        n_for_target = 0
    if 0 < args.min_easy_ratio < 1:
        # 最终 easy/(hard+easy) ≥ min_easy_ratio
        # easy ≥ min_easy_ratio * (hard + easy)
        # easy*(1 - r) ≥ r*hard  →  easy ≥ r*hard / (1-r)
        n_for_ratio = int(round(n_hard * args.min_easy_ratio / (1 - args.min_easy_ratio)))
    else:
        n_for_ratio = 0
    n_easy_to_add = max(n_for_target, n_for_ratio)
    n_easy_to_add = min(n_easy_to_add, len(easy_indices))

    sampled_easy = random.sample(easy_indices, n_easy_to_add) if n_easy_to_add > 0 else []
    final_indices = hard_indices + sampled_easy
    random.shuffle(final_indices)

    # ============ Print stats ============
    print(f"\n{'='*70}")
    print(f"Mining Result")
    print(f"{'='*70}")
    print(f"Model           : {args.model_dir}")
    print(f"Num samples / ex: {args.num_samples} (T={args.temperature}, top_p={args.top_p})")
    print(f"Reward mode     : {args.reward_mode}")
    print(f"Hard threshold  : mean_reward < {args.reward_threshold}")
    print()
    print(f"Total examples           : {len(examples)}")
    print(f"  Hard (mean_r < {args.reward_threshold:.2f})   : "
          f"{n_hard}  ({n_hard/len(examples)*100:.1f}%)")
    print(f"  Easy (mean_r >= {args.reward_threshold:.2f}) : "
          f"{len(easy_indices)}  ({len(easy_indices)/len(examples)*100:.1f}%)")
    print(f"Mean of mean_rewards     : {sum(mean_rewards)/len(mean_rewards):.4f}")
    print(f"Mean success_rate        : {sum(success_rates)/len(success_rates):.4f}")
    print(f"Mean group_std (signal)  : {sum(group_stds)/len(group_stds):.4f}")

    # 直方图
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0001]
    hist = [0] * (len(bins) - 1)
    for mr in mean_rewards:
        for k in range(len(bins) - 1):
            if bins[k] <= mr < bins[k + 1]:
                hist[k] += 1
                break
    max_h = max(hist) if max(hist) > 0 else 1
    print("\nMean reward distribution:")
    for k in range(len(bins) - 1):
        bar = "#" * int(hist[k] / max_h * 50)
        print(f"  [{bins[k]:.2f}, {bins[k+1]:.2f}): {hist[k]:6d}  {bar}")

    print(f"\nMixing:")
    print(f"  Hard kept                 : {n_hard}")
    print(f"  Easy needed (target_size) : {n_for_target}")
    print(f"  Easy needed (min_ratio)   : {n_for_ratio}")
    print(f"  Easy actually added       : {n_easy_to_add}")
    print(f"  → Final dataset size      : {len(final_indices)}")
    if len(final_indices) > 0:
        print(f"  → Final hard ratio        : "
              f"{n_hard/len(final_indices)*100:.1f}%")
        print(f"  → Final easy ratio        : "
              f"{n_easy_to_add/len(final_indices)*100:.1f}%")

    # ============ Write files ============
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    # 主输出文件: 带上 mining metadata (RL 会忽略未知字段)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for i in final_indices:
            rec = dict(examples[i])
            rec["_mine_mean_reward"] = float(mean_rewards[i])
            rec["_mine_success_rate"] = float(success_rates[i])
            rec["_mine_group_std"] = float(group_stds[i])
            rec["_mine_is_hard"] = bool(mean_rewards[i] < args.reward_threshold)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n✓ Final dataset → {args.output_file}  ({len(final_indices)} examples)")

    # debug: hard 样本
    debug_dir = os.path.dirname(args.output_file) or "."
    base = os.path.basename(args.output_file).replace(".jsonl", "")

    debug_hard = os.path.join(debug_dir, f"{base}_debug_hard.jsonl")
    with open(debug_hard, "w", encoding="utf-8") as f:
        for i in random.sample(hard_indices, min(100, len(hard_indices))):
            rec = dict(examples[i])
            rec["_mine_mean_reward"] = float(mean_rewards[i])
            rec["_mine_success_rate"] = float(success_rates[i])
            rec["_mine_group_std"] = float(group_stds[i])
            rec["_mine_outputs"] = outputs_per_ex[i]
            rec["_mine_rewards_per_sample"] = [float(r) for r in rewards_per_ex[i]]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"✓ Hard debug (100 with full outputs) → {debug_hard}")

    debug_easy = os.path.join(debug_dir, f"{base}_debug_easy.jsonl")
    with open(debug_easy, "w", encoding="utf-8") as f:
        for i in random.sample(easy_indices, min(100, len(easy_indices))):
            rec = dict(examples[i])
            rec["_mine_mean_reward"] = float(mean_rewards[i])
            rec["_mine_success_rate"] = float(success_rates[i])
            rec["_mine_outputs"] = outputs_per_ex[i][:2]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"✓ Easy debug (100 with first 2 outputs) → {debug_easy}")


if __name__ == "__main__":
    main()