"""
eval_compositional.py

在 2-hop / 3-hop linear / 4-hop linear 上评估 bridge-aware 模型。

输出指标：
  - Final EM / F1     : 最终答案准确率（与 gold final_answer + aliases 比对）
  - Bridge Recall     : 模型输出的 bridges 覆盖了多少 gold bridges (lenient match)
  - Bridge F1         : 对每个 gold bridge，取它与所有 pred bridges 的最大 F1，再平均
  - Chain EM          : 全部 gold bridges 都被命中 + final answer EM 都正确

主图：x 轴 = bridge depth (1, 2, 3)，y 轴 = 各指标 → 看 OOD 退化曲线
"""

import json
import os
import re
import string
import argparse
from collections import Counter
from typing import List, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

DEFAULT_MODEL_DIR = "checkpoints/qwen2.5-3b-2hop-sft"
EVAL_DIR = "prepared_data_2hop"
EVAL_FILES = {
    "2hop":        "eval_2hop.jsonl",
    "3hop_linear": "eval_3hop_linear.jsonl",
    "4hop_linear": "eval_4hop_linear.jsonl",
}
SYSTEM_PROMPT = (
    "You answer multi-step questions based strictly on the given passages.\n"
    "For each question:\n"
    "1. Identify each intermediate entity needed to reach the final answer (a \"bridge\"). "
    "List them in order, one per line, as 'Bridge 1: ...', 'Bridge 2: ...', etc.\n"
    "2. On the final line, output 'Answer: <short final answer>'.\n"
    "Use only information from the passages. Keep answers short (a few words)."
)


# ============ SQuAD-style normalization ============
def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def em_score(pred: str, golds: List[str]) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds if g))


def _f1_pair(pred: str, gold: str) -> float:
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_toks)
    recall    = num_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def f1_score(pred: str, golds: List[str]) -> float:
    return max(_f1_pair(pred, g) for g in golds if g) if golds else 0.0


# ============ Output parsing ============
RE_BRIDGE = re.compile(r'^\s*Bridge\s*\d+\s*[:：]\s*(.+?)\s*$', re.IGNORECASE)
RE_ANSWER = re.compile(r'^\s*Answer\s*[:：]\s*(.+?)\s*$', re.IGNORECASE)


def parse_output(text: str) -> Tuple[List[str], str]:
    """
    解析模型输出。返回 (bridges_list, final_answer_str).
    支持任意数量的 Bridge N: 行。如果没找到 Answer:，把最后一行当 answer。
    """
    bridges = []
    answer = ""
    answer_found = False
    last_nonempty = ""
    for raw in text.strip().split('\n'):
        line = raw.strip()
        if not line:
            continue
        last_nonempty = line
        m_b = RE_BRIDGE.match(line)
        if m_b:
            bridges.append(m_b.group(1).strip())
            continue
        m_a = RE_ANSWER.match(line)
        if m_a:
            answer = m_a.group(1).strip()
            answer_found = True
            continue
    # fallback: 模型没按格式输出，把最后一行当 answer
    if not answer_found and last_nonempty:
        answer = last_nonempty
    return bridges, answer


# ============ Bridge metrics ============
def bridge_match(pred_bridge: str, gold_bridge: str) -> bool:
    """Lenient match: normalized 字符串相等 / 互为子串 / token F1 ≥ 0.7"""
    p = normalize_answer(pred_bridge)
    g = normalize_answer(gold_bridge)
    if not p or not g:
        return False
    if p == g:
        return True
    if g in p or p in g:
        return True
    if _f1_pair(pred_bridge, gold_bridge) >= 0.7:
        return True
    return False


def compute_bridge_metrics(pred_bridges: List[str], gold_bridges: List[str]) -> Dict[str, float]:
    """
    返回:
      bridge_recall : gold 中有多少被 pred 命中
      bridge_f1     : 对每个 gold bridge 取 max F1 vs pred_bridges，再平均
      bridge_hits   : 命中数（int，方便 chain_em 计算）
    """
    if not gold_bridges:
        return {"bridge_recall": 1.0, "bridge_f1": 1.0, "bridge_hits": 0,
                "n_gold": 0, "n_pred": len(pred_bridges)}
    if not pred_bridges:
        return {"bridge_recall": 0.0, "bridge_f1": 0.0, "bridge_hits": 0,
                "n_gold": len(gold_bridges), "n_pred": 0}

    # recall: 命中数 / gold 数
    hits = 0
    for g in gold_bridges:
        if any(bridge_match(p, g) for p in pred_bridges):
            hits += 1
    recall = hits / len(gold_bridges)

    # f1: 对每个 gold，取它与 pred 的最大 F1，再平均
    f1s = []
    for g in gold_bridges:
        best = max(_f1_pair(p, g) for p in pred_bridges)
        f1s.append(best)
    bridge_f1 = sum(f1s) / len(f1s)

    return {"bridge_recall": recall, "bridge_f1": bridge_f1, "bridge_hits": hits,
            "n_gold": len(gold_bridges), "n_pred": len(pred_bridges)}


# ============ Input formatting ============
def format_user_content(example: Dict) -> str:
    parts = []
    for p in example["context_paragraphs"]:
        title = p.get("title", "")
        text  = p.get("text", p.get("paragraph_text", ""))
        parts.append(f"Passage Title: {title}\nPassage: {text}")
    ctx = "\n\n".join(parts)
    return f"{ctx}\n\nQuestion: {example['question']}"


def get_final_golds(example: Dict) -> List[str]:
    golds = []
    if example.get("final_answer"):
        golds.append(example["final_answer"])
    if example.get("answer"):
        golds.append(example["answer"])
    if example.get("answer_aliases"):
        golds.extend(example["answer_aliases"])
    return [g for g in golds if g] or [""]


# ============ Generation ============
@torch.no_grad()
def generate_batch(model, tokenizer, prompts: List[str],
                   max_new_tokens: int, max_input_len: int) -> List[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_len,
        add_special_tokens=False,
    ).to(model.device)

    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=eos_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    prompt_len = inputs["input_ids"].shape[1]
    answers = []
    for seq in out:
        new_tokens = seq[prompt_len:]
        ans = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        answers.append(ans)
    return answers


# ============ Per-file evaluation ============
def evaluate_file(model, tokenizer, jsonl_path: str,
                  batch_size: int, max_new_tokens: int,
                  max_input_len: int, limit: int = None):
    examples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if limit:
        examples = examples[:limit]

    # 构造 prompts
    prompts = []
    for ex in examples:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": format_user_content(ex)},
        ]
        p = tokenizer.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True)
        prompts.append(p)

    # 按长度排序减少 padding
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    sorted_prompts = [prompts[i] for i in order]

    sorted_outputs = []
    for i in tqdm(range(0, len(sorted_prompts), batch_size),
                  desc=os.path.basename(jsonl_path)):
        batch = sorted_prompts[i:i+batch_size]
        sorted_outputs.extend(generate_batch(
            model, tokenizer, batch,
            max_new_tokens=max_new_tokens,
            max_input_len=max_input_len,
        ))

    # 还原顺序
    raw_outputs = [None] * len(prompts)
    for rank, orig_idx in enumerate(order):
        raw_outputs[orig_idx] = sorted_outputs[rank]

    # 计算指标
    em_total = 0.0
    f1_total = 0.0
    bridge_recall_total = 0.0
    bridge_f1_total = 0.0
    chain_em_total = 0.0
    detailed = []

    for ex, raw in zip(examples, raw_outputs):
        pred_bridges, pred_answer = parse_output(raw)
        gold_bridges = ex.get("bridges", [])
        final_golds  = get_final_golds(ex)

        em = em_score(pred_answer, final_golds)
        f1 = f1_score(pred_answer, final_golds)
        bm = compute_bridge_metrics(pred_bridges, gold_bridges)
        chain_em = float(em == 1.0 and bm["bridge_hits"] == bm["n_gold"])

        em_total += em
        f1_total += f1
        bridge_recall_total += bm["bridge_recall"]
        bridge_f1_total     += bm["bridge_f1"]
        chain_em_total      += chain_em

        detailed.append({
            "id":            ex.get("id", ""),
            "hop":           ex.get("hop"),
            "question":      ex.get("question", ""),
            "gold_bridges":  gold_bridges,
            "gold_final":    final_golds,
            "pred_raw":      raw,
            "pred_bridges":  pred_bridges,
            "pred_answer":   pred_answer,
            "em":            em,
            "f1":            f1,
            "bridge_recall": bm["bridge_recall"],
            "bridge_f1":     bm["bridge_f1"],
            "bridge_hits":   bm["bridge_hits"],
            "chain_em":      chain_em,
        })

    n = len(examples)
    return {
        "n":             n,
        "em":            em_total / n * 100,
        "f1":            f1_total / n * 100,
        "bridge_recall": bridge_recall_total / n * 100,
        "bridge_f1":     bridge_f1_total / n * 100,
        "chain_em":      chain_em_total / n * 100,
    }, detailed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",      type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--eval_dir",       type=str, default=EVAL_DIR)
    parser.add_argument("--out_dir",        type=str, default="eval_results_2hop")
    parser.add_argument("--batch_size",     type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)  # 4-hop 输出可能 ~80 tokens
    parser.add_argument("--max_input_len",  type=int, default=4096)
    parser.add_argument("--limit",          type=int, default=None,
                        help="每个 split 只评估前 N 条（调试用）")
    parser.add_argument("--splits", nargs="+",
                        default=["2hop", "3hop_linear", "4hop_linear"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # generate 必须 left padding

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).cuda()
    model.eval()

    summary = {}
    for split in args.splits:
        if split not in EVAL_FILES:
            print(f"[skip] unknown split: {split}")
            continue
        path = os.path.join(args.eval_dir, EVAL_FILES[split])
        if not os.path.exists(path):
            print(f"[skip] not found: {path}")
            continue
        print(f"\n=== {split} ===")
        metrics, detailed = evaluate_file(
            model, tokenizer, path,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            max_input_len=args.max_input_len,
            limit=args.limit,
        )
        print(f"  n={metrics['n']:>5}  "
              f"EM={metrics['em']:>5.2f}  F1={metrics['f1']:>5.2f}  "
              f"BridgeR={metrics['bridge_recall']:>5.2f}  "
              f"BridgeF1={metrics['bridge_f1']:>5.2f}  "
              f"ChainEM={metrics['chain_em']:>5.2f}")
        summary[split] = metrics

        with open(os.path.join(args.out_dir, f"{split}_predictions.jsonl"),
                  "w", encoding="utf-8") as f:
            for d in detailed:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # 汇总
    print("\n" + "=" * 90)
    header = f"{'Split':<14} {'N':>6} {'EM':>7} {'F1':>7} {'BridgeR':>9} {'BridgeF1':>10} {'ChainEM':>9}"
    print(header)
    print("-" * 90)
    for split, m in summary.items():
        print(f"{split:<14} {m['n']:>6} "
              f"{m['em']:>7.2f} {m['f1']:>7.2f} "
              f"{m['bridge_recall']:>9.2f} {m['bridge_f1']:>10.2f} "
              f"{m['chain_em']:>9.2f}")
    print("=" * 90)

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.out_dir}/")


if __name__ == "__main__":
    main()