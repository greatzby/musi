"""
prepare_data_2hop.py

把官方 MuSiQue-Ans 切成"原始 2-hop 作为 graph 1-hop analogue"的格式。

核心思路：
- MuSiQue 2-hop = bridge depth 1 = graph 上的 single-stage transition
- MuSiQue 3-hop linear = bridge depth 2 = graph gap=2
- MuSiQue 4-hop linear = bridge depth 3 = graph gap=3

输出：
- train_2hop.jsonl              主训练集 (only original 2-hop, with bridges)
- eval_2hop.jsonl               in-distribution 测试 (dev 上的 2-hop)
- eval_3hop_linear.jsonl        1-step OOD (3hop1 linear)
- eval_4hop_linear.jsonl        2-step OOD (4hop1 linear)

每条样本字段：
  id, hop, hop_variant, is_linear,
  question,                      # 原始 composed 问题
  context_paragraphs,            # 仅 gold supporting paragraphs
  bridges,                       # 中间答案列表 (N-1 个，对于 N-hop)
  final_answer,                  # 最终答案
  answer_aliases,
  gold_chain                     # 完整推理链（含子问题，用于细粒度分析）
"""

import json
import re
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

# ---------- 路径 ----------
MUSIQUE_DIR = Path("./data")
OUT_DIR     = Path("./prepared_data_2hop")
OUT_DIR.mkdir(exist_ok=True)

TRAIN_FILE = MUSIQUE_DIR / "musique_ans_v1.0_train.jsonl"
DEV_FILE   = MUSIQUE_DIR / "musique_ans_v1.0_dev.jsonl"


# ---------- 工具函数 ----------
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dump_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(items):>6} → {path.name}")


def parse_hop_id(example):
    """解析 hop 数和结构变体。'2hop__...' → (2, ''), '3hop1__...' → (3, '1')"""
    m = re.match(r"(\d+)hop(\d*)__", example["id"])
    if m is None:
        raise ValueError(f"Cannot parse hop from id: {example['id']}")
    return int(m.group(1)), m.group(2)


def is_linear_chain(example):
    """线性链 = 2hop, 3hop1, 4hop1 (S1→S2→...→SN, 没有汇聚)"""
    hop, var = parse_hop_id(example)
    if hop == 2:
        return True
    return var == "1"


def resolve_references(subq_text, prev_answers):
    """把子问题里的 #1, #2 替换成前面 step 的 answer"""
    def repl(match):
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(prev_answers):
            return prev_answers[idx]
        return match.group(0)
    return re.sub(r"#(\d+)", repl, subq_text)


def get_supporting_paragraph(example, idx):
    for p in example["paragraphs"]:
        if p["idx"] == idx:
            return p
    return None


# ---------- 转换逻辑 ----------
def build_bridge_aware_example(ex):
    """
    把 MuSiQue 原始样本转成 bridge-aware 格式。
    - bridges  = [step_1.answer, step_2.answer, ..., step_{N-1}.answer]
    - final    = step_N.answer (== ex.answer)
    - context  = 仅 gold supporting paragraphs，按 step 顺序排列
    
    返回 None 表示数据有缺失（应丢弃）。
    """
    hop, variant = parse_hop_id(ex)
    decomp = ex["question_decomposition"]
    if len(decomp) != hop:
        return None  # 数据异常

    # 收集 gold supporting paragraphs，按 step 顺序（保证 bridge 链可读）
    gold_paras = []
    seen_idx = set()
    for step in decomp:
        para = get_supporting_paragraph(ex, step["paragraph_support_idx"])
        if para is None:
            return None  # 缺少 supporting paragraph，丢弃
        if para["idx"] not in seen_idx:
            gold_paras.append(para)
            seen_idx.add(para["idx"])

    # 构造 bridges + gold_chain
    prev_answers = []
    chain = []
    for step in decomp:
        chain.append({
            "step_idx":    step["id"],
            "subq_raw":    step["question"],
            "subq":        resolve_references(step["question"], prev_answers),
            "answer":      step["answer"],
            "support_idx": step["paragraph_support_idx"],
        })
        prev_answers.append(step["answer"])

    # bridges = 除最后一个 step 外所有 step 的 answer
    bridges      = [step["answer"] for step in decomp[:-1]]
    final_answer = decomp[-1]["answer"]

    # 一致性检查：MuSiQue 的 ex.answer 应该等于最后一步 answer
    if final_answer != ex["answer"]:
        # 偶尔会有大小写差异，以 ex.answer 为准
        final_answer = ex["answer"]

    return {
        "id":             ex["id"],
        "hop":            hop,
        "hop_variant":    variant,
        "is_linear":      is_linear_chain(ex),
        "question":       ex["question"],
        "context_paragraphs": [
            {"title": p["title"], "text": p["paragraph_text"]}
            for p in gold_paras
        ],
        "bridges":        bridges,
        "final_answer":   final_answer,
        "answer_aliases": ex.get("answer_aliases", []),
        "gold_chain":     chain,
    }


# ---------- 主流程 ----------
def main():
    print("Loading raw MuSiQue-Ans...")
    train_raw = load_jsonl(TRAIN_FILE)
    dev_raw   = load_jsonl(DEV_FILE)
    print(f"  Train: {len(train_raw)}")
    print(f"  Dev  : {len(dev_raw)}")

    # --- 结构变体分布 ---
    print("\nStructure variant distribution:")
    for name, split in [("train", train_raw), ("dev", dev_raw)]:
        cnt = defaultdict(int)
        for ex in split:
            h, v = parse_hop_id(ex)
            key = f"{h}hop{v}" if v else f"{h}hop"
            cnt[key] += 1
        print(f"  {name}: {dict(sorted(cnt.items()))}")

    # ============ 1) Train: 只取 2-hop ============
    print("\n[1/4] Building train_2hop from train split (original 2-hop only)...")
    train_2hop_raw = [ex for ex in train_raw if parse_hop_id(ex)[0] == 2]
    print(f"  Found {len(train_2hop_raw)} raw 2-hop examples in train")

    train_2hop = []
    skipped = 0
    for ex in train_2hop_raw:
        out = build_bridge_aware_example(ex)
        if out is None:
            skipped += 1
            continue
        train_2hop.append(out)
    print(f"  Built {len(train_2hop)} valid 2-hop training examples (skipped {skipped})")
    random.shuffle(train_2hop)

    # ============ 2) Eval: 2hop / 3hop linear / 4hop linear from dev ============
    print("\n[2/4] Building dev evaluation sets...")
    dev_processed = []
    for ex in dev_raw:
        out = build_bridge_aware_example(ex)
        if out is not None:
            dev_processed.append(out)
    print(f"  Processed {len(dev_processed)} valid dev examples")

    # 按 hop / linear 分桶
    eval_2hop        = [x for x in dev_processed if x["hop"] == 2]
    eval_3hop_linear = [x for x in dev_processed if x["hop"] == 3 and x["is_linear"]]
    eval_4hop_linear = [x for x in dev_processed if x["hop"] == 4 and x["is_linear"]]

    print(f"  2-hop (in-dist)         : {len(eval_2hop)}")
    print(f"  3-hop linear (1-step OOD): {len(eval_3hop_linear)}")
    print(f"  4-hop linear (2-step OOD): {len(eval_4hop_linear)}")

    # ============ 3) 写盘 ============
    print("\n[3/4] Writing files...")
    dump_jsonl(train_2hop,        OUT_DIR / "train_2hop.jsonl")
    dump_jsonl(eval_2hop,         OUT_DIR / "eval_2hop.jsonl")
    dump_jsonl(eval_3hop_linear,  OUT_DIR / "eval_3hop_linear.jsonl")
    dump_jsonl(eval_4hop_linear,  OUT_DIR / "eval_4hop_linear.jsonl")

    # ============ 4) 健康检查 / 样本预览 ============
    print("\n[4/4] Sanity check (first 2 training examples):")
    for i, ex in enumerate(train_2hop[:2]):
        print(f"\n  --- Example {i+1} ---")
        print(f"  id        : {ex['id']}")
        print(f"  question  : {ex['question']}")
        print(f"  bridges   : {ex['bridges']}")
        print(f"  final_ans : {ex['final_answer']}")
        print(f"  #paragraphs: {len(ex['context_paragraphs'])}")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  train_2hop.jsonl              : {len(train_2hop):>6}  (主训练集)")
    print(f"  eval_2hop.jsonl               : {len(eval_2hop):>6}  (in-dist)")
    print(f"  eval_3hop_linear.jsonl        : {len(eval_3hop_linear):>6}  (1-step OOD)")
    print(f"  eval_4hop_linear.jsonl        : {len(eval_4hop_linear):>6}  (2-step OOD)")
    print("=" * 60)
    print(f"\n→ Bridge depth alignment:")
    print(f"  train: bridge depth = 1  (graph 1-hop analogue)")
    print(f"  eval : bridge depth = 1, 2, 3")


if __name__ == "__main__":
    main()