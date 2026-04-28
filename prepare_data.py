"""
prepare_data.py

把官方 MuSiQue-Ans 切成实验需要的格式：
- atomic_train.jsonl:        训练集 (atomic single-hop pairs from train)
- eval_1hop.jsonl:           评估 atomic 能力 (atomic pairs from dev)
- eval_Nhop_all.jsonl:       N-hop 全部结构 (linear + tree)
- eval_Nhop_linear.jsonl:    N-hop 仅线性链 (2hop, 3hop1, 4hop1)
"""

import json
import re
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

# ---------- 路径 ----------
MUSIQUE_DIR = Path("./data")             # 官方数据所在
OUT_DIR     = Path("./prepared_data")    # 输出目录
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
    """
    解析 hop 数和结构变体.
    返回: (hop_count: int, variant: str)
      '2hop__...'    → (2, '')      唯一结构 (线性)
      '3hop1__...'   → (3, '1')     线性链 A→B→C
      '3hop2__...'   → (3, '2')     树形 A→C, B→C (汇聚)
      '4hop1__...'   → (4, '1')     线性 A→B→C→D
      '4hop2__...'   → (4, '2')     含汇聚结构
      '4hop3__...'   → (4, '3')     含汇聚结构
    """
    m = re.match(r"(\d+)hop(\d*)__", example["id"])
    if m is None:
        raise ValueError(f"Cannot parse hop from id: {example['id']}")
    return int(m.group(1)), m.group(2)


def get_hop_count(example):
    return parse_hop_id(example)[0]


def is_linear_chain(example):
    """线性链 = 2hop (唯一结构), 3hop1, 4hop1"""
    hop, var = parse_hop_id(example)
    if hop == 2:
        return True
    return var == "1"


def resolve_references(subq_text, prev_answers):
    """
    把子问题里的 #1, #2 替换成前面 step 的 answer.
    e.g. subq = 'Who is the spouse of #1?', prev_answers = ['Christopher Nolan']
         → 'Who is the spouse of Christopher Nolan?'
    """
    def repl(match):
        idx = int(match.group(1)) - 1   # #1 → prev_answers[0]
        if 0 <= idx < len(prev_answers):
            return prev_answers[idx]
        return match.group(0)            # 找不到就保留原样 (理论不发生)
    return re.sub(r"#(\d+)", repl, subq_text)


def get_supporting_paragraph(example, idx):
    """根据 paragraph_support_idx 找对应段落"""
    for p in example["paragraphs"]:
        if p["idx"] == idx:
            return p
    return None


# ---------- 转换逻辑 ----------
def build_atomic_examples(examples, source_tag):
    """
    把每个 multi-hop example 拆成多个 atomic single-hop pairs.
    每条 = 1 个 (gold paragraph, resolved subquestion, answer).
    """
    atomic = []
    skipped = 0
    for ex in examples:
        prev_answers = []
        for step in ex["question_decomposition"]:
            subq = resolve_references(step["question"], prev_answers)
            ans  = step["answer"]
            para = get_supporting_paragraph(ex, step["paragraph_support_idx"])
            if para is None:
                skipped += 1
                prev_answers.append(ans)
                continue
            atomic.append({
                "source_id":     ex["id"],
                "source_split":  source_tag,
                "step_idx":      step["id"],
                "context_title": para["title"],
                "context":       para["paragraph_text"],
                "question":      subq,
                "answer":        ans,
            })
            prev_answers.append(ans)
    if skipped:
        print(f"  [warn] skipped {skipped} steps with missing supporting paragraph")
    return atomic


def build_multihop_eval_examples(examples):
    """
    多-hop 评估格式:
    - 提供 gold supporting paragraphs (避免引入检索误差)
    - 保留 gold_chain (中间答案), 用于链式条件概率指标
    """
    out = []
    for ex in examples:
        hop, variant = parse_hop_id(ex)

        # 只保留 supporting paragraphs, 按原顺序排序
        gold_paras = sorted(
            [p for p in ex["paragraphs"] if p["is_supporting"]],
            key=lambda p: p["idx"]
        )

        # 同时构造 resolved 子问题 + gold 中间答案链
        prev_answers = []
        chain = []
        for step in ex["question_decomposition"]:
            chain.append({
                "step_idx":    step["id"],
                "subq_raw":    step["question"],                          # 含 #N
                "subq":        resolve_references(step["question"], prev_answers),
                "answer":      step["answer"],
                "support_idx": step["paragraph_support_idx"],
            })
            prev_answers.append(step["answer"])

        out.append({
            "id":             ex["id"],
            "hop":            hop,
            "hop_variant":    variant,
            "is_linear":      is_linear_chain(ex),
            "question":       ex["question"],
            "context_paragraphs": [
                {"title": p["title"], "text": p["paragraph_text"]}
                for p in gold_paras
            ],
            "gold_chain":     chain,
            "final_answer":   ex["answer"],
            "answer_aliases": ex.get("answer_aliases", []),
        })
    return out


# ---------- 主流程 ----------
def main():
    print("Loading raw MuSiQue-Ans...")
    train_raw = load_jsonl(TRAIN_FILE)
    dev_raw   = load_jsonl(DEV_FILE)
    print(f"  Train multi-hop: {len(train_raw)}")
    print(f"  Dev   multi-hop: {len(dev_raw)}")

    # --- hop 分布 ---
    print("\nHop distribution (by hop count only):")
    for name, split in [("train", train_raw), ("dev", dev_raw)]:
        cnt = defaultdict(int)
        for ex in split:
            cnt[get_hop_count(ex)] += 1
        print(f"  {name}: {dict(sorted(cnt.items()))}")

    # --- 结构变体分布 ---
    print("\nStructure variant distribution:")
    for name, split in [("train", train_raw), ("dev", dev_raw)]:
        cnt = defaultdict(int)
        for ex in split:
            h, v = parse_hop_id(ex)
            key = f"{h}hop{v}" if v else f"{h}hop"
            cnt[key] += 1
        print(f"  {name}: {dict(sorted(cnt.items()))}")

    # --- 1) Train: atomic only (用全部 multi-hop, 不分结构) ---
    print("\nBuilding atomic training set from train split...")
    atomic_train = build_atomic_examples(train_raw, source_tag="train")
    random.shuffle(atomic_train)
    print(f"  Total atomic pairs: {len(atomic_train)}")

    # --- 2) Eval-1hop: atomic from dev (model 完全没见过) ---
    print("\nBuilding 1-hop eval set from dev split (atomic pairs)...")
    atomic_dev = build_atomic_examples(dev_raw, source_tag="dev")
    random.shuffle(atomic_dev)
    eval_1hop = atomic_dev[:1500]
    print(f"  1-hop eval pairs: {len(eval_1hop)}")

    # --- 3) Eval-Nhop: multi-hop from dev, 按 hop 分桶 + 线性链单独存 ---
    print("\nBuilding multi-hop eval sets from dev split...")
    multihop_dev = build_multihop_eval_examples(dev_raw)

    by_hop = defaultdict(list)         # 按 hop 全量
    by_hop_var = defaultdict(list)     # 按 hop+variant 细分 (用于打印)
    linear_only = defaultdict(list)    # 仅线性链 (主实验用)

    for ex in multihop_dev:
        h, v = ex["hop"], ex["hop_variant"]
        by_hop[h].append(ex)
        key = f"{h}hop{v}" if v else f"{h}hop"
        by_hop_var[key].append(ex)
        if ex["is_linear"]:
            linear_only[h].append(ex)

    # --- 写盘 ---
    print("\nWriting outputs...")
    dump_jsonl(atomic_train, OUT_DIR / "atomic_train.jsonl")
    dump_jsonl(eval_1hop,    OUT_DIR / "eval_1hop.jsonl")

    # 全量 (各 hop 不分 variant)
    for h in sorted(by_hop):
        dump_jsonl(by_hop[h], OUT_DIR / f"eval_{h}hop_all.jsonl")

    # 线性链 (主实验用)
    for h in sorted(linear_only):
        dump_jsonl(linear_only[h], OUT_DIR / f"eval_{h}hop_linear.jsonl")

    # --- 总结 ---
    print("\n" + "=" * 60)
    print("Done. Summary:")
    print(f"  atomic_train.jsonl       : {len(atomic_train):>6}  (训练用)")
    print(f"  eval_1hop.jsonl          : {len(eval_1hop):>6}  (1-hop 评估)")
    print("  --- All structures (linear + tree) ---")
    for h in sorted(by_hop):
        print(f"  eval_{h}hop_all.jsonl       : {len(by_hop[h]):>6}")
    print("  --- Linear chain only (主实验) ---")
    for h in sorted(linear_only):
        print(f"  eval_{h}hop_linear.jsonl    : {len(linear_only[h]):>6}")
    print("\n  Structure breakdown in dev:")
    for k in sorted(by_hop_var):
        print(f"    {k:>8}: {len(by_hop_var[k]):>4}")
    print("=" * 60)


if __name__ == "__main__":
    main()