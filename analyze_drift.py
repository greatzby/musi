"""
analyze_drift.py

读取 eval_compositional.py 输出的 prediction.jsonl 文件,
计算 drift 指标,生成对比表 + 曲线图。

使用方法:
    python analyze_drift.py
"""

import json
import os
import re
import argparse
from collections import OrderedDict
from typing import List, Dict, Tuple

# ============================================================
# 配置:你想分析的 eval 输出目录
# ============================================================

DEFAULT_RUNS = [
    # (label, eval_root_dir, type)
    ("SFT",            "eval_results_sft_fine",                "sft"),
    ("RL no-anchor",   "eval_results_rl_noanchor_from_200",    "rl"),
]
SPLITS = ["2hop", "3hop_linear", "4hop_linear"]


def load_predictions(jsonl_path: str) -> List[Dict]:
    out = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def compute_drift_metrics(preds: List[Dict]) -> Dict[str, float]:
    """
    核心 drift 指标:
      - em            : 最终答案 EM
      - bridge_recall : 桥接 recall (论文 P(B,C))
      - chain_em      : 端到端正确率
      - cond_em_given_bridge : 桥接对的前提下,answer 也对的概率 (论文 P(D|A,B,C))
      - drift_rate    : 桥接对但 answer 错 / 桥接对 (drift 的直接量化)
    """
    n = len(preds)
    if n == 0:
        return {"n": 0}

    em_sum, br_sum, chain_sum, f1_sum = 0.0, 0.0, 0.0, 0.0
    n_bridge_all_correct = 0    # bridge_hits == n_gold_bridges
    n_bridge_correct_and_em = 0
    n_bridge_correct_and_em0 = 0  # drift cases

    for p in preds:
        em      = float(p.get("em", 0.0))
        f1      = float(p.get("f1", 0.0))
        br_rec  = float(p.get("bridge_recall", 0.0))
        chain   = float(p.get("chain_em", 0.0))
        n_gold  = len(p.get("gold_bridges", []))
        hits    = int(p.get("bridge_hits", 0))

        em_sum    += em
        f1_sum    += f1
        br_sum    += br_rec
        chain_sum += chain

        # 严格"桥接全对"
        bridge_all_ok = (n_gold > 0 and hits == n_gold) or (n_gold == 0)
        if bridge_all_ok:
            n_bridge_all_correct += 1
            if em == 1.0:
                n_bridge_correct_and_em += 1
            else:
                n_bridge_correct_and_em0 += 1

    em_pct          = em_sum / n * 100
    f1_pct          = f1_sum / n * 100
    bridge_pct      = br_sum / n * 100
    chain_pct       = chain_sum / n * 100
    bridge_all_pct  = n_bridge_all_correct / n * 100

    if n_bridge_all_correct > 0:
        cond_em = n_bridge_correct_and_em / n_bridge_all_correct * 100
        drift_rate = n_bridge_correct_and_em0 / n_bridge_all_correct * 100
    else:
        cond_em = 0.0
        drift_rate = 0.0

    return {
        "n": n,
        "em": em_pct,
        "f1": f1_pct,
        "bridge_recall": bridge_pct,
        "bridge_all_correct": bridge_all_pct,
        "chain_em": chain_pct,
        "cond_em_given_bridge": cond_em,
        "drift_rate": drift_rate,
    }


def discover_ckpt_dirs(eval_root: str) -> List[Tuple[int, str, str]]:
    """
    返回 [(step, ckpt_name, full_path), ...],按 step 排序。
    'final' 视为最后, 'best' 视为 -1 (放最前以示参考).
    """
    if not os.path.isdir(eval_root):
        return []
    out = []
    for name in os.listdir(eval_root):
        full = os.path.join(eval_root, name)
        if not os.path.isdir(full):
            continue
        if name == "best":
            out.append((-1, name, full))
        elif name == "final":
            out.append((10**9, name, full))  # 放最后
        else:
            m = re.match(r"^checkpoint-(\d+)$", name)
            if m:
                out.append((int(m.group(1)), name, full))
            elif name == "root":
                out.append((-2, name, full))  # 放更前
    out.sort(key=lambda x: x[0])
    return out


def format_pct(x: float) -> str:
    return f"{x:6.2f}"


def print_table(label: str, split: str, rows: List[Dict]):
    print()
    print("=" * 120)
    print(f"[{label}]   split: {split}")
    print("=" * 120)
    header = (
        f"{'ckpt':<14}{'n':>6}"
        f"{'EM':>9}{'F1':>9}{'Bridge':>9}{'BridgeAll':>11}{'Chain':>9}"
        f"{'P(EM|B)':>10}{'DriftRate':>11}"
    )
    print(header)
    print("-" * 120)
    for row in rows:
        print(
            f"{row['ckpt']:<14}{row['n']:>6}"
            f"{format_pct(row['em']):>9}{format_pct(row['f1']):>9}"
            f"{format_pct(row['bridge_recall']):>9}"
            f"{format_pct(row['bridge_all_correct']):>11}"
            f"{format_pct(row['chain_em']):>9}"
            f"{format_pct(row['cond_em_given_bridge']):>10}"
            f"{format_pct(row['drift_rate']):>11}"
        )


def detect_drift(rows: List[Dict], split_label: str) -> Dict:
    """
    比较第一个和最后一个 checkpoint 的差异。
    drift 判据:
        cond_em_given_bridge 显著下降 (>5pp) AND drift_rate 显著上升 (>5pp)
        OR
        bridge_recall 上升但 em 下降
    """
    if len(rows) < 2:
        return {"verdict": "N/A", "reason": "less than 2 checkpoints"}

    first, last = rows[0], rows[-1]
    delta_em      = last["em"] - first["em"]
    delta_bridge  = last["bridge_recall"] - first["bridge_recall"]
    delta_cond_em = last["cond_em_given_bridge"] - first["cond_em_given_bridge"]
    delta_drift   = last["drift_rate"] - first["drift_rate"]

    drifting = (delta_cond_em < -5.0 and delta_drift > 5.0) or \
               (delta_bridge > 2.0 and delta_em < -3.0)

    verdict = "🔴 DRIFT DETECTED" if drifting else "🟢 stable / no clear drift"

    return {
        "split": split_label,
        "verdict": verdict,
        "first_ckpt": first["ckpt"],
        "last_ckpt": last["ckpt"],
        "delta_em": delta_em,
        "delta_bridge": delta_bridge,
        "delta_cond_em_given_bridge": delta_cond_em,
        "delta_drift_rate": delta_drift,
    }


def maybe_plot(all_rows: Dict[str, List[Dict]], split: str, out_png: str):
    """如果装了 matplotlib,画 drift 趋势图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip plot] matplotlib not installed")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    metrics_to_plot = [
        ("bridge_recall",        "Bridge Recall (%)",       "tab:blue"),
        ("em",                   "Answer EM (%)",           "tab:orange"),
        ("cond_em_given_bridge", "P(EM | Bridge OK) (%)",   "tab:green"),
    ]

    for label, rows in all_rows.items():
        if not rows:
            continue
        # 把 'best'/'root'/'final' 放到 numeric x 轴
        xs = []
        for i, r in enumerate(rows):
            ckpt = r["ckpt"]
            m = re.search(r"\d+", ckpt)
            xs.append(int(m.group()) if m else i * 100)

        for key, ylabel, color in metrics_to_plot:
            ys = [r[key] for r in rows]
            ax1.plot(xs, ys, marker="o",
                     label=f"{label} | {ylabel}", color=color, alpha=0.85)

        # drift_rate 用右轴
        ys = [r["drift_rate"] for r in rows]
        ax2.plot(xs, ys, marker="x", linestyle="--",
                 label=f"{label} | Drift Rate (%)", color="tab:red", alpha=0.6)

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Metric (%)")
    ax2.set_ylabel("Drift Rate (%)", color="tab:red")
    ax1.set_title(f"Drift Analysis — {split}")
    ax1.legend(loc="lower left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  saved plot: {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="drift_analysis_output")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 120)
    print("DRIFT ANALYSIS")
    print("=" * 120)

    # 收集所有结果
    # all_results[split][run_label] = [{ckpt, ...metrics}, ...]
    all_results: Dict[str, Dict[str, List[Dict]]] = {s: {} for s in SPLITS}

    for label, eval_root, run_type in DEFAULT_RUNS:
        if not os.path.isdir(eval_root):
            print(f"  [skip] {eval_root} not found")
            continue

        ckpts = discover_ckpt_dirs(eval_root)
        print(f"\n[{label}] found {len(ckpts)} checkpoints in {eval_root}")
        for step, name, _ in ckpts:
            print(f"  - {name} (step={step if step >= 0 else 'special'})")

        for split in SPLITS:
            rows = []
            for step, ckpt_name, ckpt_path in ckpts:
                pred_file = os.path.join(ckpt_path, f"{split}_predictions.jsonl")
                if not os.path.exists(pred_file):
                    continue
                preds = load_predictions(pred_file)
                metrics = compute_drift_metrics(preds)
                metrics["ckpt"] = ckpt_name
                metrics["step"] = step
                rows.append(metrics)
            all_results[split][label] = rows

    # 打印每个 split 的对比表
    summary = {}
    for split in SPLITS:
        if not any(all_results[split].values()):
            continue
        for label, rows in all_results[split].items():
            if rows:
                print_table(label, split, rows)
        # drift 判定 (只针对 RL run)
        for label, rows in all_results[split].items():
            if "rl" in label.lower() or "no-anchor" in label.lower():
                verdict = detect_drift(rows, split)
                summary.setdefault(split, []).append({"run": label, **verdict})

        # 画图
        plot_path = os.path.join(args.out_dir, f"drift_{split}.png")
        maybe_plot(all_results[split], split, plot_path)

    # 总结
    print("\n" + "=" * 120)
    print("DRIFT VERDICT SUMMARY")
    print("=" * 120)
    for split, items in summary.items():
        for item in items:
            print(f"\n[{split}] {item['run']}: {item['verdict']}")
            print(f"  {item['first_ckpt']} -> {item['last_ckpt']}")
            print(f"    Δ Answer EM            = {item['delta_em']:+.2f} pp")
            print(f"    Δ Bridge Recall        = {item['delta_bridge']:+.2f} pp")
            print(f"    Δ P(EM | Bridge OK)    = {item['delta_cond_em_given_bridge']:+.2f} pp  (drift's smoking gun)")
            print(f"    Δ Drift Rate           = {item['delta_drift_rate']:+.2f} pp  (the direct count)")

    # 保存 JSON
    out_json = os.path.join(args.out_dir, "drift_analysis.json")
    save_data = {}
    for split, runs in all_results.items():
        save_data[split] = {}
        for label, rows in runs.items():
            save_data[split][label] = rows
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"per_split": save_data, "verdict": summary}, f,
                  indent=2, ensure_ascii=False)
    print(f"\nFull results saved: {out_json}")


if __name__ == "__main__":
    main()