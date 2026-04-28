#!/bin/bash
set -e

MODEL_ROOT="checkpoints/qwen2.5-3b-atomic-sft"
OUT_ROOT="eval_results"

# 1) 自动发现所有中间 checkpoint
echo "=== Found checkpoints ==="
ls -d ${MODEL_ROOT}/checkpoint-* 2>/dev/null || echo "(no intermediate checkpoints)"
echo ""

# 2) 评估每个中间 checkpoint
for ckpt_dir in ${MODEL_ROOT}/checkpoint-*; do
    [ -d "$ckpt_dir" ] || continue
    ckpt_name=$(basename "$ckpt_dir")
    echo ""
    echo "########## Evaluating $ckpt_name ##########"
    python eval.py \
        --model_dir "$ckpt_dir" \
        --out_dir "${OUT_ROOT}/${ckpt_name}" \
        --batch_size 16
done

# 3) 评估最终模型（顶层目录）
echo ""
echo "########## Evaluating final ##########"
python eval.py \
    --model_dir "$MODEL_ROOT" \
    --out_dir "${OUT_ROOT}/final" \
    --batch_size 16

# 4) 汇总所有 summary.json
echo ""
echo "============================================================"
echo "ALL CHECKPOINTS SUMMARY"
echo "============================================================"
python - <<'PY'
import json, os, glob
rows = []
for sd in sorted(glob.glob("eval_results/*/summary.json")):
    name = os.path.basename(os.path.dirname(sd))
    with open(sd) as f:
        s = json.load(f)
    rows.append((name, s))

splits = ["1hop", "2hop_linear", "3hop_linear", "4hop_linear"]
header = f"{'Checkpoint':<22}" + "".join(f"{sp+' EM':>14}{sp+' F1':>8}" for sp in splits)
print(header)
print("-" * len(header))
for name, s in rows:
    line = f"{name:<22}"
    for sp in splits:
        if sp in s:
            line += f"{s[sp]['em']:>14.2f}{s[sp]['f1']:>8.2f}"
        else:
            line += f"{'-':>14}{'-':>8}"
    print(line)
PY