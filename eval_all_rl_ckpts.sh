#!/bin/bash
# Usage: ./eval_all_rl_ckpts.sh checkpoints/qwen2.5-3b-grpo

RL_DIR=${1:-checkpoints/qwen2.5-3b-grpo}
BASE=${2:-checkpoints/qwen2.5-3b-atomic-sft}
EVAL_ROOT="eval_results/$(basename $RL_DIR)"
STRIDE=${3:-1}   # 1 = eval every saved ckpt; 4 = every 4th to scan first

mkdir -p "$EVAL_ROOT"
echo "=== RL checkpoints in $RL_DIR (stride=$STRIDE) ==="

ckpts=( $(ls -d "$RL_DIR"/step-* 2>/dev/null | sort -t- -k2 -n) )
echo "Found ${#ckpts[@]} checkpoints"

i=0
for ckpt in "${ckpts[@]}"; do
    if (( i % STRIDE != 0 )); then i=$((i+1)); continue; fi
    name=$(basename "$ckpt")
    out="$EVAL_ROOT/$name"
    if [[ -f "$out/summary.json" ]]; then
        echo "[skip] $name (already evaluated)"
        i=$((i+1)); continue
    fi
    echo ""
    echo "########## Evaluating $name ##########"
    python eval_rl.py --ckpt "$ckpt" --base "$BASE" \
        --out_dir "$out" --batch_size 16
    i=$((i+1))
done

# Aggregate
echo ""
echo "============================================================"
echo "ALL RL CHECKPOINTS SUMMARY"
echo "============================================================"
printf "%-20s %8s %8s %8s %8s %8s %8s %8s %8s\n" \
    "Checkpoint" "1h_EM" "1h_F1" "2h_EM" "2h_F1" "3h_EM" "3h_F1" "4h_EM" "4h_F1"
echo "------------------------------------------------------------------------------------------"

python - <<EOF
import json, os, glob
root = "$EVAL_ROOT"
rows = []
for d in sorted(glob.glob(os.path.join(root, "step-*")),
                key=lambda x: int(x.rsplit("-",1)[-1])):
    fp = os.path.join(d, "summary.json")
    if not os.path.exists(fp): continue
    s = json.load(open(fp))
    rows.append((os.path.basename(d), s))
for name, s in rows:
    print(f"{name:<20} "
          f"{s['1hop']['EM']:>8.2f} {s['1hop']['F1']:>8.2f} "
          f"{s['2hop_linear']['EM']:>8.2f} {s['2hop_linear']['F1']:>8.2f} "
          f"{s['3hop_linear']['EM']:>8.2f} {s['3hop_linear']['F1']:>8.2f} "
          f"{s['4hop_linear']['EM']:>8.2f} {s['4hop_linear']['F1']:>8.2f}")
EOF