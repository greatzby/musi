"""
eval_compositional.py  (v2: 支持 --include_bridge_count)

在 2-hop / 3-hop linear / 4-hop linear 上评估 bridge-aware 模型。
"""

import json
import os
import re
import gc
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
def normalize_answer(s):
    if s is None:
        return ""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def em_score(pred, golds):
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds if g))


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
    recall    = num_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def f1_score(pred, golds):
    return max(_f1_pair(pred, g) for g in golds if g) if golds else 0.0


# ============ Output parsing ============
RE_BRIDGE = re.compile(r'^\s*Bridge\s*\d+\s*[:：]\s*(.+?)\s*$', re.IGNORECASE)
RE_ANSWER = re.compile(r'^\s*Answer\s*[:：]\s*(.+?)\s*$', re.IGNORECASE)


def parse_output(text):
    bridges, answer, last = [], "", ""
    found = False
    for raw in text.strip().split('\n'):
        line = raw.strip()
        if not line:
            continue
        last = line
        m_b = RE_BRIDGE.match(line)
        if m_b:
            bridges.append(m_b.group(1).strip())
            continue
        m_a = RE_ANSWER.match(line)
        if m_a:
            answer = m_a.group(1).strip()
            found = True
            continue
    if not found and last:
        answer = last
    return bridges, answer


def bridge_match(p, g):
    p, g = normalize_answer(p), normalize_answer(g)
    if not p or not g:
        return False
    if p == g or g in p or p in g:
        return True
    return _f1_pair(p, g) >= 0.7


def compute_bridge_metrics(pred_bridges, gold_bridges):
    if not gold_bridges:
        return {"bridge_recall": 1.0, "bridge_f1": 1.0, "bridge_hits": 0,
                "n_gold": 0, "n_pred": len(pred_bridges)}
    if not pred_bridges:
        return {"bridge_recall": 0.0, "bridge_f1": 0.0, "bridge_hits": 0,
                "n_gold": len(gold_bridges), "n_pred": 0}
    hits = sum(any(bridge_match(p, g) for p in pred_bridges) for g in gold_bridges)
    recall = hits / len(gold_bridges)
    f1s = [max(_f1_pair(p, g) for p in pred_bridges) for g in gold_bridges]
    bridge_f1 = sum(f1s) / len(f1s)
    return {"bridge_recall": recall, "bridge_f1": bridge_f1, "bridge_hits": hits,
            "n_gold": len(gold_bridges), "n_pred": len(pred_bridges)}


# ============ Input formatting ============
def format_user_content(example, include_bridge_count=False):
    parts = []
    for p in example["context_paragraphs"]:
        title = p.get("title", "")
        text  = p.get("text", p.get("paragraph_text", ""))
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


def get_final_golds(example):
    golds = []
    if example.get("final_answer"):
        golds.append(example["final_answer"])
    if example.get("answer"):
        golds.append(example["answer"])
    if example.get("answer_aliases"):
        golds.extend(example["answer_aliases"])
    return [g for g in golds if g] or [""]


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_new_tokens, max_input_len):
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_input_len, add_special_tokens=False,
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
    return [tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
            for seq in out]


def evaluate_file(model, tokenizer, jsonl_path, batch_size, max_new_tokens,
                  max_input_len, include_bridge_count, limit=None):
    examples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if limit:
        examples = examples[:limit]

    prompts = []
    for ex in examples:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": format_user_content(ex, include_bridge_count)},
        ]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    sorted_prompts = [prompts[i] for i in order]
    sorted_outputs = []
    for i in tqdm(range(0, len(sorted_prompts), batch_size),
                  desc=os.path.basename(jsonl_path)):
        batch = sorted_prompts[i:i + batch_size]
        sorted_outputs.extend(generate_batch(
            model, tokenizer, batch, max_new_tokens, max_input_len))

    raw_outputs = [None] * len(prompts)
    for rank, orig_idx in enumerate(order):
        raw_outputs[orig_idx] = sorted_outputs[rank]

    em_t = f1_t = br_t = bf_t = chain_t = 0.0
    detailed = []
    for ex, raw in zip(examples, raw_outputs):
        pred_bridges, pred_answer = parse_output(raw)
        gold_bridges = ex.get("bridges", [])
        final_golds  = get_final_golds(ex)
        em = em_score(pred_answer, final_golds)
        f1 = f1_score(pred_answer, final_golds)
        bm = compute_bridge_metrics(pred_bridges, gold_bridges)
        chain_em = float(em == 1.0 and bm["bridge_hits"] == bm["n_gold"])
        em_t += em; f1_t += f1; br_t += bm["bridge_recall"]
        bf_t += bm["bridge_f1"]; chain_t += chain_em
        detailed.append({
            "id": ex.get("id", ""),
            "hop": ex.get("hop"),
            "question": ex.get("question", ""),
            "gold_bridges": gold_bridges,
            "gold_final": final_golds,
            "pred_raw": raw,
            "pred_bridges": pred_bridges,
            "pred_answer": pred_answer,
            "em": em, "f1": f1,
            "bridge_recall": bm["bridge_recall"],
            "bridge_f1": bm["bridge_f1"],
            "bridge_hits": bm["bridge_hits"],
            "chain_em": chain_em,
        })
    n = len(examples)
    return {
        "n": n,
        "em":            em_t / n * 100,
        "f1":            f1_t / n * 100,
        "bridge_recall": br_t / n * 100,
        "bridge_f1":     bf_t / n * 100,
        "chain_em":      chain_t / n * 100,
    }, detailed


def discover_checkpoints(parent_dir):
    pattern = re.compile(r'^(checkpoint-\d+|final|best)$')
    found = []
    for name in os.listdir(parent_dir):
        if pattern.match(name) and os.path.isdir(os.path.join(parent_dir, name)):
            m = re.search(r'\d+', name)
            step = int(m.group()) if m else 10**9
            found.append((step, os.path.join(parent_dir, name)))
    found.sort()
    return [p for _, p in found]


def load_model_and_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).cuda()
    model.eval()
    return model, tokenizer


def free_gpu(model, tokenizer):
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--model_dirs", type=str, nargs="+", default=None)
    parser.add_argument("--auto_discover", type=str, default=None)
    parser.add_argument("--include_root", action="store_true")
    parser.add_argument("--include_bridge_count", action="store_true",
                        help="prompt 中告诉模型需要多少个 bridge (oracle)")

    parser.add_argument("--eval_dir", type=str, default=EVAL_DIR)
    parser.add_argument("--out_dir", type=str, default="eval_results_2hop")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_input_len", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--splits", nargs="+",
                        default=["2hop", "3hop_linear", "4hop_linear"])
    args = parser.parse_args()

    if args.auto_discover:
        model_dirs = discover_checkpoints(args.auto_discover)
        if args.include_root:
            model_dirs.append(args.auto_discover)
        if not model_dirs:
            raise ValueError(f"No checkpoint-* found in {args.auto_discover}")
    elif args.model_dirs:
        model_dirs = args.model_dirs
    elif args.model_dir:
        model_dirs = [args.model_dir]
    else:
        model_dirs = [DEFAULT_MODEL_DIR]

    print(f"Will evaluate {len(model_dirs)} model(s) "
          f"(include_bridge_count={args.include_bridge_count}):")
    for d in model_dirs:
        print(f"  - {d}")

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = {}

    for model_dir in model_dirs:
        ckpt_name = os.path.basename(os.path.normpath(model_dir))
        if not re.match(r'^(checkpoint-\d+|final|best)$', ckpt_name):
            ckpt_name = "root"
        ckpt_out_dir = os.path.join(args.out_dir, ckpt_name)
        os.makedirs(ckpt_out_dir, exist_ok=True)

        print(f"\n{'#'*70}\n# Evaluating: {model_dir}\n# Output: {ckpt_out_dir}\n{'#'*70}")
        model, tokenizer = load_model_and_tokenizer(model_dir)
        ckpt_summary = {}

        for split in args.splits:
            if split not in EVAL_FILES:
                continue
            path = os.path.join(args.eval_dir, EVAL_FILES[split])
            if not os.path.exists(path):
                print(f"[skip] not found: {path}")
                continue
            print(f"\n=== {ckpt_name} / {split} ===")
            metrics, detailed = evaluate_file(
                model, tokenizer, path,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                max_input_len=args.max_input_len,
                include_bridge_count=args.include_bridge_count,
                limit=args.limit,
            )
            print(f"  n={metrics['n']:>5}  EM={metrics['em']:>5.2f}  F1={metrics['f1']:>5.2f}  "
                  f"BridgeR={metrics['bridge_recall']:>5.2f}  "
                  f"BridgeF1={metrics['bridge_f1']:>5.2f}  ChainEM={metrics['chain_em']:>5.2f}")
            ckpt_summary[split] = metrics
            with open(os.path.join(ckpt_out_dir, f"{split}_predictions.jsonl"),
                      "w", encoding="utf-8") as f:
                for d in detailed:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")

        with open(os.path.join(ckpt_out_dir, "summary.json"),
                  "w", encoding="utf-8") as f:
            json.dump(ckpt_summary, f, indent=2, ensure_ascii=False)
        all_results[ckpt_name] = ckpt_summary
        free_gpu(model, tokenizer)

    print("\n" + "=" * 100)
    print(f"CROSS-CHECKPOINT COMPARISON  (include_bridge_count={args.include_bridge_count})")
    print("=" * 100)
    metric_keys = ["em", "f1", "bridge_recall", "bridge_f1", "chain_em"]
    metric_labels = {"em": "EM", "f1": "F1", "bridge_recall": "BridgeR",
                     "bridge_f1": "BridgeF1", "chain_em": "ChainEM"}
    for split in args.splits:
        print(f"\n--- {split} ---")
        header = f"{'Checkpoint':<22}{'N':>6}" + "".join(f"{metric_labels[m]:>10}" for m in metric_keys)
        print(header)
        print("-" * len(header))
        for ckpt_name, ckpt_summary in all_results.items():
            if split not in ckpt_summary:
                continue
            m = ckpt_summary[split]
            print(f"{ckpt_name:<22}{m['n']:>6}" + "".join(f"{m[k]:>10.2f}" for k in metric_keys))
    print("=" * 100)
    with open(os.path.join(args.out_dir, "all_checkpoints_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()