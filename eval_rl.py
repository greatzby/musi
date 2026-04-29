# eval_rl.py — aligned with eval.py's prompt format, supports LoRA adapters.
import argparse, json, os, re, string
from collections import Counter
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


EVAL_DIR = "prepared_data"
EVAL_FILES = {
    "1hop":        "eval_1hop.jsonl",
    "2hop_linear": "eval_2hop_linear.jsonl",
    "3hop_linear": "eval_3hop_linear.jsonl",
    "4hop_linear": "eval_4hop_linear.jsonl",
}
SYSTEM_PROMPT = (
    "You answer questions based strictly on the given passage. "
    "Output only a short, direct answer (a few words), with no explanation."
)


# ============ SQuAD-style normalization ============
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def em_score(pred: str, golds: List[str]) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def f1_score(pred: str, golds: List[str]) -> float:
    def _f1(p, g):
        p_toks = normalize_answer(p).split()
        g_toks = normalize_answer(g).split()
        if not p_toks or not g_toks:
            return float(p_toks == g_toks)
        common = Counter(p_toks) & Counter(g_toks)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(p_toks)
        recall    = num_same / len(g_toks)
        return 2 * precision * recall / (precision + recall)
    return max(_f1(pred, g) for g in golds)


# ============ Input formatting (mirrors eval.py exactly) ============
def format_user_content(example: Dict) -> str:
    # 1) 1-hop / atomic: single passage
    if 'context' in example and 'context_title' in example:
        return (
            f"Passage Title: {example['context_title']}\n"
            f"Passage: {example['context']}\n\n"
            f"Question: {example['question']}"
        )
    # 2) Multi-hop: context_paragraphs (all supporting, no distractors)
    if 'context_paragraphs' in example:
        parts = []
        for p in example['context_paragraphs']:
            title = p.get('title', '')
            text  = p.get('text', p.get('paragraph_text', ''))
            parts.append(f"Passage Title: {title}\nPassage: {text}")
        ctx = '\n\n'.join(parts)
        return f"{ctx}\n\nQuestion: {example['question']}"
    raise ValueError(f"Unknown schema: keys={list(example.keys())}")


def get_golds(example: Dict) -> List[str]:
    """1-hop -> answer; multi-hop -> final_answer (+ aliases)."""
    golds = []
    if example.get('answer'):
        golds.append(example['answer'])
    if example.get('final_answer'):
        golds.append(example['final_answer'])
    if example.get('answer_aliases'):
        golds.extend(example['answer_aliases'])
    return golds if golds else ['']


def get_id(example: Dict) -> str:
    return example.get('id') or example.get('source_id') or ''


# ============ Model loading (full ckpt OR LoRA adapter) ============
def load_model(ckpt: str, base_default: str):
    is_lora = os.path.exists(os.path.join(ckpt, "adapter_config.json"))
    if is_lora:
        from peft import PeftModel
        with open(os.path.join(ckpt, "adapter_config.json")) as f:
            base_path = json.load(f).get("base_model_name_or_path") or base_default
        print(f"  LoRA adapter | base = {base_path}")
        base = AutoModelForCausalLM.from_pretrained(
            base_path, dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=True,
        ).cuda()
        model = PeftModel.from_pretrained(base, ckpt).cuda()
        # tokenizer: prefer base (must match what SFT was trained with)
        tok_path = base_path
    else:
        print("  Full model checkpoint")
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=True,
        ).cuda()
        tok_path = ckpt

    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for batched generate
    model.eval()
    return model, tok


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
        add_special_tokens=False,  # chat template already adds them
    ).to(model.device)

    # Qwen2.5: stop on <|im_end|>, NOT default eos_token (<|endoftext|>)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_id = im_end if im_end is not None and im_end >= 0 else tokenizer.eos_token_id

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=eos_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    prompt_len = inputs['input_ids'].shape[1]
    answers = []
    for seq in out:
        new_tokens = seq[prompt_len:]
        ans = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        answers.append(ans)
    return answers


def evaluate_file(model, tokenizer, jsonl_path: str,
                  batch_size: int, max_new_tokens: int,
                  max_input_len: int, hop: int, limit: int = None,
                  debug_n: int = 3):
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if limit:
        examples = examples[:limit]

    print(f"  [debug] keys: {list(examples[0].keys())}")

    prompts = []
    for ex in examples:
        user = format_user_content(ex)
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ]
        p = tokenizer.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True)
        prompts.append(p)

    # length-sorted batching
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    sorted_prompts = [prompts[i] for i in order]

    sorted_preds = []
    for i in tqdm(range(0, len(sorted_prompts), batch_size),
                  desc=os.path.basename(jsonl_path)):
        batch = sorted_prompts[i:i+batch_size]
        sorted_preds.extend(generate_batch(
            model, tokenizer, batch,
            max_new_tokens=max_new_tokens,
            max_input_len=max_input_len,
        ))

    preds = [None] * len(prompts)
    for rank, orig_idx in enumerate(order):
        preds[orig_idx] = sorted_preds[rank]

    em_total = f1_total = 0.0
    detailed = []
    for k, (ex, pred) in enumerate(zip(examples, preds)):
        golds = get_golds(ex)
        em = em_score(pred, golds)
        f1 = f1_score(pred, golds)
        em_total += em
        f1_total += f1
        detailed.append({
            'id':         get_id(ex),
            'hop':        ex.get('hop', hop),
            'question':   ex.get('question', ''),
            'gold':       golds,
            'gold_chain': ex.get('gold_chain', []),
            'pred':       pred,
            'em':         em,
            'f1':         f1,
        })
        if k < debug_n:
            print(f"  [debug] Q: {ex.get('question','')[:120]}")
            print(f"  [debug] gold: {golds}")
            print(f"  [debug] pred: {pred!r}")

    n = len(examples)
    return {'n': n, 'em': em_total / n * 100, 'f1': f1_total / n * 100}, detailed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default="checkpoints/qwen2.5-3b-atomic-sft")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eval_dir", default=EVAL_DIR)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--max_input_len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--splits", nargs='+',
                    default=['1hop', '2hop_linear', '3hop_linear', '4hop_linear'])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading model from {args.ckpt}")
    model, tok = load_model(args.ckpt, args.base)

    summary = {}
    for split_name in args.splits:
        if split_name not in EVAL_FILES:
            print(f"[skip] unknown split: {split_name}")
            continue
        path = os.path.join(args.eval_dir, EVAL_FILES[split_name])
        if not os.path.exists(path):
            print(f"[skip] not found: {path}")
            continue
        hop = int(split_name[0])
        print(f"\n=== {split_name} ===")
        metrics, detailed = evaluate_file(
            model, tok, path,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            max_input_len=args.max_input_len,
            hop=hop,
            limit=args.limit,
        )
        print(f"  n={metrics['n']}  EM={metrics['em']:.2f}  F1={metrics['f1']:.2f}")
        summary[split_name] = metrics

        with open(os.path.join(args.out_dir, f"{split_name}_predictions.jsonl"),
                  'w', encoding='utf-8') as f:
            for d in detailed:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')

    print("\n" + "="*60)
    print(f"{'Split':<15} {'N':>6} {'EM':>8} {'F1':>8}")
    print("-"*60)
    for split, m in summary.items():
        print(f"{split:<15} {m['n']:>6} {m['em']:>8.2f} {m['f1']:>8.2f}")
    print("="*60)

    with open(os.path.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.out_dir}/")


if __name__ == "__main__":
    main()