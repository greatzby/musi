# eval_rl.py
"""Evaluate an RL LoRA adapter (or any HF model) on the same 4 splits as eval.py.
Output format mirrors eval.py: per-split predictions.jsonl + summary.json."""

import argparse, json, os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl_common import f1_score, em_score, build_messages

SPLITS = {
    "1hop":        "prepared_data/eval_1hop.jsonl",
    "2hop_linear": "prepared_data/eval_2hop_linear.jsonl",
    "3hop_linear": "prepared_data/eval_3hop_linear.jsonl",
    "4hop_linear": "prepared_data/eval_4hop_linear.jsonl",
}

MULTIHOP_SYSTEM = (
    "You answer questions based strictly on the given passages. "
    "Output only a short, direct answer (a few words), with no explanation."
)


def load_model(ckpt, base_default):
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
        # tokenizer: prefer adapter dir, fall back to base
        tok_path = ckpt if os.path.exists(os.path.join(ckpt, "tokenizer_config.json")) else base_path
    else:
        print("  Full model checkpoint")
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=True,
        ).cuda()
        tok_path = ckpt

    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model.eval()
    return model, tok


def build_prompt(ex, hop, tokenizer):
    if hop == 1:
        msgs = build_messages(ex)  # uses context_title / context / question
    else:
        passages = ex.get("passages")
        if passages:
            blocks = [f"Passage {i+1} ({p.get('title','')}):\n{p.get('text','')}"
                      for i, p in enumerate(passages)]
            user = "\n\n".join(blocks) + f"\n\nQuestion: {ex['question']}"
        else:
            user = (f"Passage Title: {ex.get('context_title','')}\n"
                    f"Passage: {ex.get('context','')}\n\n"
                    f"Question: {ex['question']}")
        msgs = [
            {"role": "system", "content": MULTIHOP_SYSTEM},
            {"role": "user",   "content": user},
        ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def best_em(p, gs): return max((em_score(p,g) for g in gs), default=0.0)
def best_f1(p, gs): return max((f1_score(p,g) for g in gs), default=0.0)


@torch.no_grad()
def eval_split(model, tok, path, hop, batch_size, max_new_tokens, limit=None):
    with open(path) as f:
        examples = [json.loads(l) for l in f if l.strip()]
    if limit: examples = examples[:limit]

    preds = []
    for i in tqdm(range(0, len(examples), batch_size), desc=os.path.basename(path)):
        chunk = examples[i:i+batch_size]
        prompts = [build_prompt(e, hop, tok) for e in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=2048,
                  add_special_tokens=False).to(model.device)
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        )
        plen = enc["input_ids"].size(1)
        for j, ex in enumerate(chunk):
            gen = out[j, plen:]
            np_ = (gen != tok.pad_token_id).nonzero()
            gen = gen[:np_.max().item()+1] if len(np_) else gen[:0]
            text = tok.decode(gen, skip_special_tokens=True).strip()
            golds = ex.get("gold") or [ex["answer"]]
            preds.append({
                "id": ex.get("id",""), "hop": hop,
                "question": ex["question"], "gold": golds,
                "gold_chain": ex.get("gold_chain", []),
                "pred": text,
                "em": best_em(text, golds), "f1": best_f1(text, golds),
            })
    n = len(preds) or 1
    return preds, sum(p["em"] for p in preds)/n*100, sum(p["f1"] for p in preds)/n*100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default="checkpoints/qwen2.5-3b-atomic-sft")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading model from {args.ckpt}")
    model, tok = load_model(args.ckpt, args.base)

    summary = {}
    for split_name, path in SPLITS.items():
        hop = int(split_name[0])
        print(f"\n=== {split_name} ===")
        preds, em, f1 = eval_split(
            model, tok, path, hop,
            args.batch_size, args.max_new_tokens, args.limit
        )
        with open(os.path.join(args.out_dir, f"{split_name}_predictions.jsonl"), "w") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        summary[split_name] = {"n": len(preds), "EM": em, "F1": f1}
        print(f"  n={len(preds)}  EM={em:.2f}  F1={f1:.2f}")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print(f"{'Split':<14}{'N':>8}{'EM':>10}{'F1':>10}")
    print("-"*60)
    for s, r in summary.items():
        print(f"{s:<14}{r['n']:>8}{r['EM']:>10.2f}{r['F1']:>10.2f}")
    print("="*60)
    print(f"\nResults saved to {args.out_dir}/")


if __name__ == "__main__":
    main()