"""
rl_grpo_2hop.py  (v2: chain_binary reward + scheduler fix + diagnostics)

GRPO fine-tuning for MuSiQue 2-hop bridge-aware training.
"""

import argparse
import json
import math
import os
import random
import re
import string
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from tqdm.auto import tqdm


# ============================================================
# Prompt
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


def format_user_content(example: Dict, include_bridge_count: bool = False) -> str:
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


def build_prompt(tokenizer, example: Dict, include_bridge_count: bool = False) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": format_user_content(example, include_bridge_count=include_bridge_count)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ============================================================
# Dataset
# ============================================================

class JsonlDataset(Dataset):
    def __init__(self, path: str, limit: Optional[int] = None):
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))
                if limit is not None and len(self.items) >= limit:
                    break

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_examples(batch):
    return batch


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
# Reward
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


def compute_reward(text, example, reward_mode,
                   w_format, w_bridge, w_final, bridge_gate_floor):
    parsed = parse_output(text)
    gold_bridges = example.get("bridges", [])
    final_golds = get_final_golds(example)

    bridge_score = compute_bridge_score(parsed.bridges, gold_bridges)
    answer_f1 = f1_score(parsed.answer, final_golds)
    answer_em = em_score(parsed.answer, final_golds)
    format_score = compute_format_score(parsed, len(gold_bridges))

    # ---- chain_binary: 完全做对才给 1.0, 否则 cap 0.85 ----
    if reward_mode == "chain_binary":
        bridge_threshold = 0.5
        bridge_pass = float(bridge_score >= bridge_threshold)
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
        final_effective = answer_f1 * (
            bridge_gate_floor + (1.0 - bridge_gate_floor) * bridge_score
        )
        denom = max(w_format + w_bridge + w_final, 1e-8)
        reward = (
            w_format * format_score
            + w_bridge * bridge_score
            + w_final * final_effective
        ) / denom
    elif reward_mode == "final_only":
        reward = 0.15 * format_score + 0.85 * answer_f1
    elif reward_mode == "process_only":
        reward = 0.15 * format_score + 0.85 * bridge_score
    elif reward_mode == "format_only":
        reward = format_score
    else:
        raise ValueError(f"Unknown reward_mode: {reward_mode}")

    reward = float(max(0.0, min(1.0, reward)))

    # 计算严格 chain_em (供日志)
    bridge_pass_strict = float(bridge_score >= 0.5)
    chain_em_strict = float(answer_em == 1.0 and bridge_pass_strict == 1.0)

    comps = {
        "reward": reward,
        "format": float(format_score),
        "bridge": float(bridge_score),
        "answer_f1": float(answer_f1),
        "answer_em": float(answer_em),
        "chain_em": float(chain_em_strict),
        "n_pred_bridge": float(len(parsed.bridges)),
        "n_gold_bridge": float(len(gold_bridges)),
        "has_answer_line": float(parsed.has_answer_line),
    }
    return reward, comps


# ============================================================
# Tokenizer / EOS helpers
# ============================================================

def load_tokenizer_safely(path, fix_mistral_regex):
    kwargs = {"trust_remote_code": True}
    if fix_mistral_regex:
        kwargs["fix_mistral_regex"] = True
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, **kwargs)
    except TypeError:
        kwargs.pop("fix_mistral_regex", None)
        tokenizer = AutoTokenizer.from_pretrained(path, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def get_eos_ids(tokenizer):
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    try:
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end >= 0 and im_end not in eos_ids:
            eos_ids.append(im_end)
    except Exception:
        pass
    return eos_ids if eos_ids else None


# ============================================================
# Generation / logprobs
# ============================================================

@torch.no_grad()
def generate_completions(model, tokenizer, prompts, args, accelerator):
    unwrapped = accelerator.unwrap_model(model)
    was_training = unwrapped.training
    unwrapped.eval()
    unwrapped.config.use_cache = True

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_len,
        add_special_tokens=False,
    )
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        num_return_sequences=args.num_generations,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=get_eos_ids(tokenizer),
        use_cache=True,
        synced_gpus=accelerator.num_processes > 1,
    )
    if args.top_k > 0:
        gen_kwargs["top_k"] = args.top_k

    sequences = unwrapped.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        **gen_kwargs,
    )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = sequences[:, prompt_len:]
    texts = tokenizer.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    texts = [t.strip() for t in texts]

    attention_mask = (sequences != tokenizer.pad_token_id).long()
    completion_mask = torch.zeros_like(sequences, dtype=torch.float32)
    new_mask = (new_tokens != tokenizer.pad_token_id).float()
    completion_mask[:, prompt_len:] = new_mask

    unwrapped.config.use_cache = False
    if was_training:
        unwrapped.train()
    return sequences, attention_mask, completion_mask, texts


def token_logprobs(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    gathered = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    zero = torch.zeros(gathered.shape[0], 1, device=gathered.device, dtype=gathered.dtype)
    return torch.cat([zero, gathered], dim=1)


def masked_mean(x, mask, eps=1e-8):
    return (x * mask).sum() / mask.sum().clamp(min=eps)


def compute_grpo_loss(model, ref_model, sequences, attention_mask, completion_mask,
                      advantages, old_logps, args):
    new_logps = token_logprobs(model, sequences, attention_mask)
    log_ratio = new_logps - old_logps
    log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)

    adv = advantages.view(-1, 1)
    pg_loss = torch.max(-adv * ratio, -adv * clipped_ratio)
    pg_loss = masked_mean(pg_loss, completion_mask)

    if ref_model is not None and args.kl_beta > 0:
        with torch.no_grad():
            ref_logps = token_logprobs(ref_model, sequences, attention_mask)
        log_ratio_ref = torch.clamp(ref_logps - new_logps, -20.0, 20.0)
        kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
        kl_loss = masked_mean(kl, completion_mask)
    else:
        kl_loss = torch.zeros_like(pg_loss)

    loss = pg_loss + args.kl_beta * kl_loss

    with torch.no_grad():
        clipfrac = (
            ((ratio - 1.0).abs() > args.clip_range).float() * completion_mask
        ).sum() / completion_mask.sum().clamp(min=1.0)

    stats = {
        "pg_loss": float(pg_loss.detach().cpu()),
        "kl_loss": float(kl_loss.detach().cpu()),
        "clipfrac": float(clipfrac.detach().cpu()),
    }
    return loss, stats


# ============================================================
# Eval-during-training
# ============================================================

@torch.no_grad()
def quick_eval(model, tokenizer, eval_examples, args, accelerator,
               max_eval_examples=None):
    if not accelerator.is_main_process:
        return None
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()
    unwrapped.config.use_cache = True

    if max_eval_examples is not None:
        eval_examples = eval_examples[:max_eval_examples]

    em_sum, f1_sum, bridge_sum, chain_em_sum = 0.0, 0.0, 0.0, 0.0
    bs = 4
    for i in range(0, len(eval_examples), bs):
        batch = eval_examples[i : i + bs]
        prompts = [build_prompt(tokenizer, ex,
                                include_bridge_count=args.include_bridge_count)
                   for ex in batch]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_input_len, add_special_tokens=False,
        ).to(accelerator.device)
        out = unwrapped.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=get_eos_ids(tokenizer),
            use_cache=True,
        )
        prompt_len = inputs["input_ids"].shape[1]
        new = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        for ex, txt in zip(batch, new):
            parsed = parse_output(txt.strip())
            golds = get_final_golds(ex)
            em = em_score(parsed.answer, golds)
            f1 = f1_score(parsed.answer, golds)
            bs_ = compute_bridge_score(parsed.bridges, ex.get("bridges", []))
            em_sum += em
            f1_sum += f1
            bridge_sum += bs_
            chain_em_sum += float(em == 1.0 and bs_ >= 0.5)

    n = len(eval_examples)
    unwrapped.config.use_cache = False
    return {
        "eval_em":       em_sum / n * 100,
        "eval_f1":       f1_sum / n * 100,
        "eval_bridge":   bridge_sum / n * 100,
        "eval_chain_em": chain_em_sum / n * 100,
        "eval_n":        n,
    }


# ============================================================
# Logging / saving
# ============================================================

def gather_mean(accelerator, value):
    t = torch.tensor([value], device=accelerator.device, dtype=torch.float32)
    g = accelerator.gather_for_metrics(t)
    return float(g.mean().detach().cpu())


def save_checkpoint(accelerator, model, tokenizer, output_dir, step_name):
    accelerator.wait_for_everyone()
    ckpt_dir = os.path.join(output_dir, step_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    unwrapped.save_pretrained(
        ckpt_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(ckpt_dir)
    accelerator.wait_for_everyone()


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--ref_model_name_or_path", type=str, default=None)
    p.add_argument("--train_file", type=str, default="prepared_data_2hop/train_2hop.jsonl")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--limit_train_examples", type=int, default=None)
    p.add_argument("--include_bridge_count", action="store_true")

    # Train
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_ppo_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-7)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--attn_implementation", type=str, default="sdpa")

    # Generation
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--max_input_len", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=0)

    # GRPO / PPO
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--kl_beta", type=float, default=0.02)
    p.add_argument("--disable_std_norm", action="store_true")

    # Reward
    p.add_argument("--reward_mode", type=str, default="chain_binary",
                   choices=["chain_binary", "process_final", "final_only",
                            "process_only", "format_only"])
    p.add_argument("--w_format", type=float, default=0.10)
    p.add_argument("--w_bridge", type=float, default=0.45)
    p.add_argument("--w_final", type=float, default=0.45)
    p.add_argument("--bridge_gate_floor", type=float, default=0.0)

    # Eval-during-training
    p.add_argument("--eval_file", type=str, default="prepared_data_2hop/eval_2hop.jsonl")
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--eval_subset_size", type=int, default=200)

    # Misc
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_best", action="store_true", default=True)
    p.add_argument("--fix_mistral_regex", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16" if args.bf16 else "no",
    )
    set_seed(args.seed)
    random.seed(args.seed)

    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    tokenizer = load_tokenizer_safely(args.model_name_or_path,
                                      fix_mistral_regex=args.fix_mistral_regex)
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    accelerator.print(f"Loading policy from: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    ref_model = None
    if args.kl_beta > 0:
        ref_path = args.ref_model_name_or_path or args.model_name_or_path
        accelerator.print(f"Loading reference model from: {ref_path}")
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_path,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is not None:
            ref_model.config.pad_token_id = tokenizer.pad_token_id
        ref_model.eval()
        for p_ in ref_model.parameters():
            p_.requires_grad_(False)

    train_ds = JsonlDataset(args.train_file, limit=args.limit_train_examples)
    accelerator.print(f"Loaded {len(train_ds)} training examples from {args.train_file}")

    eval_examples = []
    if args.eval_file and os.path.exists(args.eval_file):
        with open(args.eval_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    eval_examples.append(json.loads(line))
        random.Random(0).shuffle(eval_examples)
        eval_examples = eval_examples[: args.eval_subset_size]
        accelerator.print(f"Will eval on {len(eval_examples)} samples every {args.eval_steps} steps")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_examples,
        num_workers=args.dataloader_num_workers,
        drop_last=False,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    # ★ FIX: 多卡下 accelerate 会把 scheduler.step() 调用 num_processes 次,
    #   把 num_training_steps 乘上 num_processes 让 cosine 周期对齐
    n_proc = accelerator.num_processes
    warmup_steps = int(args.max_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps * n_proc,
        num_training_steps=args.max_steps * n_proc,
    )
    accelerator.print(f"Scheduler: warmup={warmup_steps*n_proc}, total={args.max_steps*n_proc} (n_proc={n_proc})")

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    if ref_model is not None:
        ref_model.to(accelerator.device)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(total=args.max_steps,
                    disable=not accelerator.is_main_process,
                    desc="GRPO")
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    eval_log_path = os.path.join(args.output_dir, "eval_log.jsonl")

    global_step = 0
    best_eval_em = -1.0
    start_time = time.time()
    smoothed_reward = deque(maxlen=50)

    while global_step < args.max_steps:
        for batch_examples in train_loader:
            if global_step >= args.max_steps:
                break

            prompts = [build_prompt(tokenizer, ex,
                                    include_bridge_count=args.include_bridge_count)
                       for ex in batch_examples]

            sequences, attention_mask, completion_mask, texts = generate_completions(
                model, tokenizer, prompts, args, accelerator
            )

            local_bsz = len(batch_examples)
            G = args.num_generations
            assert len(texts) == local_bsz * G

            repeated_examples = []
            for ex in batch_examples:
                repeated_examples.extend([ex] * G)

            # ---- Reward + advantage ----
            rewards, comps_list = [], []
            for text, ex in zip(texts, repeated_examples):
                r, comps = compute_reward(
                    text=text, example=ex,
                    reward_mode=args.reward_mode,
                    w_format=args.w_format,
                    w_bridge=args.w_bridge,
                    w_final=args.w_final,
                    bridge_gate_floor=args.bridge_gate_floor,
                )
                rewards.append(r)
                comps_list.append(comps)

            rewards_t = torch.tensor(
                rewards, dtype=torch.float32, device=accelerator.device
            ).view(local_bsz, G)

            group_mean = rewards_t.mean(dim=1, keepdim=True)
            group_std_full = rewards_t.std(dim=1, keepdim=True, unbiased=False)
            if args.disable_std_norm:
                advantages = (rewards_t - group_mean)
            else:
                advantages = (rewards_t - group_mean) / (group_std_full + 1e-4)
            advantages = advantages.view(-1).detach()

            # ---- 诊断: 多少 prompt 的 group 完全饱和(零方差) ----
            zero_adv_frac = float((group_std_full.squeeze(1) < 1e-6).float().mean().item())
            group_std_mean = float(group_std_full.mean().item())

            # ---- Old logprobs ----
            with torch.no_grad():
                old_logps = token_logprobs(model, sequences, attention_mask).detach()

            # ---- PPO 更新 ----
            for ppo_epoch in range(args.num_ppo_epochs):
                with accelerator.accumulate(model):
                    loss, loss_stats = compute_grpo_loss(
                        model=model,
                        ref_model=ref_model,
                        sequences=sequences,
                        attention_mask=attention_mask,
                        completion_mask=completion_mask,
                        advantages=advantages,
                        old_logps=old_logps,
                        args=args,
                    )
                    if torch.isnan(loss) or torch.isinf(loss):
                        accelerator.print(f"[warn] loss NaN/Inf at step {global_step}, skip")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

            # ---- Logging ----
            local_reward_mean = float(sum(rewards) / max(len(rewards), 1))
            def cm(k):
                return float(sum(c[k] for c in comps_list) / max(len(comps_list), 1))

            local_stats = {
                "loss": float(loss.detach().cpu()) if torch.isfinite(loss) else 0.0,
                "pg_loss": loss_stats["pg_loss"],
                "kl_loss": loss_stats["kl_loss"],
                "clipfrac": loss_stats["clipfrac"],
                "reward_mean": local_reward_mean,
                "reward_max": float(max(rewards)),
                "reward_min": float(min(rewards)),
                "format": cm("format"),
                "bridge": cm("bridge"),
                "answer_f1": cm("answer_f1"),
                "answer_em": cm("answer_em"),
                "chain_em": cm("chain_em"),
                "n_pred_bridge": cm("n_pred_bridge"),
                "has_answer_line": cm("has_answer_line"),
                "group_std": group_std_mean,
                "zero_adv_frac": zero_adv_frac,
                "lr": scheduler.get_last_lr()[0],
            }
            smoothed_reward.append(local_reward_mean)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)

                if global_step % args.logging_steps == 0:
                    agg = {k: gather_mean(accelerator, v) for k, v in local_stats.items()}
                    agg["step"] = global_step
                    agg["elapsed_sec"] = time.time() - start_time
                    if accelerator.is_main_process:
                        msg = (
                            f"step={global_step:04d} "
                            f"loss={agg['loss']:+.4f} "
                            f"pg={agg['pg_loss']:+.4f} "
                            f"kl={agg['kl_loss']:.4f} "
                            f"R={agg['reward_mean']:.3f} "
                            f"std={agg['group_std']:.3f} "
                            f"zero%={agg['zero_adv_frac']*100:.0f} "
                            f"chain={agg['chain_em']:.3f} "
                            f"B={agg['bridge']:.3f} "
                            f"A_EM={agg['answer_em']:.3f} "
                            f"fmt={agg['format']:.3f} "
                            f"nB={agg['n_pred_bridge']:.2f} "
                            f"lr={agg['lr']:.2e}"
                        )
                        print(msg, flush=True)
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(agg, ensure_ascii=False) + "\n")

                # ---- Eval ----
                if (eval_examples and args.eval_steps > 0
                        and global_step % args.eval_steps == 0):
                    eval_metrics = quick_eval(
                        model, tokenizer, eval_examples, args, accelerator,
                        max_eval_examples=args.eval_subset_size,
                    )
                    if accelerator.is_main_process and eval_metrics is not None:
                        eval_metrics["step"] = global_step
                        print(
                            f"  [eval@{global_step}] "
                            f"EM={eval_metrics['eval_em']:.2f} "
                            f"F1={eval_metrics['eval_f1']:.2f} "
                            f"Bridge={eval_metrics['eval_bridge']:.2f} "
                            f"ChainEM={eval_metrics['eval_chain_em']:.2f} "
                            f"(n={eval_metrics['eval_n']})"
                        )
                        with open(eval_log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(eval_metrics, ensure_ascii=False) + "\n")
                        if args.save_best and eval_metrics["eval_em"] > best_eval_em:
                            best_eval_em = eval_metrics["eval_em"]
                            accelerator.print(f"  → New best EM={best_eval_em:.2f}, saving best/")
                            save_checkpoint(accelerator, model, tokenizer,
                                            args.output_dir, "best")

                if global_step % args.save_steps == 0:
                    save_checkpoint(accelerator, model, tokenizer,
                                    args.output_dir, f"checkpoint-{global_step}")

            if global_step >= args.max_steps:
                break

    progress.close()
    save_checkpoint(accelerator, model, tokenizer, args.output_dir, "final")
    accelerator.print(f"Done. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()