# rl_common.py
"""Shared utilities for RL training on ATOMIC (1-hop) data only.
Multi-hop transfer is measured at test time, NOT during training."""

import collections, re, string
from typing import Dict, List


# ---------- SQuAD/MuSiQue style metrics ----------

def normalize_answer(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred, gold):
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = collections.Counter(pt) & collections.Counter(gt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p, r = n / len(pt), n / len(gt)
    return 2 * p * r / (p + r)


def em_score(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


# ---------- Atomic reward (no intermediate signal — pure 1-hop) ----------

DEFAULT_REWARD = {
    "f1_coef":         1.0,
    "em_bonus":        0.5,
    "format_penalty":  0.2,
    "max_words":       20,
}


def compute_reward(pred: str, sample: Dict, cfg: Dict = None) -> Dict:
    """Reward for an atomic 1-hop example.
    sample must contain {'answer': str, ...}. No chain assumed."""
    cfg = {**DEFAULT_REWARD, **(cfg or {})}
    gold = sample["answer"]

    f1 = f1_score(pred, gold)
    em = em_score(pred, gold)
    outcome = cfg["f1_coef"] * f1 + cfg["em_bonus"] * em

    n_words = len(pred.split())
    fmt_pen = cfg["format_penalty"] if n_words > cfg["max_words"] else 0.0

    total = outcome - fmt_pen
    return {
        "total": float(total),
        "f1": float(f1),
        "em": float(em),
        "format_penalty": float(fmt_pen),
        "n_words": int(n_words),
    }


# ---------- Prompt format (matches sft_train.py exactly) ----------

DEFAULT_SYSTEM = (
    "You answer questions based strictly on the given passage. "
    "Output only a short, direct answer (a few words), with no explanation."
)


def build_messages(sample: Dict, system: str = DEFAULT_SYSTEM) -> List[Dict]:
    user = (
        f"Passage Title: {sample['context_title']}\n"
        f"Passage: {sample['context']}\n\n"
        f"Question: {sample['question']}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]