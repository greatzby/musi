# rl_grpo.py
"""GRPO / PG fine-tuning on ATOMIC (1-hop) data, on top of SFT.

Core experimental claim: training distribution = SFT's atomic data, but RL
yields better multi-hop test accuracy than SFT — so RL learns better
*compositional primitives*, not better data coverage.

Algorithm variants (selected via CLI):
  GRPO (default):
      --rollouts_per_prompt 4 --ppo_epochs 2 --clip_eps 0.2
  PG / Q-learning style (single rollout, no clip):
      --rollouts_per_prompt 1 --ppo_epochs 1 --clip_eps 1e6
  No-anchor ablation (paper Sec 5.2, expected to collapse):
      --kl_coef 0.0 --kl_min 0.0
"""

import argparse, json, os, random, time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

from rl_common import build_messages, compute_reward


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_ckpt",   default="checkpoints/qwen2.5-3b-atomic-sft")
    p.add_argument("--train_file", default="prepared_data/atomic_train.jsonl")
    p.add_argument("--output_dir", default="checkpoints/qwen2.5-3b-grpo")

    # optimization
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--num_steps", type=int, default=1500)
    p.add_argument("--prompts_per_step", type=int, default=4)
    p.add_argument("--rollouts_per_prompt", type=int, default=4)   # G in GRPO
    p.add_argument("--ppo_epochs", type=int, default=2)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)

    # GRPO / PPO
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--kl_coef", type=float, default=0.05)
    p.add_argument("--kl_anneal_steps", type=int, default=1000)
    p.add_argument("--kl_min", type=float, default=0.01)

    # generation
    p.add_argument("--max_new_tokens", type=int, default=32)   # atomic ans short
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temp_min", type=float, default=0.7)
    p.add_argument("--temp_anneal_steps", type=int, default=1000)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_prompt_len", type=int, default=1024)

    # reward
    p.add_argument("--reward_f1_coef", type=float, default=1.0)
    p.add_argument("--reward_em_bonus", type=float, default=0.5)
    p.add_argument("--reward_format_penalty", type=float, default=0.2)

    # checkpointing — densely!
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--log_every",  type=int, default=5)
    p.add_argument("--save_total_limit", type=int, default=0)  # 0 = keep all

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # filtering very long passages to save compute
    p.add_argument("--max_passage_words", type=int, default=350)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------- data ----------

class AtomicDataset(Dataset):
    def __init__(self, path, max_passage_words=None):
        with open(path) as f:
            data = [json.loads(l) for l in f if l.strip()]
        if max_passage_words:
            data = [d for d in data if len(d.get("context","").split()) <= max_passage_words]
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


# ---------- generation ----------

@torch.no_grad()
def generate_rollouts(policy, tokenizer, prompts, n_per_prompt,
                      max_new_tokens, temperature, top_p, max_prompt_len):
    device = next(policy.parameters()).device
    policy.eval()

    repeated = [p for p in prompts for _ in range(n_per_prompt)]
    enc = tokenizer(
        repeated, return_tensors="pt", padding=True, truncation=True,
        max_length=max_prompt_len, add_special_tokens=False,
    ).to(device)

    out = policy.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    rollouts = []
    padded_len = enc["input_ids"].size(1)
    for i in range(out.size(0)):
        prompt_idx = i // n_per_prompt
        attn_i = enc["attention_mask"][i]
        real_p_len = int(attn_i.sum().item())
        # left-padded layout: real prompt occupies [padded_len - real_p_len, padded_len)
        real_prompt = enc["input_ids"][i, padded_len - real_p_len:]
        gen_ids = out[i, padded_len:]
        # strip trailing pads
        nonpad = (gen_ids != tokenizer.pad_token_id).nonzero()
        if len(nonpad):
            gen_ids = gen_ids[:nonpad.max().item() + 1]
        else:
            gen_ids = gen_ids[:0]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        rollouts.append({
            "prompt_idx":   prompt_idx,
            "prompt_ids":   real_prompt.cpu(),
            "response_ids": gen_ids.cpu(),
            "response_text": text,
        })
    return rollouts


# ---------- packing & log-probs ----------

def pack_batch(rollouts, pad_id, device):
    """Right-pad full sequences (prompt+response) for forward pass."""
    full_seqs, prompt_lens, full_lens = [], [], []
    for r in rollouts:
        full = torch.cat([r["prompt_ids"], r["response_ids"]])
        full_seqs.append(full)
        prompt_lens.append(r["prompt_ids"].size(0))
        full_lens.append(full.size(0))

    B, T = len(full_seqs), max(full_lens)
    input_ids     = torch.full((B, T), pad_id, dtype=torch.long)
    attn_mask     = torch.zeros((B, T), dtype=torch.long)
    response_mask = torch.zeros((B, T), dtype=torch.bool)
    for i, s in enumerate(full_seqs):
        L = s.size(0)
        input_ids[i, :L]     = s
        attn_mask[i, :L]     = 1
        response_mask[i, prompt_lens[i]:L] = True
    # logits[t] predicts token[t+1] → shift
    shifted = response_mask[:, 1:]
    return input_ids.to(device), attn_mask.to(device), shifted.to(device)


def token_logp(model, input_ids, attn_mask):
    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    log_probs = F.log_softmax(logits, dim=-1)
    target = input_ids[:, 1:]
    return log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)


# ---------- main ----------

def main():
    args = parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.sft_ckpt, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # base SFT model + LoRA
    print(f"Loading SFT model from {args.sft_ckpt}")
    base = AutoModelForCausalLM.from_pretrained(
        args.sft_ckpt,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    policy = get_peft_model(base, lora_cfg).cuda()
    policy.print_trainable_parameters()

    optim = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95),
    )
    sched = get_cosine_schedule_with_warmup(
        optim, int(args.num_steps * args.warmup_ratio), args.num_steps
    )

    dataset = AtomicDataset(args.train_file, args.max_passage_words)
    print(f"Train data: {len(dataset)} atomic examples")

    reward_cfg = {
        "f1_coef":        args.reward_f1_coef,
        "em_bonus":       args.reward_em_bonus,
        "format_penalty": args.reward_format_penalty,
        "max_words":      20,
    }

    # logging
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    log_f = open(log_path, "a")
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    cursor = 0
    device = next(policy.parameters()).device
    saved_steps = []

    for step in range(1, args.num_steps + 1):
        t0 = time.time()

        # 1) sample atomic prompts
        if cursor + args.prompts_per_step > len(indices):
            random.shuffle(indices); cursor = 0
        batch_idx = indices[cursor: cursor + args.prompts_per_step]
        cursor += args.prompts_per_step
        samples = [dataset[i] for i in batch_idx]

        prompts = [
            tokenizer.apply_chat_template(
                build_messages(s), tokenize=False, add_generation_prompt=True
            )
            for s in samples
        ]

        # 2) anneal temperature & KL coef
        if step < args.temp_anneal_steps:
            cur_temp = args.temperature - (args.temperature - args.temp_min) * (step / args.temp_anneal_steps)
        else:
            cur_temp = args.temp_min
        if step < args.kl_anneal_steps:
            cur_kl = args.kl_coef - (args.kl_coef - args.kl_min) * (step / args.kl_anneal_steps)
        else:
            cur_kl = args.kl_min

        # 3) rollouts
        rollouts = generate_rollouts(
            policy, tokenizer, prompts,
            n_per_prompt=args.rollouts_per_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=cur_temp, top_p=args.top_p,
            max_prompt_len=args.max_prompt_len,
        )
        rollouts = [r for r in rollouts if r["response_ids"].numel() > 0]
        if not rollouts:
            print(f"[step {step}] no valid rollouts, skip"); continue

        # 4) reward each rollout
        for r in rollouts:
            r["reward"] = compute_reward(r["response_text"],
                                          samples[r["prompt_idx"]],
                                          reward_cfg)

        # 5) GRPO group-normalized advantage
        for i in range(args.prompts_per_step):
            grp = [r for r in rollouts if r["prompt_idx"] == i]
            if len(grp) < 2:
                # rollouts=1 → REINFORCE-like fallback
                for g in grp: g["advantage"] = g["reward"]["total"]
                continue
            rs = torch.tensor([g["reward"]["total"] for g in grp])
            if rs.std() < 1e-6:
                # all rollouts identical (e.g. all wrong with F1=0) → no learning signal
                for g in grp: g["advantage"] = 0.0
            else:
                adv = (rs - rs.mean()) / (rs.std() + 1e-8)
                for g, a in zip(grp, adv): g["advantage"] = float(a)

        # 6) pack
        input_ids, attn_mask, resp_mask = pack_batch(
            rollouts, tokenizer.pad_token_id, device
        )
        advantages = torch.tensor(
            [r["advantage"] for r in rollouts], device=device, dtype=torch.float32
        )

        # 7) old & ref log-probs (frozen)
        policy.eval()
        with torch.no_grad():
            old_logp = token_logp(policy, input_ids, attn_mask).detach()
            with policy.disable_adapter():           # ref = SFT base
                ref_logp = token_logp(policy, input_ids, attn_mask).detach()

        # 8) PPO-clipped update with KL anchor to ref
        policy.train()
        ppo_losses, ppo_kls, clip_fracs = [], [], []
        for _ in range(args.ppo_epochs):
            new_logp = token_logp(policy, input_ids, attn_mask)
            ratio = torch.exp(new_logp - old_logp)
            adv = advantages.unsqueeze(1).expand_as(ratio)

            unclipped = ratio * adv
            clipped   = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv
            pg_loss = -torch.min(unclipped, clipped)

            # KL(π_new || π_ref), Schulman k3 estimator (always >= 0)
            log_r = ref_logp - new_logp
            kl_per_tok = torch.exp(log_r) - 1.0 - log_r

            loss_per_tok = pg_loss + cur_kl * kl_per_tok
            denom = resp_mask.sum().clamp(min=1).float()
            loss = (loss_per_tok * resp_mask).sum() / denom

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optim.step()

            with torch.no_grad():
                cf = ((ratio < 1 - args.clip_eps) | (ratio > 1 + args.clip_eps)).float()
                cf = (cf * resp_mask.float()).sum() / denom
            ppo_losses.append(loss.item())
            ppo_kls.append(((kl_per_tok * resp_mask).sum() / denom).item())
            clip_fracs.append(cf.item())
        sched.step()

        # 9) log
        avg = lambda key: sum(r["reward"][key] for r in rollouts) / len(rollouts)
        rec = {
            "step": step, "time_s": time.time() - t0,
            "lr": sched.get_last_lr()[0], "temp": cur_temp, "kl_coef": cur_kl,
            "reward_total": avg("total"),
            "reward_f1":    avg("f1"),
            "reward_em":    avg("em"),
            "format_penalty": avg("format_penalty"),
            "avg_n_words":  sum(r["reward"]["n_words"] for r in rollouts)/len(rollouts),
            "advantage_std": float(advantages.std()),
            "ppo_loss":  sum(ppo_losses)/len(ppo_losses),
            "kl_to_ref": sum(ppo_kls)/len(ppo_kls),
            "clip_frac": sum(clip_fracs)/len(clip_fracs),
            "n_rollouts": len(rollouts),
        }
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()

        if step % args.log_every == 0:
            print(f"step={step:4d} R={rec['reward_total']:+.3f} "
                  f"(F1={rec['reward_f1']:.3f} EM={rec['reward_em']:.3f}) "
                  f"loss={rec['ppo_loss']:+.4f} kl={rec['kl_to_ref']:+.4f} "
                  f"clip={rec['clip_frac']:.2f} temp={cur_temp:.2f} "
                  f"klc={cur_kl:.3f} ({rec['time_s']:.1f}s)")

        # 10) save (LoRA adapter only, ~50MB)
        if step % args.save_every == 0:
            ckpt_dir = os.path.join(args.output_dir, f"step-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            policy.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            saved_steps.append(step)
            if args.save_total_limit > 0 and len(saved_steps) > args.save_total_limit:
                old_step = saved_steps.pop(0)
                os.system(f"rm -rf {os.path.join(args.output_dir, f'step-{old_step}')}")
            print(f"  → saved {ckpt_dir}")

    log_f.close()
    print("Done.")


if __name__ == "__main__":
    main()