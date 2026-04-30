"""
Phase 2: Confidence-regularization fine-tuning on top of Phase 1 LoRA.

Adds two extra penalties to standard CE loss (computed during teacher forcing):

  1) Hallucination penalty: tokens in the generated assistant answer that have
     no n-gram overlap with the patient note → those positions get higher loss
     weight.
  2) Missing penalty: ground-truth assistant tokens that the model assigns low
     prob to (likely missed) → those positions get higher loss weight.

Implementation: per-token CE × position-aware weight.

  weight = 1.0  + α * has_hallu_signal[t]  + β * is_missing_signal[t]

where (during teacher forcing, no actual generation):
  - has_hallu_signal[t] = 1 if the token at position t is part of the assistant
    span that doesn't appear in the input note (verbatim) → "model would
    hallucinate this if it copies".
  - is_missing_signal[t] = 1 if the model's predicted prob for this gold token
    is < THRESHOLD (e.g. 0.3) → model is "about to miss" this token.

Top NBME paper used: hallucination weight 0.2, missing weight 0.5. We add to the
CE base loss (which already has weight 1).
"""
import os, json, argparse, time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import PeftModel
from torch.optim import AdamW

MODEL_NAME = "mistralai/Mistral-Nemo-Instruct-2407"
DATA_DIR   = "/raid/yiren/ghy/motion_transfer/medical/mistral_nemo/data"
PHASE1_LORA = "/raid/yiren/ghy/motion_transfer/medical/mistral_nemo/ckpt/phase1_lora"
CKPT_DIR    = "/raid/yiren/ghy/motion_transfer/medical/mistral_nemo/ckpt/phase2_lora"
LOG_DIR     = "/raid/yiren/ghy/motion_transfer/medical/mistral_nemo/logs"
MAX_LEN  = 1024
BS       = 1
GRAD_ACC = 8
LR       = 5e-5
EPOCHS   = 2
WARMUP   = 0.05
SEED     = 42

ALPHA_HALLU  = 0.2  # paper value
BETA_MISSING = 0.5  # paper value
MISSING_PROB_THR = 0.30
HALLU_NGRAM = 4  # 4-gram overlap check

os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)


class JsonlChat(Dataset):
    def __init__(self, path, tok, max_len):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok = tok
        self.max_len = max_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        msgs = r["messages"]
        full = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt = self.tok.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True)
        full_ids = self.tok(full, max_length=self.max_len, truncation=True)["input_ids"]
        prompt_ids = self.tok(prompt, max_length=self.max_len, truncation=True)["input_ids"]
        labels = list(full_ids)
        # Hallucination signal: assistant tokens whose 4-gram doesn't appear in user prompt
        prompt_text_lower = msgs[0]["content"].lower()
        # Compute per-token hallu flag for assistant positions
        hallu_flag = [0] * len(full_ids)
        L = len(prompt_ids)
        for t in range(L, len(full_ids)):
            # check 4-gram around this token
            window = self.tok.decode(full_ids[max(L, t - HALLU_NGRAM + 1): t + 1], skip_special_tokens=True).lower().strip()
            if not window: continue
            if window not in prompt_text_lower:
                hallu_flag[t] = 1
        for j in range(min(L, len(labels))):
            labels[j] = -100
        return {"input_ids": full_ids, "labels": labels, "hallu_flag": hallu_flag}


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    ids, lab, am, hf = [], [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = max_len - n
        ids.append(b["input_ids"] + [pad_id] * pad)
        lab.append(b["labels"] + [-100] * pad)
        am.append([1] * n + [0] * pad)
        hf.append(b["hallu_flag"] + [0] * pad)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
            "attention_mask": torch.tensor(am), "hallu_flag": torch.tensor(hf)}


def conf_reg_loss(logits, labels, hallu_flag, alpha=ALPHA_HALLU, beta=BETA_MISSING,
                  miss_thr=MISSING_PROB_THR):
    """
    logits: [B, T, V]
    labels: [B, T] with -100 for ignored positions
    hallu_flag: [B, T] in {0,1}, 1 = potential hallucination
    Returns scalar loss = mean over valid positions of weight * CE(token).
    """
    # shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_hallu  = hallu_flag[:, 1:].contiguous().float()

    B, Tm1, V = shift_logits.shape
    flat_logits = shift_logits.view(-1, V)
    flat_labels = shift_labels.view(-1)
    flat_hallu  = shift_hallu.view(-1)

    valid = (flat_labels != -100)
    if valid.sum() == 0: return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # standard CE per position (no reduction)
    ce = nn.functional.cross_entropy(flat_logits, flat_labels.clamp(min=0),
                                     reduction="none")
    ce = ce * valid.float()

    # missing signal: model prob for gold token < miss_thr → boost
    with torch.no_grad():
        gold_probs = nn.functional.softmax(flat_logits.float(), dim=-1).gather(
            1, flat_labels.clamp(min=0).unsqueeze(1)).squeeze(1)
        miss_flag = ((gold_probs < miss_thr) & valid).float()

    weight = 1.0 + alpha * flat_hallu * valid.float() + beta * miss_flag
    weighted_ce = (ce * weight).sum() / valid.float().sum().clamp(min=1)
    return weighted_ce


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--bs", type=int, default=BS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    parser.add_argument("--phase1_lora", type=str, default=PHASE1_LORA)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    is_dist = "RANK" in os.environ
    if is_dist:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank(); world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world, local_rank = 0, 1, 0
        device = torch.device("cuda")
    is_main = (rank == 0)

    torch.manual_seed(SEED + rank)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    ds = JsonlChat(os.path.join(DATA_DIR, "train_split.jsonl"), tok, args.max_len)
    if args.debug: ds.rows = ds.rows[:64]
    if is_main: print(f"phase2 train rows: {len(ds)}", flush=True)

    if is_dist:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        dl = DataLoader(ds, batch_size=args.bs, sampler=sampler, collate_fn=lambda b: collate(b, tok.pad_token_id),
                        num_workers=2, pin_memory=True, drop_last=True)
    else:
        sampler = None
        dl = DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id),
                        num_workers=2, pin_memory=True, drop_last=True)

    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa")
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, args.phase1_lora, is_trainable=True)
    if is_main: model.print_trainable_parameters()
    model.to(device)

    if is_dist:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                  betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = (len(dl) // GRAD_ACC) * args.epochs
    sched = get_cosine_schedule_with_warmup(optim, int(total_steps * WARMUP), total_steps)

    log_f = open(os.path.join(LOG_DIR, "phase2.log"), "a") if is_main else None
    step = 0; t0 = time.time(); running = 0.0

    for epoch in range(args.epochs):
        if is_dist and sampler is not None: sampler.set_epoch(epoch)
        model.train()
        for it, batch in enumerate(dl):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = conf_reg_loss(out.logits, batch["labels"], batch["hallu_flag"])
            (loss / GRAD_ACC).backward()
            running += loss.item()
            if (it + 1) % GRAD_ACC == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0 and is_main:
                    avg = running / (it + 1)
                    msg = f"phase2 ep {epoch} step {step} loss {avg:.4f} lr {sched.get_last_lr()[0]:.2e} elapsed {time.time()-t0:.0f}s"
                    print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

        if is_main:
            mod = model.module if is_dist else model
            mod.save_pretrained(CKPT_DIR)
            tok.save_pretrained(CKPT_DIR)
            print(f"saved phase2 LoRA to {CKPT_DIR} after epoch {epoch}", flush=True)

    if is_main and log_f is not None: log_f.close()
    if is_dist: dist.destroy_process_group()


if __name__ == "__main__":
    main()
