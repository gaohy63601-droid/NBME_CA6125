"""
Phase 1: Instruction SFT of Mistral Nemo 12B Instruct on NBME train_split.
Uses LoRA (r=16, alpha=32) over q/k/v/o + gate/up/down projections.
DDP via torchrun.
"""
import os, json, argparse, time, math
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.amp import GradScaler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from torch.optim import AdamW

MODEL_NAME = "mistralai/Mistral-Nemo-Instruct-2407"
DATA_DIR   = "/raid/yiren/ghy/motion_transfer/medical/mistral_nemo/data"
CKPT_DIR   = "/raid/yiren/ghy/motion_transfer/medical/mistral_nemo/ckpt/phase1_lora"
LOG_DIR    = "/raid/yiren/ghy/motion_transfer/medical/mistral_nemo/logs"
MAX_LEN    = 1024
LR         = 2e-4
EPOCHS     = 2
BS         = 1
GRAD_ACC   = 8
WARMUP     = 0.03
LORA_R     = 16
LORA_ALPHA = 32
LORA_DROP  = 0.05
SEED       = 42

os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)


class JsonlChat(Dataset):
    def __init__(self, path, tok, max_len):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok = tok
        self.max_len = max_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        msgs = self.rows[i]["messages"]
        # full chat template input
        full = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        # prompt portion (just user) so we can mask labels for prompt tokens (only train on assistant)
        prompt = self.tok.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True)
        full_ids = self.tok(full, max_length=self.max_len, truncation=True, return_tensors=None)["input_ids"]
        prompt_ids = self.tok(prompt, max_length=self.max_len, truncation=True, return_tensors=None)["input_ids"]
        labels = list(full_ids)
        for j in range(min(len(prompt_ids), len(labels))):
            labels[j] = -100
        return {"input_ids": full_ids, "labels": labels}


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    ids, lab, am = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = max_len - n
        ids.append(b["input_ids"] + [pad_id] * pad)
        lab.append(b["labels"] + [-100] * pad)
        am.append([1] * n + [0] * pad)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
            "attention_mask": torch.tensor(am)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--bs", type=int, default=BS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
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
    if is_main: print(f"train rows: {len(ds)}", flush=True)

    if is_dist:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        dl = DataLoader(ds, batch_size=args.bs, sampler=sampler, collate_fn=lambda b: collate(b, tok.pad_token_id),
                        num_workers=2, pin_memory=True, drop_last=True)
    else:
        sampler = None
        dl = DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id),
                        num_workers=2, pin_memory=True, drop_last=True)

    # Load base in bf16 (no quantization needed - Mistral 12B fits in 24GB vram in bf16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROP,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type=TaskType.CAUSAL_LM, bias="none")
    model = get_peft_model(model, lora)
    if is_main: model.print_trainable_parameters()
    model.to(device)

    if is_dist:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                  betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = (len(dl) // GRAD_ACC) * args.epochs
    sched = get_cosine_schedule_with_warmup(optim, int(total_steps * WARMUP), total_steps)

    log_f = open(os.path.join(LOG_DIR, "phase1.log"), "a") if is_main else None
    step = 0; t0 = time.time(); running = 0.0

    for epoch in range(args.epochs):
        if is_dist and sampler is not None: sampler.set_epoch(epoch)
        model.train()
        for it, batch in enumerate(dl):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / GRAD_ACC
            loss.backward()
            running += out.loss.item()
            if (it + 1) % GRAD_ACC == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0 and is_main:
                    avg = running / (it + 1)
                    msg = f"phase1 ep {epoch} step {step} loss {avg:.4f} lr {sched.get_last_lr()[0]:.2e} elapsed {time.time()-t0:.0f}s"
                    print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

        if is_main:
            mod = model.module if is_dist else model
            mod.save_pretrained(CKPT_DIR)
            tok.save_pretrained(CKPT_DIR)
            print(f"saved LoRA to {CKPT_DIR} after epoch {epoch}", flush=True)

    if is_main and log_f is not None: log_f.close()
    if is_dist: dist.destroy_process_group()


if __name__ == "__main__":
    main()
