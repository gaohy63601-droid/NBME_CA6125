"""
Continue MLM pretraining of deberta-v3-large on patient_notes.csv (~42k notes).
Held-out test pn_num are EXCLUDED to avoid leakage.
DDP-enabled: launch with `torchrun --nproc_per_node=N mlm_pretrain.py ...`.
"""
import os, argparse, time, math, random
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_cosine_schedule_with_warmup, DataCollatorForLanguageModeling
from torch.optim import AdamW

DATA_DIR     = "/raid/yiren/ghy/motion_transfer/medical"
SPLITS_DIR   = "/raid/yiren/ghy/motion_transfer/medical/nbme_baseline/splits"
OUT_DIR      = "/raid/yiren/ghy/motion_transfer/medical/nbme_baseline/ckpt/mlm_backbone"
LOG_DIR      = "/raid/yiren/ghy/motion_transfer/medical/nbme_baseline/logs"
MODEL_NAME   = "microsoft/deberta-v3-large"
MAX_LEN      = 416
BATCH_SIZE   = 8
GRAD_ACC     = 2
LR           = 2e-5
EPOCHS       = 5
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
MLM_PROB     = 0.15
SEED         = 42

os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class NotesDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--bs", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    # DDP setup
    is_dist = "RANK" in os.environ
    if is_dist:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0; world = 1; local_rank = 0
        device = torch.device("cuda")
    is_main = (rank == 0)

    def log(s):
        if is_main: print(s, flush=True)

    notes = pd.read_csv(os.path.join(DATA_DIR, "patient_notes.csv"))
    log(f"all notes: {len(notes)}")

    # Exclude held-out 200 pn_num to prevent test leakage during MLM
    test_split = pd.read_csv(os.path.join(SPLITS_DIR, "test_split.csv"))
    held_pn = set(test_split["pn_num"].unique())
    notes = notes[~notes["pn_num"].isin(held_pn)].reset_index(drop=True)
    log(f"after dropping held-out test pn_num: {len(notes)}")

    texts = notes["pn_history"].astype(str).tolist()
    if args.debug:
        texts = texts[:512]

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = NotesDataset(texts, tok, args.max_len)

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=MLM_PROB)

    def custom_collate(batch):
        # collator wants list of dicts with input_ids; it'll mask & build labels
        return collator(batch)

    if is_dist:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        dl = DataLoader(ds, batch_size=args.bs, sampler=sampler, num_workers=4,
                        collate_fn=custom_collate, pin_memory=True, drop_last=True)
    else:
        sampler = None
        dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=4,
                        collate_fn=custom_collate, pin_memory=True, drop_last=True)

    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, use_safetensors=True).to(device)
    if is_dist:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optim = AdamW(grouped, lr=args.lr)
    total_steps = (len(dl) // GRAD_ACC) * args.epochs
    sched = get_cosine_schedule_with_warmup(optim, int(total_steps * WARMUP_RATIO), total_steps)

    scaler = GradScaler("cuda")

    log_f = open(os.path.join(LOG_DIR, "mlm.log"), "a") if is_main else None
    step = 0
    t0 = time.time()
    best_loss = float("inf")

    for epoch in range(args.epochs):
        if is_dist:
            sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        for it, batch in enumerate(dl):
            ids = batch["input_ids"].to(device, non_blocking=True)
            am  = batch["attention_mask"].to(device, non_blocking=True)
            lab = batch["labels"].to(device, non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=am, labels=lab)
                loss = out.loss

            scaler.scale(loss / GRAD_ACC).backward()
            running += loss.item()

            if (it + 1) % GRAD_ACC == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0 and is_main:
                    avg = running / (it + 1)
                    msg = f"mlm ep {epoch} step {step} loss {avg:.4f} lr {sched.get_last_lr()[0]:.2e} elapsed {time.time()-t0:.0f}s"
                    print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

        ep_loss = running / max(len(dl), 1)
        if is_main:
            msg = f"[EP] mlm ep {epoch} avg_loss {ep_loss:.4f}"
            print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()
            # Save backbone every epoch (overwrite); unwrap DDP if needed
            mod_to_save = model.module if is_dist else model
            mod_to_save.save_pretrained(OUT_DIR, safe_serialization=True)
            tok.save_pretrained(OUT_DIR)
            msg = f"saved backbone to {OUT_DIR} (ep_loss={ep_loss:.4f})"
            print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()
        if is_dist:
            dist.barrier()

    if is_main:
        log_f.close()
    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
