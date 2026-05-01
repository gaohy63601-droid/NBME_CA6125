"""
NBME finetune: deberta-v3-large token classification with NBME top-solution tricks:
  - Layer-wise LR decay (LLRD, decay 0.9)
  - AWP (Adversarial Weight Perturbation) starting epoch 2
  - 5-fold CV via --fold N (uses splits/train_split_5fold.csv)
  - Optional --backbone path for MLM-pretrained checkpoint
  - Optional --extra_csv to mix in pseudo-labeled rows
Loss: BCE per-token over note tokens only.
"""
import os, ast, json, math, random, argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_cosine_schedule_with_warmup
from torch.optim import AdamW

MODEL_NAME   = "microsoft/deberta-v3-large"
SPLITS_DIR   = "/raid/yiren/ghy/motion_transfer/medical/nbme_baseline/splits"
CKPT_DIR     = "/raid/yiren/ghy/motion_transfer/medical/nbme_baseline/ckpt"
LOG_DIR      = "/raid/yiren/ghy/motion_transfer/medical/nbme_baseline/logs"
MAX_LEN      = 416
BATCH_SIZE   = 8
GRAD_ACC     = 2
LR_HEAD      = 1e-4
LR_BACKBONE  = 2e-5
LLRD_DECAY   = 0.9
EPOCHS       = 5
WARMUP_RATIO = 0.0
WEIGHT_DECAY = 0.01
SEED         = 42

# AWP
AWP_LR        = 1e-4
AWP_EPS       = 1e-3
AWP_START_EP  = 2  # 0-indexed: kicks in from epoch 2 onward (i.e. last 3 epochs of 5)

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def parse_locations(loc_str):
    if not isinstance(loc_str, str) or loc_str in ("[]", "", "nan"):
        return []
    items = ast.literal_eval(loc_str)
    spans = []
    for it in items:
        for piece in it.split(";"):
            piece = piece.strip()
            if not piece:
                continue
            a, b = piece.split()
            spans.append((int(a), int(b)))
    return spans


def feature_text_clean(t):
    t = t.replace("-OR-", " or ").replace("-", " ")
    return t


class NBMEDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN, is_train=True):
        self.tok = tokenizer
        self.max_len = max_len
        self.is_train = is_train
        self.rows = df.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        feat = feature_text_clean(str(r["feature_text"]))
        note = str(r["pn_history"])
        spans = parse_locations(r["location"]) if self.is_train else []

        enc = self.tok(
            feat, note,
            max_length=self.max_len,
            padding="max_length",
            truncation="only_second",
            return_offsets_mapping=True,
            return_tensors=None,
        )
        input_ids      = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        offsets        = enc["offset_mapping"]
        seq_ids        = enc.sequence_ids()

        labels = np.full(len(input_ids), -100, dtype=np.int64)
        note_token_mask = np.zeros(len(input_ids), dtype=np.bool_)
        for i, (s, e) in enumerate(offsets):
            if seq_ids[i] != 1:
                continue
            if s == e == 0:
                continue
            note_token_mask[i] = True
            labels[i] = 0
            for gs, ge in spans:
                if s < ge and e > gs:
                    labels[i] = 1
                    break

        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels, dtype=torch.long),
            "offsets":        torch.tensor(offsets, dtype=torch.long),
            "note_mask":      torch.tensor(note_token_mask, dtype=torch.bool),
            "id":             str(r["id"]),
        }


class TokenClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, backbone_path=None):
        super().__init__()
        if backbone_path is not None and os.path.isdir(backbone_path):
            self.backbone = AutoModel.from_pretrained(backbone_path, use_safetensors=True)
        else:
            try:
                self.backbone = AutoModel.from_pretrained(model_name, use_safetensors=True)
            except Exception:
                # Some models (e.g. deberta-v2-xlarge) only ship pytorch_model.bin
                self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        # Multi-sample dropout (5 different masks, mean) — small but stable F1 gain
        self.ms_dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(5)])
        self.head = nn.Linear(h, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        logits = torch.stack([self.head(d(h)) for d in self.ms_dropouts], dim=0).mean(0).squeeze(-1)
        return logits


def collate(batch):
    out = {}
    for k in ["input_ids", "attention_mask", "labels", "offsets", "note_mask"]:
        out[k] = torch.stack([b[k] for b in batch])
    out["id"] = [b["id"] for b in batch]
    return out


def build_llrd_param_groups(model, lr_backbone, lr_head, decay, weight_decay):
    """
    Layer-wise LR decay: bottom encoder layer gets lr_backbone * decay^(N-1),
    top layer gets lr_backbone * decay^0. Embeddings = bottom layer's LR.
    Head + non-encoder modules get lr_head, no LLRD.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    groups = []
    backbone = model.backbone
    n_layers = backbone.config.num_hidden_layers  # deberta-v3-large = 24

    # Embeddings + rel_embeddings get the lowest LR
    emb_lr = lr_backbone * (decay ** n_layers)
    emb_params_decay, emb_params_nodecay = [], []
    for n, p in backbone.named_parameters():
        if not (n.startswith("embeddings.") or n.startswith("encoder.rel_embeddings")):
            continue
        if any(nd in n for nd in no_decay):
            emb_params_nodecay.append(p)
        else:
            emb_params_decay.append(p)
    if emb_params_decay:
        groups.append({"params": emb_params_decay, "lr": emb_lr, "weight_decay": weight_decay})
    if emb_params_nodecay:
        groups.append({"params": emb_params_nodecay, "lr": emb_lr, "weight_decay": 0.0})

    # Encoder layers
    for layer_i in range(n_layers):
        layer_lr = lr_backbone * (decay ** (n_layers - 1 - layer_i))
        prefix = f"encoder.layer.{layer_i}."
        decay_p, nodecay_p = [], []
        for n, p in backbone.named_parameters():
            if not n.startswith(prefix):
                continue
            if any(nd in n for nd in no_decay):
                nodecay_p.append(p)
            else:
                decay_p.append(p)
        if decay_p:
            groups.append({"params": decay_p, "lr": layer_lr, "weight_decay": weight_decay})
        if nodecay_p:
            groups.append({"params": nodecay_p, "lr": layer_lr, "weight_decay": 0.0})

    # encoder.LayerNorm if exists (deberta encoder has no top LayerNorm but guard)
    other_decay, other_nodecay = [], []
    captured = set()
    for g in groups:
        for p in g["params"]:
            captured.add(id(p))
    for n, p in backbone.named_parameters():
        if id(p) in captured:
            continue
        if any(nd in n for nd in no_decay):
            other_nodecay.append(p)
        else:
            other_decay.append(p)
    if other_decay:
        groups.append({"params": other_decay, "lr": lr_backbone, "weight_decay": weight_decay})
    if other_nodecay:
        groups.append({"params": other_nodecay, "lr": lr_backbone, "weight_decay": 0.0})

    # Head + multi-sample dropout (no params) -> head only
    head_decay, head_nodecay = [], []
    for n, p in model.head.named_parameters():
        if any(nd in n for nd in no_decay):
            head_nodecay.append(p)
        else:
            head_decay.append(p)
    if head_decay:
        groups.append({"params": head_decay, "lr": lr_head, "weight_decay": weight_decay})
    if head_nodecay:
        groups.append({"params": head_nodecay, "lr": lr_head, "weight_decay": 0.0})

    return groups


class AWP:
    """
    Adversarial Weight Perturbation. Saves a copy of weights, adds a tiny adversarial
    perturbation aligned with grad sign, recomputes loss/grad on perturbed weights,
    then restores. Top NBME solutions used adv_lr=1e-4, adv_eps=1e-3.
    """
    def __init__(self, model, optim, scaler, adv_lr=AWP_LR, adv_eps=AWP_EPS, adv_param="weight"):
        self.model = model
        self.optim = optim
        self.scaler = scaler
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.adv_param = adv_param
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and self.adv_param in n and p.grad is not None:
                self.backup[n] = p.data.clone()
                grad_eps = self.adv_eps * p.abs().detach()
                self.backup_eps[n] = (p.data - grad_eps, p.data + grad_eps)

    def _attack(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and self.adv_param in n and p.grad is not None and n in self.backup:
                norm1 = torch.norm(p.grad)
                norm2 = torch.norm(p.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * p.grad / (norm1 + 1e-8) * (norm2 + 1e-8)
                    p.data.add_(r_at)
                    p.data = torch.min(torch.max(p.data, self.backup_eps[n][0]), self.backup_eps[n][1])

    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self.backup:
                p.data = self.backup[n]
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, batch, bce, autocast_dtype):
        """Perturb -> fwd/bwd to add adversarial gradient on top of clean grad."""
        self._save()
        self._attack()
        self.optim.zero_grad(set_to_none=True)  # clear clean grad? NO — top sols accumulate
        # Standard AWP keeps original grad and adds adv grad:
        # but the clean grad has already been accumulated by .backward() before attack;
        # we need the ATTACKED forward to produce extra grad on top.
        # Trick: don't zero grad; just .backward() again — autograd accumulates.
        with autocast(device_type="cuda", dtype=autocast_dtype):
            logits = self.model(batch["input_ids"], batch["attention_mask"])
            lab = batch["labels"]
            mask = (lab != -100).float()
            loss = bce(logits, lab.float().clamp(min=0))
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        self.scaler.scale(loss).backward()
        self.restore()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--bs", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr_backbone", type=float, default=LR_BACKBONE)
    parser.add_argument("--lr_head", type=float, default=LR_HEAD)
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    parser.add_argument("--fold", type=int, default=0, help="which fold to hold out as val (0..4)")
    parser.add_argument("--no_awp", action="store_true")
    parser.add_argument("--awp_start_ep", type=int, default=AWP_START_EP)
    parser.add_argument("--backbone", type=str, default=None,
                        help="path to MLM-pretrained backbone dir (HF format)")
    parser.add_argument("--extra_csv", type=str, default=None,
                        help="extra training CSV (e.g. pseudo-labeled), same columns as train_split")
    parser.add_argument("--ckpt_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED, help="base seed; effective = seed + fold")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME,
                        help="HF model id or local path; used for tokenizer + backbone if --backbone not set")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # DDP setup (torchrun sets RANK / WORLD_SIZE / LOCAL_RANK env vars)
    is_dist = "RANK" in os.environ
    if is_dist:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world, local_rank = 0, 1, 0
        device = torch.device("cuda")
    is_main = (rank == 0)

    set_seed(args.seed + args.fold + rank)  # different seed per rank for diff dropout

    tok = AutoTokenizer.from_pretrained(args.model_name)

    df_full = pd.read_csv(os.path.join(SPLITS_DIR, "train_split_5fold.csv"))
    df_tr = df_full[df_full["fold"] != args.fold].reset_index(drop=True)
    df_val = df_full[df_full["fold"] == args.fold].reset_index(drop=True)
    print(f"fold {args.fold}: train={len(df_tr)} val={len(df_val)}", flush=True)

    if args.extra_csv is not None and os.path.exists(args.extra_csv):
        extra = pd.read_csv(args.extra_csv)
        print(f"adding extra (pseudo) rows: {len(extra)}", flush=True)
        df_tr = pd.concat([df_tr, extra], ignore_index=True)

    if args.debug:
        df_tr = df_tr.head(256); df_val = df_val.head(64)

    ds_tr = NBMEDataset(df_tr, tok, max_len=args.max_len, is_train=True)
    ds_val = NBMEDataset(df_val, tok, max_len=args.max_len, is_train=True)
    if is_dist:
        sampler_tr = DistributedSampler(ds_tr, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        dl_tr = DataLoader(ds_tr, batch_size=args.bs, sampler=sampler_tr,
                           num_workers=4, collate_fn=collate, pin_memory=True, drop_last=True)
    else:
        sampler_tr = None
        dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True,
                           num_workers=4, collate_fn=collate, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.bs, shuffle=False,
                        num_workers=2, collate_fn=collate, pin_memory=True)

    model = TokenClassifier(model_name=args.model_name, backbone_path=args.backbone).to(device)
    if is_dist:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    base_for_groups = model.module if is_dist else model
    groups = build_llrd_param_groups(base_for_groups, args.lr_backbone, args.lr_head, LLRD_DECAY, WEIGHT_DECAY)
    optim = AdamW(groups)

    total_steps = (len(dl_tr) // GRAD_ACC) * args.epochs
    sched = get_cosine_schedule_with_warmup(
        optim, int(total_steps * WARMUP_RATIO), total_steps)

    scaler = GradScaler("cuda")
    bce = nn.BCEWithLogitsLoss(reduction="none")

    awp = None if args.no_awp else AWP(model, optim, scaler)

    ckpt_name = args.ckpt_name or f"fold{args.fold}.pt"
    log_path = os.path.join(LOG_DIR, ckpt_name.replace(".pt", ".log"))
    log_f = open(log_path, "a") if is_main else None

    best_val = float("inf")
    step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        if is_dist and sampler_tr is not None:
            sampler_tr.set_epoch(epoch)
        model.train()
        running = 0.0
        for it, batch in enumerate(dl_tr):
            batch_dev = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            ids = batch_dev["input_ids"]; am = batch_dev["attention_mask"]; lab = batch_dev["labels"]

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(ids, am)
                lab_f = lab.float()
                mask = (lab != -100).float()
                loss = bce(logits, lab_f.clamp(min=0))
                loss = (loss * mask).sum() / mask.sum().clamp(min=1)

            scaler.scale(loss / GRAD_ACC).backward()
            running += loss.item()

            # AWP: add adv grad on top of clean grad before optim.step()
            if awp is not None and epoch >= args.awp_start_ep and (it + 1) % GRAD_ACC == 0:
                awp.attack_backward(batch_dev, bce, torch.bfloat16)

            if (it + 1) % GRAD_ACC == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0 and is_main:
                    avg = running / (it + 1)
                    msg = (f"fold {args.fold} ep {epoch} step {step} loss {avg:.4f} "
                           f"lr_head {sched.get_last_lr()[-1]:.2e} elapsed {time.time()-t0:.0f}s")
                    print(msg, flush=True)
                    log_f.write(msg + "\n"); log_f.flush()

        # validation on ALL ranks (avoids the rank-0-only sync deadlock with dist.barrier)
        model.eval()
        v_loss, v_n = 0.0, 0
        with torch.no_grad():
            for batch in dl_val:
                ids = batch["input_ids"].to(device, non_blocking=True)
                am  = batch["attention_mask"].to(device, non_blocking=True)
                lab = batch["labels"].to(device, non_blocking=True)
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(ids, am)
                    lab_f = lab.float()
                    mask = (lab != -100).float()
                    loss = bce(logits, lab_f.clamp(min=0))
                    loss = (loss * mask).sum() / mask.sum().clamp(min=1)
                v_loss += loss.item() * mask.sum().item()
                v_n    += mask.sum().item()
        val_loss = v_loss / max(v_n, 1)
        if is_main:
            msg = f"[VAL] fold {args.fold} ep {epoch} val_loss {val_loss:.4f}"
            print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()
            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
                state_dict_to_save = model.module.state_dict() if is_dist else model.state_dict()
                torch.save({"model": state_dict_to_save,
                            "epoch": epoch, "val_loss": val_loss,
                            "model_name": args.model_name,
                            "fold": args.fold,
                            "backbone_path": args.backbone}, ckpt_path)
                msg = f"saved {ckpt_path} (val_loss={val_loss:.4f})"
                print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

    if log_f is not None:
        log_f.close()
    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
