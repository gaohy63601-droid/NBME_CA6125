"""
Inference: Mistral Nemo (base + LoRA) generates extracted spans, then we map to
char positions in the original note and compute char-level micro F1 (Kaggle metric).

Generation is done per (note, feature) pair. For each generated span text we
locate it in the original note via simple substring match (case-insensitive,
whitespace-normalized).
"""
import os, json, ast, re, argparse, time, random
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "mistralai/Mistral-Nemo-Instruct-2407"
DATA_DIR   = "/raid/yiren/ghy/motion_transfer/medical/mistral_nemo/data"
PRED_DIR   = "/raid/yiren/ghy/motion_transfer/medical/mistral_nemo/preds"
SPLITS_DIR = "/raid/yiren/ghy/motion_transfer/medical/nbme_baseline/splits"
os.makedirs(PRED_DIR, exist_ok=True)


def parse_locations(loc_str):
    if not isinstance(loc_str, str) or loc_str in ("[]", "", "nan"): return []
    items = ast.literal_eval(loc_str)
    spans = []
    for it in items:
        for piece in it.split(";"):
            piece = piece.strip()
            if not piece: continue
            a, b = piece.split()
            spans.append((int(a), int(b)))
    return spans


def find_substr_spans(note, text):
    """Find ALL char (start, end) positions of `text` in `note`. Whitespace-normalized."""
    if not text or text == "NO_MATCH": return []
    # Normalize whitespace in text but keep original char offsets in note
    # Strategy: try direct substring first, then fuzzy (lower-case + collapsed whitespace)
    spans = []
    if text in note:
        i = 0
        while True:
            j = note.find(text, i)
            if j < 0: break
            spans.append((j, j + len(text)))
            i = j + 1
        return spans
    # Fuzzy: collapse whitespace, lowercase
    note_l = note.lower()
    text_l = re.sub(r"\s+", " ", text.strip().lower())
    if not text_l: return []
    # Build a map from collapsed-pos to original-pos
    norm = []
    map_to_orig = []
    prev_ws = True
    for k, ch in enumerate(note_l):
        if ch.isspace():
            if not prev_ws:
                norm.append(" "); map_to_orig.append(k)
            prev_ws = True
        else:
            norm.append(ch); map_to_orig.append(k)
            prev_ws = False
    norm_str = "".join(norm)
    i = 0
    while True:
        j = norm_str.find(text_l, i)
        if j < 0: break
        end = j + len(text_l)
        if end > len(map_to_orig): break
        os_ = map_to_orig[j]
        oe_ = map_to_orig[min(end - 1, len(map_to_orig) - 1)] + 1
        spans.append((os_, oe_))
        i = j + 1
    return spans


def parse_assistant_output(s):
    """Convert generated string into list of span texts."""
    s = s.strip()
    if not s or s.upper() == "NO_MATCH": return []
    parts = [p.strip() for p in s.split("|||")]
    return [p for p in parts if p and p.upper() != "NO_MATCH"]


def gt_mask(spans, n):
    m = np.zeros(n, dtype=bool)
    for s, e in spans:
        s = max(0, s); e = min(n, e)
        if e > s: m[s:e] = True
    return m


class JsonlPrompts(Dataset):
    def __init__(self, path, tok, max_input):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok = tok
        self.max_input = max_input

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt = self.tok.apply_chat_template([r["messages"][0]], tokenize=False, add_generation_prompt=True)
        return {"id": r["id"], "prompt": prompt, "row_idx": i}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", type=str, default="/raid/yiren/ghy/motion_transfer/medical/mistral_nemo/ckpt/phase1_lora")
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_input", type=int, default=1024)
    parser.add_argument("--out_tag", type=str, default="phase1")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
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

    # Per-rank seed: shift by rank so different ranks see different sampling streams
    # but the (seed, rank) pair is fully reproducible.
    seed = args.seed + rank * 1000003
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa")
    if args.lora and os.path.isdir(args.lora):
        if is_main: print(f"loading LoRA from {args.lora}", flush=True)
        model = PeftModel.from_pretrained(base, args.lora)
    else:
        if is_main: print("WARNING: no LoRA, using base model", flush=True)
        model = base
    model.to(device)
    model.eval()

    ds = JsonlPrompts(os.path.join(DATA_DIR, "test_split.jsonl"), tok, args.max_input)
    if args.debug: ds.rows = ds.rows[:32]
    if is_main: print(f"test rows: {len(ds)}", flush=True)

    # Shard rows across ranks
    rows_for_rank = [ds.rows[i] for i in range(rank, len(ds), world)]

    test_split = pd.read_csv(os.path.join(SPLITS_DIR, "test_split.csv"))
    notes_by_pn = {int(r["pn_num"]): str(r["pn_history"]) for _, r in test_split.drop_duplicates("pn_num").iterrows()}
    locations_by_id = dict(zip(test_split["id"].astype(str), test_split["location"].astype(str)))

    out_local = []
    t0 = time.time()
    for bi in range(0, len(rows_for_rank), args.bs):
        batch = rows_for_rank[bi: bi + args.bs]
        prompts = [tok.apply_chat_template([b["messages"][0]], tokenize=False, add_generation_prompt=True) for b in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_input).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=args.do_sample, num_beams=args.num_beams,
                                 temperature=args.temperature if args.do_sample else 1.0,
                                 top_p=args.top_p if args.do_sample else 1.0,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
        for b, txt in zip(batch, texts):
            out_local.append({"id": b["id"], "pn_num": b["pn_num"], "case_num": b["case_num"],
                              "feature_num": b["feature_num"], "raw_output": txt})
        if is_main and bi % (args.bs * 5) == 0:
            print(f"  rank {rank} {bi+len(batch)}/{len(rows_for_rank)} elapsed {time.time()-t0:.0f}s", flush=True)

    # Gather to rank 0
    all_out = [None] * world if is_dist else [out_local]
    if is_dist:
        dist.barrier()
        # use all_gather_object
        gathered = [None] * world
        dist.all_gather_object(gathered, out_local)
        all_out = gathered

    if not is_main:
        if is_dist: dist.destroy_process_group()
        return

    flat = [x for sub in all_out for x in sub]
    by_id = {x["id"]: x for x in flat}
    if len(by_id) != len(ds):
        print(f"WARN: gathered {len(by_id)} vs expected {len(ds)}", flush=True)

    # Convert outputs to char spans + compute F1
    tp = fp = fn = 0
    rows_out = []
    for ex in ds.rows:
        eid = ex["id"]; pn = ex["pn_num"]
        note = notes_by_pn[pn]
        gen_text = by_id.get(eid, {}).get("raw_output", "")
        span_texts = parse_assistant_output(gen_text)
        pred_spans = []
        for st in span_texts:
            pred_spans.extend(find_substr_spans(note, st))
        gt_spans = parse_locations(locations_by_id.get(eid, "[]"))
        n = len(note)
        gm = gt_mask(gt_spans, n)
        pm = gt_mask(pred_spans, n)
        tp += int((gm & pm).sum())
        fp += int((~gm & pm).sum())
        fn += int((gm & ~pm).sum())
        rows_out.append({"id": eid, "pn_num": pn,
                         "feature_num": ex["feature_num"],
                         "gt_location": locations_by_id.get(eid, ""),
                         "raw_output": gen_text,
                         "pred_location": ";".join(f"{s} {e}" for s, e in pred_spans)})

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"\n[FINAL] P={prec:.4f} R={rec:.4f} F1={f1:.4f}  tp={tp} fp={fp} fn={fn}", flush=True)
    pd.DataFrame(rows_out).to_csv(os.path.join(PRED_DIR, f"preds_{args.out_tag}.csv"), index=False)
    print(f"saved preds_{args.out_tag}.csv", flush=True)

    if is_dist: dist.destroy_process_group()


if __name__ == "__main__":
    main()
