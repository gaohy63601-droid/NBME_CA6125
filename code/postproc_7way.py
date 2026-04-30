"""Per-case postproc on top of 7-way per-case best masks.
Reads per_case_7way_dump.npz (written by per_case_7way.py)."""
import os, ast
import numpy as np
import pandas as pd

PRED_DIR_R1 = "/raid/yiren/ghy/motion_transfer/medical/nbme_baseline/preds"
PRED_DIR_MS = "/raid/yiren/ghy/motion_transfer/medical/mistral_nemo/preds"


def parse_loc(s):
    if not isinstance(s, str) or s in ("[]", "", "nan"): return []
    spans = []
    try: items = ast.literal_eval(s)
    except: items = s.split(";")
    for it in items:
        for piece in (it.split(";") if isinstance(it, str) else [str(it)]):
            piece = piece.strip()
            if not piece: continue
            try: a, b = piece.split(); spans.append((int(a), int(b)))
            except: pass
    return spans


def gt_mask(spans, n):
    m = np.zeros(n, dtype=bool)
    for s, e in spans:
        s = max(0, s); e = min(n, e)
        if e > s: m[s:e] = True
    return m


def f1(gms, pms):
    tp = sum(int((g & p).sum()) for g, p in zip(gms, pms))
    fp = sum(int((~g & p).sum()) for g, p in zip(gms, pms))
    fn = sum(int((g & ~p).sum()) for g, p in zip(gms, pms))
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return 2 * P * R / max(P + R, 1e-9)


def mask_to_spans(m):
    spans = []; i = 0; n = len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]: j += 1
            spans.append((i, j)); i = j
        else: i += 1
    return spans


def postproc(m, dilate=0, merge_gap=0, min_len=0):
    if not m.any(): return m
    spans = mask_to_spans(m); n = len(m)
    if dilate > 0:
        spans = [(max(0, s - dilate), min(n, e + dilate)) for s, e in spans]
        spans = sorted(spans); merged = [spans[0]]
        for s, e in spans[1:]:
            if s <= merged[-1][1]: merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else: merged.append((s, e))
        spans = merged
    if merge_gap > 0 and len(spans) > 1:
        spans = sorted(spans); merged = [spans[0]]
        for s, e in spans[1:]:
            if s - merged[-1][1] <= merge_gap:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else: merged.append((s, e))
        spans = merged
    if min_len > 0:
        spans = [(s, e) for s, e in spans if (e - s) >= min_len]
    out = np.zeros(n, dtype=bool)
    for s, e in spans:
        out[s:e] = True
    return out


dump = np.load(os.path.join(PRED_DIR_MS, "per_case_7way_dump.npz"), allow_pickle=True)
overall_in = list(dump["overall"])
note_lens = dump["note_lens"]; cases = dump["cases"]; ids = dump["ids"]
locations = dump["locations"]
n = len(ids)

base = [np.asarray(m, dtype=bool) for m in overall_in]
gms = [gt_mask(parse_loc(str(l)), int(note_lens[i])) for i, l in enumerate(locations)]

print(f"loaded {n} rows")
print(f"\n[7-way per-case base] F1 = {f1(gms, base):.4f}")

cases_uni = sorted(set(cases.tolist()))
case_idx = {c: np.where(cases == c)[0] for c in cases_uni}
overall = [None] * n
print("\n--- per-case postproc on 7-way ---")
for c in cases_uni:
    idx = case_idx[c]
    base_c = [base[i] for i in idx]
    gt_c = [gms[i] for i in idx]
    best = (0, 0, 0, f1(gt_c, base_c))
    for d in [0, 1, 2, 3, 4, 5]:
        for g in [0, 1, 2, 3, 5, 8, 10, 15]:
            for ml in [0, 2, 3, 5, 7]:
                pms = [postproc(m, d, g, ml) for m in base_c]
                f_ = f1(gt_c, pms)
                if f_ > best[3]: best = (d, g, ml, f_)
    print(f"  case {c}: d={best[0]} g={best[1]} ml={best[2]} F1={best[3]:.4f}")
    for i in idx:
        overall[i] = postproc(base[i], best[0], best[1], best[2])

print(f"\n[7-WAY + per-case postproc OVERALL] F1 = {f1(gms, overall):.4f}")
print(f"\nbaseline:  5-way + postproc:  0.8785")
