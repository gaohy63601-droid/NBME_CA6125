"""Per-case postproc on top of 5-way per-case best masks."""
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


def parse_pred(s):
    if not isinstance(s, str) or s in ("", "nan"): return []
    spans = []
    for piece in s.split(";"):
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


# Load
r1npz = np.load(os.path.join(PRED_DIR_R1, "char_probs_r1_dump.npz"), allow_pickle=True)
xlnpz = np.load(os.path.join(PRED_DIR_R1, "char_probs_xl.npz"), allow_pickle=True)
ids = r1npz["ids"]; r1_probs = r1npz["char_probs"]; xl_probs = xlnpz["char_probs"]
note_lens = r1npz["note_lens"]; cases = r1npz["case_nums"]; locations = r1npz["locations"]
n = len(ids)

ms = {}
for name, fn in [("ms1", "preds_phase1_5ep.csv"), ("ms2", "preds_phase2.csv"), ("ms3", "preds_phase1_5ep_beam4.csv")]:
    loc = dict(zip(pd.read_csv(os.path.join(PRED_DIR_MS, fn))["id"].astype(str),
                   pd.read_csv(os.path.join(PRED_DIR_MS, fn))["pred_location"].fillna("").astype(str)))
    ms[name] = [gt_mask(parse_pred(loc.get(str(ids[i]), "")), int(note_lens[i])) for i in range(n)]
gms = [gt_mask(parse_loc(str(l)), int(note_lens[i])) for i, l in enumerate(locations)]

# 5-way per-case configs (from previous run)
PER_CASE_CFG = {
    0: {"w_xl": 0.0, "thr": 0.70, "ms": "ms1&ms2",  "soft": 0.05},
    1: {"w_xl": 0.6, "thr": 0.58, "ms": "ms1&ms2",  "soft": 0.25},
    2: {"w_xl": 0.0, "thr": 0.42, "ms": "ms1",       "soft": 0.35},
    3: {"w_xl": 0.8, "thr": 0.74, "ms": "ms1",       "soft": 0.25},
    4: {"w_xl": 0.4, "thr": 0.70, "ms": "ms1",       "soft": 0.45},
    5: {"w_xl": 0.0, "thr": 0.62, "ms": "ms2|ms3",  "soft": 0.25},
    6: {"w_xl": 0.8, "thr": 0.54, "ms": "ms3",       "soft": 0.35},
    7: {"w_xl": 0.6, "thr": 0.70, "ms": "ms1|ms3",  "soft": 0.05},
    8: {"w_xl": 0.2, "thr": 0.62, "ms": "ms1",       "soft": 0.35},
    9: {"w_xl": 0.0, "thr": 0.78, "ms": "ms2",       "soft": 0.35},
}


def get_extra(combo, i):
    if combo == "ms1": return ms["ms1"][i]
    if combo == "ms2": return ms["ms2"][i]
    if combo == "ms3": return ms["ms3"][i]
    if combo == "ms1|ms2": return ms["ms1"][i] | ms["ms2"][i]
    if combo == "ms1|ms3": return ms["ms1"][i] | ms["ms3"][i]
    if combo == "ms2|ms3": return ms["ms2"][i] | ms["ms3"][i]
    if combo == "ms1|ms2|ms3": return ms["ms1"][i] | ms["ms2"][i] | ms["ms3"][i]
    if combo == "ms1&ms2": return ms["ms1"][i] & ms["ms2"][i]
    if combo == "ms1&ms3": return ms["ms1"][i] & ms["ms3"][i]
    if combo == "ms2&ms3": return ms["ms2"][i] & ms["ms3"][i]
    if combo == "ms1&ms2&ms3": return ms["ms1"][i] & ms["ms2"][i] & ms["ms3"][i]


def build(i, c):
    cfg = PER_CASE_CFG[c]
    comb = (1 - cfg["w_xl"]) * r1_probs[i].astype(np.float32) + cfg["w_xl"] * xl_probs[i].astype(np.float32)
    rm = comb >= cfg["thr"]
    if cfg["ms"] is None: return rm
    soft = comb >= cfg["soft"]
    return rm | (get_extra(cfg["ms"], i) & soft)


base = [build(i, int(cases[i])) for i in range(n)]
print(f"loaded {n} rows")
print(f"\n[5-way per-case base] F1 = {f1(gms, base):.4f}")

cases_uni = sorted(set(cases.tolist()))
case_idx = {c: np.where(cases == c)[0] for c in cases_uni}
overall = [None] * n
print("\n--- per-case postproc on 5-way ---")
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

print(f"\n[5-WAY + per-case postproc OVERALL] F1 = {f1(gms, overall):.4f}")
print(f"\nbaseline:  5-way no postproc:  0.8783")
