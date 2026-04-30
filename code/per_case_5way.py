"""5-way per-case ensemble: r1 + xl + ms1(greedy 5ep) + ms2(phase2) + ms3(beam4 5ep)."""
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


r1npz = np.load(os.path.join(PRED_DIR_R1, "char_probs_r1_dump.npz"), allow_pickle=True)
xlnpz = np.load(os.path.join(PRED_DIR_R1, "char_probs_xl.npz"), allow_pickle=True)
ids = r1npz["ids"]
r1_probs = r1npz["char_probs"]; xl_probs = xlnpz["char_probs"]
note_lens = r1npz["note_lens"]; cases = r1npz["case_nums"]; locations = r1npz["locations"]
n = len(ids)

ms_files = {"ms1": "preds_phase1_5ep.csv", "ms2": "preds_phase2.csv", "ms3": "preds_phase1_5ep_beam4.csv"}
ms_masks = {}
for name, fn in ms_files.items():
    loc = dict(zip(pd.read_csv(os.path.join(PRED_DIR_MS, fn))["id"].astype(str),
                   pd.read_csv(os.path.join(PRED_DIR_MS, fn))["pred_location"].fillna("").astype(str)))
    ms_masks[name] = [gt_mask(parse_pred(loc.get(str(ids[i]), "")), int(note_lens[i])) for i in range(n)]
gms = [gt_mask(parse_loc(str(l)), int(note_lens[i])) for i, l in enumerate(locations)]
print(f"loaded {n} rows + 3 ms predictions")

cases_uni = sorted(set(cases.tolist()))
case_idx = {c: np.where(cases == c)[0] for c in cases_uni}

# Per-char vote-count: count how many of {ms1,ms2,ms3} say "yes" at each position.
# Then for combined ensemble: include char if (r1+xl prob >= thr) OR (vote_count >= K AND r1+xl >= soft).
ms_combo_options = ["ms1", "ms2", "ms3", "ms1|ms2", "ms1|ms3", "ms2|ms3", "ms1|ms2|ms3",
                    "ms1&ms2", "ms1&ms3", "ms2&ms3", "ms1&ms2&ms3",
                    "vote>=1", "vote>=2", "vote>=3"]


def get_extra_mask(combo, i):
    if combo == "ms1": return ms_masks["ms1"][i]
    if combo == "ms2": return ms_masks["ms2"][i]
    if combo == "ms3": return ms_masks["ms3"][i]
    if combo == "ms1|ms2": return ms_masks["ms1"][i] | ms_masks["ms2"][i]
    if combo == "ms1|ms3": return ms_masks["ms1"][i] | ms_masks["ms3"][i]
    if combo == "ms2|ms3": return ms_masks["ms2"][i] | ms_masks["ms3"][i]
    if combo == "ms1|ms2|ms3": return ms_masks["ms1"][i] | ms_masks["ms2"][i] | ms_masks["ms3"][i]
    if combo == "ms1&ms2": return ms_masks["ms1"][i] & ms_masks["ms2"][i]
    if combo == "ms1&ms3": return ms_masks["ms1"][i] & ms_masks["ms3"][i]
    if combo == "ms2&ms3": return ms_masks["ms2"][i] & ms_masks["ms3"][i]
    if combo == "ms1&ms2&ms3": return ms_masks["ms1"][i] & ms_masks["ms2"][i] & ms_masks["ms3"][i]
    if combo == "vote>=1":
        v = ms_masks["ms1"][i].astype(int) + ms_masks["ms2"][i].astype(int) + ms_masks["ms3"][i].astype(int)
        return v >= 1
    if combo == "vote>=2":
        v = ms_masks["ms1"][i].astype(int) + ms_masks["ms2"][i].astype(int) + ms_masks["ms3"][i].astype(int)
        return v >= 2
    if combo == "vote>=3":
        v = ms_masks["ms1"][i].astype(int) + ms_masks["ms2"][i].astype(int) + ms_masks["ms3"][i].astype(int)
        return v >= 3
    raise ValueError(combo)


print("\n--- 5-way per-case search ---")
best_per_case = {}
for c in cases_uni:
    idx = case_idx[c]
    gt_c = [gms[i] for i in idx]
    best = ("base", 0.0, None)
    for w_xl in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for thr in np.arange(0.30, 0.81, 0.04):
            comb = [(1 - w_xl) * r1_probs[i].astype(np.float32) + w_xl * xl_probs[i].astype(np.float32) for i in idx]
            base_pms = [c_ >= thr for c_ in comb]
            f = f1(gt_c, base_pms)
            if f > best[1]: best = ("base", f, {"w_xl": w_xl, "thr": thr, "ms": None, "soft": None})
            for combo in ms_combo_options:
                for soft in np.arange(0.05, thr + 0.01, 0.10):
                    pms = []
                    for k, i in enumerate(idx):
                        rm = base_pms[k]
                        soft_mask = comb[k] >= soft
                        pms.append(rm | (get_extra_mask(combo, i) & soft_mask))
                    f = f1(gt_c, pms)
                    if f > best[1]: best = (combo, f, {"w_xl": w_xl, "thr": thr, "ms": combo, "soft": soft})
    best_per_case[c] = best
    print(f"  case {c}: F1={best[1]:.4f}  {best[0]}  w_xl={best[2]['w_xl']:.1f} thr={best[2]['thr']:.2f} soft={best[2].get('soft','-')}")

overall = [None] * n
for c in cases_uni:
    cfg = best_per_case[c][2]
    for i in case_idx[c]:
        comb = (1 - cfg["w_xl"]) * r1_probs[i].astype(np.float32) + cfg["w_xl"] * xl_probs[i].astype(np.float32)
        rm = comb >= cfg["thr"]
        if cfg["ms"] is None:
            overall[i] = rm
        else:
            soft_mask = comb >= cfg["soft"]
            extra = get_extra_mask(cfg["ms"], i) & soft_mask
            overall[i] = rm | extra

f_overall = f1(gms, overall)
print(f"\n[5-WAY PER-CASE OVERALL] F1 = {f_overall:.4f}")
print(f"\nbaselines:")
print(f"  4-way per-case (greedy + phase2):    0.8777")
print(f"  5-way per-case (+ beam):              {f_overall:.4f}")
