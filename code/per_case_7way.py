"""7-way per-case ensemble:
   r1 + xl + ms1(greedy 5ep) + ms2(phase2) + ms3(beam4 5ep)
       + ms4(sample s42 T=0.7) + ms5(sample s100 T=0.7) + ms6(sample s200 T=0.7)

Search space stays tractable by collapsing the 3 sample streams into derived masks:
  ms_v2  = vote>=2 of {ms4, ms5, ms6}
  ms_uni = ms4 | ms5 | ms6
"""
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

ms_files = {
    "ms1": "preds_phase1_5ep.csv",
    "ms2": "preds_phase2.csv",
    "ms3": "preds_phase1_5ep_beam4.csv",
    "ms4": "preds_phase1_5ep_t07_s42.csv",
    "ms5": "preds_phase1_5ep_t07_s100.csv",
    "ms6": "preds_phase1_5ep_t07_s200.csv",
}
ms_masks = {}
for name, fn in ms_files.items():
    df = pd.read_csv(os.path.join(PRED_DIR_MS, fn))
    loc = dict(zip(df["id"].astype(str), df["pred_location"].fillna("").astype(str)))
    ms_masks[name] = [gt_mask(parse_pred(loc.get(str(ids[i]), "")), int(note_lens[i])) for i in range(n)]
gms = [gt_mask(parse_loc(str(l)), int(note_lens[i])) for i, l in enumerate(locations)]
print(f"loaded {n} rows + {len(ms_files)} ms predictions")

# Per-row standalone F1s for each ms (sanity check)
for name in ms_files:
    fi = f1(gms, ms_masks[name])
    print(f"  {name} alone F1 = {fi:.4f}")

# Build derived sample-aggregate masks
ms_v2 = []   # vote>=2 of {ms4,ms5,ms6}
ms_uni = []  # union of {ms4,ms5,ms6}
ms_int = []  # intersection of {ms4,ms5,ms6}
for i in range(n):
    a, b, c = ms_masks["ms4"][i], ms_masks["ms5"][i], ms_masks["ms6"][i]
    v = a.astype(int) + b.astype(int) + c.astype(int)
    ms_v2.append(v >= 2)
    ms_uni.append(a | b | c)
    ms_int.append(a & b & c)
ms_masks["msV2"] = ms_v2
ms_masks["msUNI"] = ms_uni
ms_masks["msINT"] = ms_int
for name in ("msV2", "msUNI", "msINT"):
    print(f"  {name} (derived) alone F1 = {f1(gms, ms_masks[name]):.4f}")

cases_uni = sorted(set(cases.tolist()))
case_idx = {c: np.where(cases == c)[0] for c in cases_uni}

# Combinations to try.  Same shape as 5-way's space, plus sample-aggregate variants.
ms_combo_options = [
    # singletons
    "ms1", "ms2", "ms3", "msV2", "msUNI",
    # 5-way unions
    "ms1|ms2", "ms1|ms3", "ms2|ms3", "ms1|ms2|ms3",
    # 5-way intersections
    "ms1&ms2", "ms1&ms3", "ms2&ms3", "ms1&ms2&ms3",
    # 5-way votes
    "vote>=1[1,2,3]", "vote>=2[1,2,3]", "vote>=3[1,2,3]",
    # sample-aggregate combined with greedy/beam
    "ms1|msV2", "ms3|msV2", "ms1|msUNI", "ms3|msUNI",
    "ms1&msV2", "ms3&msV2",
    # 3-way vote of {ms1, ms3, msV2}
    "vote>=1[1,3,V2]", "vote>=2[1,3,V2]",
    # 3-way vote of {ms1, ms2, msV2}
    "vote>=2[1,2,V2]",
]


def get_extra_mask(combo, i):
    if combo in ms_masks:
        return ms_masks[combo][i]
    if combo == "ms1|ms2": return ms_masks["ms1"][i] | ms_masks["ms2"][i]
    if combo == "ms1|ms3": return ms_masks["ms1"][i] | ms_masks["ms3"][i]
    if combo == "ms2|ms3": return ms_masks["ms2"][i] | ms_masks["ms3"][i]
    if combo == "ms1|ms2|ms3": return ms_masks["ms1"][i] | ms_masks["ms2"][i] | ms_masks["ms3"][i]
    if combo == "ms1&ms2": return ms_masks["ms1"][i] & ms_masks["ms2"][i]
    if combo == "ms1&ms3": return ms_masks["ms1"][i] & ms_masks["ms3"][i]
    if combo == "ms2&ms3": return ms_masks["ms2"][i] & ms_masks["ms3"][i]
    if combo == "ms1&ms2&ms3": return ms_masks["ms1"][i] & ms_masks["ms2"][i] & ms_masks["ms3"][i]
    if combo.startswith("vote>=") and "[1,2,3]" in combo:
        k = int(combo.split(">=")[1].split("[")[0])
        v = ms_masks["ms1"][i].astype(int) + ms_masks["ms2"][i].astype(int) + ms_masks["ms3"][i].astype(int)
        return v >= k
    if combo == "ms1|msV2": return ms_masks["ms1"][i] | ms_masks["msV2"][i]
    if combo == "ms3|msV2": return ms_masks["ms3"][i] | ms_masks["msV2"][i]
    if combo == "ms1|msUNI": return ms_masks["ms1"][i] | ms_masks["msUNI"][i]
    if combo == "ms3|msUNI": return ms_masks["ms3"][i] | ms_masks["msUNI"][i]
    if combo == "ms1&msV2": return ms_masks["ms1"][i] & ms_masks["msV2"][i]
    if combo == "ms3&msV2": return ms_masks["ms3"][i] & ms_masks["msV2"][i]
    if combo.startswith("vote>=") and "[1,3,V2]" in combo:
        k = int(combo.split(">=")[1].split("[")[0])
        v = ms_masks["ms1"][i].astype(int) + ms_masks["ms3"][i].astype(int) + ms_masks["msV2"][i].astype(int)
        return v >= k
    if combo.startswith("vote>=") and "[1,2,V2]" in combo:
        k = int(combo.split(">=")[1].split("[")[0])
        v = ms_masks["ms1"][i].astype(int) + ms_masks["ms2"][i].astype(int) + ms_masks["msV2"][i].astype(int)
        return v >= k
    raise ValueError(combo)


print("\n--- 7-way per-case search ---")
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
                    for k_, i in enumerate(idx):
                        rm = base_pms[k_]
                        soft_mask = comb[k_] >= soft
                        pms.append(rm | (get_extra_mask(combo, i) & soft_mask))
                    f = f1(gt_c, pms)
                    if f > best[1]: best = (combo, f, {"w_xl": w_xl, "thr": thr, "ms": combo, "soft": soft})
    best_per_case[c] = best
    cfg = best[2]
    print(f"  case {c}: F1={best[1]:.4f}  ms={best[0]}  w_xl={cfg['w_xl']:.1f} thr={cfg['thr']:.2f} soft={cfg.get('soft','-')}")

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

# Save per-case overall mask + best config for postproc
np.savez(os.path.join(PRED_DIR_MS, "per_case_7way_dump.npz"),
         overall=np.array(overall, dtype=object),
         note_lens=note_lens, cases=cases, ids=ids,
         locations=locations, allow_pickle=True)

f_overall = f1(gms, overall)
print(f"\n[7-WAY PER-CASE OVERALL] F1 = {f_overall:.4f}")
print(f"\nbaselines:")
print(f"  4-way per-case (greedy + phase2 + xl):    0.8775")
print(f"  5-way per-case (+ beam):                   0.8783")
print(f"  5-way + per-case postproc:                 0.8785")
print(f"  7-way (this run, +3 sample seeds):         {f_overall:.4f}")
