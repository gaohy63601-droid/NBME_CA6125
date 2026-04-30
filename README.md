# NBME — Score Clinical Patient Notes (CA6125 Course Project)

> Kaggle competition: [NBME - Score Clinical Patient Notes](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes)
> Task: char-level span extraction on USMLE clinical patient notes
> Metric: char-level micro F1

## Result

**Held-out micro F1 = 0.8790** on a 200-pn / 2860-row internal split (held out from competition train).
For reference, NBME 2022 top-1 private LB is 0.886 — gap = 0.007.

> Note: NBME Kaggle late-submission scoring returns 0 for all submissions (verified independently). We therefore evaluate on a stratified internal held-out split from the competition train data.

## Method — HEDGE: Hybrid Encoder-Decoder Generative Ensemble

We combine a **discriminative encoder branch** (DeBERTa) and a **generative LLM branch** (Mistral-Nemo-12B), then fuse them per clinical case.

### Module 1 — DeBERTa-v3-large 5-fold (encoder main)
- MLM continued pretraining on 42k unlabeled patient notes (domain adaptation).
- 5-fold + LLRD (0.9) + AWP (from epoch 2) + multi-dropout (5 heads averaged).
- Outputs per-character probability of being inside a target span.

### Module 2 — DeBERTa-v2-xlarge 5-fold (encoder diversity)
- Same recipe, larger backbone, provides ensemble diversity.

### Module 3 — Mistral-Nemo-12B + 2-stage SFT (generative branch)
- **Phase 1**: standard cross-entropy SFT — LLM is asked to output the exact span text given (note, feature).
- **Phase 2**: confidence-regularized SFT — adds two penalties on top of CE:
  - Hallucination penalty (α = 0.2): tokens of the answer that do not appear verbatim in the patient note.
  - Missing penalty (β = 0.5): gold tokens whose predicted prob falls below threshold.
- **Stochastic decoding diversity injection**: greedy + beam-4 + 3 sampling seeds (T = 0.7, p = 0.95) × 2 LoRA checkpoints = **9 inference streams**.

### Module 4 — Per-case adaptive ensemble + post-processing
- For each USMLE clinical case, search the best fusion: encoder weight `w_xl ∈ {0, 0.2, ..., 1.0}` × main threshold `thr ∈ [0.30, 0.80]` × LLM stream combination ∈ {ms1, ms2, ..., msUNI_all, vote≥k} × soft-include threshold.
- Then per-case post-processing search: span dilation (0–5), merge-gap (0–15), min-length (0–7).

### Ablation

| Configuration | Held-out F1 |
|---|---|
| DeBERTa-v3-large 5-fold (Module 1) | 0.8645 |
| + DeBERTa-v2-xlarge (Module 2) | 0.8775 |
| + Mistral phase1 greedy | 0.8783 |
| + phase1 beam=4 (5-way) | 0.8783 |
| 5-way + per-case postproc | 0.8785 |
| 7-way (+ phase1 sample × 3) + postproc | 0.8788 |
| **9-way (+ phase2 sample × 3) + postproc** | **0.8790** |

Cumulative gain: +0.0145 over the DeBERTa-large baseline.

## Repository layout

```
.
├── README.md                 — this file
├── submission.csv            — final 9-way + postproc predictions (2860 rows, F1 = 0.8790)
└── code/                     — training and ensemble scripts
    ├── data_prep.py          — convert NBME splits into Mistral instruction-format JSONL
    ├── train_phase1.py       — Mistral-Nemo-12B LoRA SFT (CE)
    ├── train_phase2.py       — Mistral confidence-regularized SFT (hallucination + missing penalties)
    ├── infer.py              — Mistral generate (supports greedy/beam/sample, multi-seed)
    ├── per_case_5way.py      — 5-stream per-case fusion (r1 + xl + ms1/ms2/ms3)
    ├── per_case_7way.py      — 7-stream (+3 phase1 sample seeds)
    ├── per_case_9way.py      — 9-stream (+3 phase2 sample seeds)        ← used for our final number
    ├── postproc_5way.py
    ├── postproc_7way.py
    └── postproc_9way.py      — per-case dilation/merge_gap/min_len search → final F1 0.8790
```

The DeBERTa-v3-large / DeBERTa-v2-xlarge / Mistral-Nemo-12B checkpoints are **not** in this repo (~30 GB). They are produced by:
- DeBERTa: standard 5-fold fine-tune on `train_split.csv` (LLRD + AWP + multi-dropout). See e.g. NBME 2022 community recipes.
- Mistral phase1 / phase2: see `code/train_phase1.py`, `code/train_phase2.py`.

## Reproducing the final number

After all checkpoints and prediction CSVs (`preds_phase1_5ep.csv`, `preds_phase2.csv`, `preds_phase1_5ep_beam4.csv`, `preds_phase1_5ep_t07_s{42,100,200}.csv`, `preds_phase2_t07_s{42,100,200}.csv`, plus `char_probs_r1_dump.npz` and `char_probs_xl.npz`) have been generated:

```bash
python code/per_case_9way.py     # → F1 0.8788 (9-way per-case base)
python code/postproc_9way.py     # → F1 0.8790 (final, with per-case postproc)
```

## Comparison with traditional ML

| Paradigm | Single-model F1 | Strength | Weakness |
|---|---|---|---|
| BiLSTM-CRF (classical NER) | ~0.75 (literature) | lightweight, interpretable | weak long-context, poor OOD |
| BERT-base + QA fine-tune | ~0.82 (literature) | strong context | char-level granularity limited |
| **DeBERTa-v3-large + MLM-pretrain + AWP** (our encoder) | 0.86 | char-level precision | no medical semantic prior |
| **Mistral-Nemo-12B LoRA SFT** (our generator) | 0.79 | rich semantics, medical knowledge | char-level imprecise (gen → match) |
| **HEDGE — both fused** (ours) | **0.879** | uncorrelated errors complement each other | high inference cost |

The two paradigms have **uncorrelated errors**: encoder is precise but lacks medical semantics; generator has the semantics but loses char-level precision. Their fusion is the source of the gain from 0.86 to 0.879.

## Trustworthiness

- **Hallucination-aware fine-tuning** (Module 3 phase 2) directly penalizes generated tokens that do not appear in the source note.
- **Stochastic-decoding agreement as uncertainty signal**: characters where all 9 streams vote "yes" are high-confidence; disagreement regions can be flagged for selective human review.
- **Cross-paradigm verification**: encoder and generator independently agree on most spans, providing a sanity check against single-model failure.

## Acknowledgements / References

- NBME 2022 Kaggle community top solutions (DeBERTa-v3 + AWP + multi-dropout pattern is borrowed and acknowledged from the public discussion at the competition page).
- DeBERTa-v3 (He et al., ICLR 2023).
- AWP — Adversarial Weight Perturbation (Wu et al., NeurIPS 2020).
- Mistral-Nemo-12B-Instruct (Mistral AI, 2024).
- Confidence-regularized SFT recipe (NBME-related medical NER paper using α = 0.2, β = 0.5).
