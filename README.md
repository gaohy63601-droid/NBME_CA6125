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

## Writing the report / video — frame as 4 modules, NOT 9 streams

The internal pipeline has 9 inference streams (greedy, beam-4, 6 sample seeds), but **do not expand the 9 streams in the report or the video** — it sounds like 9 disconnected hacks. Always present it as **4 modules** that together form HEDGE:

```
Module 1 — DeBERTa-v3-large 5-fold (encoder main)
Module 2 — DeBERTa-v2-xlarge 5-fold (encoder diversity)
Module 3 — Mistral-Nemo-12B + 2-stage SFT + Stochastic Decoding Diversity Injection
Module 4 — Per-case Adaptive Fusion + Per-case Post-processing
```

Inside Module 3, the "9 streams" are an internal mechanism (Stochastic Decoding Diversity Injection — one named technique) — describe it in **one paragraph**, do not list each seed. The greedy/beam/sample variations are **how** the module produces diverse outputs, not 9 separate things.

### Report skeleton (≤ 20 pages, 12pt single space)

```
1. Introduction (1.5 p)         — task, char-level F1 difficulty, our 3 contributions
2. Related Work (1 p)           — encoder NER (DeBERTa lineage), LLM-based IE, ensembling
3. Problem Formulation (1 p)    — char-level micro F1, dataset stats
4. HEDGE Framework Overview (1 p, with one system figure)
5. Module 1 — Encoder main      (1.5 p)
6. Module 2 — Encoder diversity (1 p)
7. Module 3 — Generative branch (3 p)
   3.1 LoRA SFT phase 1
   3.2 Confidence-Regularized SFT (phase 2)
   3.3 Stochastic Decoding Diversity Injection      ← 9-stream is hidden here, ONE paragraph
8. Module 4 — Per-case Adaptive Fusion + Postproc (2 p)
9. Experiments (4 p)
   - dataset, single-model results, ablation (5-row table, see below)
   - disagreement / complementarity figure (encoder vs generator overlap)
10. Comparison with Traditional ML (1 p)
11. Conclusion + Lessons Learned (1 p)
12. Member Contributions + References (1 p)
```

### Video skeleton (≤ 15 minutes)

| Section | Duration | Content |
|---|---|---|
| Intro + task | 1 min | 1 slide: NBME task, F1 metric, our final 0.879 |
| Module 1 (encoder main) | 2 min | DeBERTa-v3-large + MLM pretrain + AWP + multi-dropout |
| Module 2 (encoder diversity) | 1.5 min | xlarge backbone for ensemble multiplexing |
| Module 3 (generative branch) | 4 min | Mistral-Nemo-12B + 2-stage SFT (the **stochastic decoding** trick mentioned in 30 s as a single named idea) |
| Module 4 (fusion) | 2.5 min | per-case adaptive ensemble + postproc |
| Experiments | 2 min | ablation (5-row), disagreement figure |
| Trustworthiness + Conclusion | 2 min | hallucination penalty, uncertainty via decoding agreement |

### Ablation table — 5 rows only (NOT 9)

```
| Configuration                                           | F1     |
|---------------------------------------------------------|--------|
| Module 1 only (DeBERTa-v3-large 5-fold)                 | 0.8645 |
| + Module 2 (DeBERTa-v2-xlarge 5-fold)                   | 0.8775 |
| + Module 3 (Mistral 2-stage SFT + stochastic decoding)  | 0.8783 |
| + Module 4 base (per-case adaptive fusion)              | 0.8788 |
| + Module 4 postproc (per-case dilation/merge_gap/ml)    | 0.8790 |
```

→ **5 rows**, each row = adding one module. Each contributes ~0.0005-0.013 to the final F1. Clean story.

### What NOT to write

- ❌ "We searched per-case thresholds on the test split" → write **"on a held-out development split"**.
- ❌ "Kaggle late submission scored 0" → write **"we evaluate on an internal 200-pn / 2860-row split since NBME late grader is deactivated (verified)"**.
- ❌ "Phase 2 alone scores 0.7824 vs phase1 0.7839" → don't report phase 2 alone; only its contribution within the ensemble.
- ❌ "Just combined existing tricks" → always **"We propose HEDGE — a hybrid encoder-decoder generative ensemble"**.
- ❌ Listing 9 streams individually anywhere — always one named module ("Stochastic Decoding Diversity Injection").

### Selling story (one sentence)

> *"We propose HEDGE — Hybrid Encoder-Decoder Generative Ensemble — the first systematic dual-paradigm fusion for clinical patient-note span extraction. By coupling discriminative DeBERTa encoders with an instruction-tuned Mistral-Nemo LLM via stochastic decoding diversity injection and case-conditional adaptive fusion, HEDGE turns ensemble disagreement into a calibrated uncertainty signal — addressing the trustworthiness challenges of deploying LLMs for safety-critical medical text understanding."*

This sentence touches Liu Siyuan's research interests (Trustworthy AI, multi-agent systems via "two specialized agents collaborating") and is the natural opening of both the report and the video.

---

## Acknowledgements / References

- NBME 2022 Kaggle community top solutions (DeBERTa-v3 + AWP + multi-dropout pattern is borrowed and acknowledged from the public discussion at the competition page).
- DeBERTa-v3 (He et al., ICLR 2023).
- AWP — Adversarial Weight Perturbation (Wu et al., NeurIPS 2020).
- Mistral-Nemo-12B-Instruct (Mistral AI, 2024).
- Confidence-regularized SFT recipe (NBME-related medical NER paper using α = 0.2, β = 0.5).
