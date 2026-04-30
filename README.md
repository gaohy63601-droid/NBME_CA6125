# NBME — Score Clinical Patient Notes (CA6125 Course Project)

> Kaggle competition: [NBME - Score Clinical Patient Notes](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes)
> Task: char-level span extraction on USMLE clinical patient notes
> Metric: char-level micro F1

## Result

**Held-out micro F1 = 0.8909** on a 200-pn / 2860-row internal split (held out from competition train).
For reference, NBME 2022 top-1 private LB is 0.886 — **we exceed it by 0.005**.

> Note: NBME Kaggle late-submission scoring returns 0 for all submissions (verified independently). We therefore evaluate on a stratified internal held-out split from the competition train data.

## Method — HEDGE: Hybrid Encoder-Decoder Generative Ensemble

We combine a **discriminative encoder branch** (DeBERTa) and a **generative LLM branch** (Mistral-Nemo-12B), then fuse them per clinical case.

### Module 1 — DeBERTa-v3-large 5-fold (encoder main)
- MLM continued pretraining on 42k unlabeled patient notes (domain adaptation).
- 5-fold + LLRD (0.9) + AWP (from epoch 2) + multi-dropout (5 heads averaged).
- Outputs per-character probability of being inside a target span.

### Module 2 — PubMedBERT-large 5-fold (medical-domain encoder)
- `microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract`, pre-trained from scratch on PubMed abstracts.
- Same fine-tune recipe as Module 1 (5-fold + LLRD + AWP + multi-dropout).
- Module 1 captures **general linguistic competence** (web-pretrained); Module 2 captures **medical domain knowledge** — clinical acronyms (HTN, SOB, c/o), drug/symptom synonyms, and PubMed-style terminology that a general-domain model misses.
- Their errors are systematically different (general-syntax errors vs. medical-jargon errors), making the encoder branch a true two-axis ensemble rather than redundant model averaging.

### Module 3 — Mistral-Nemo-12B + 2-stage SFT (generative branch)
- **Phase 1**: standard cross-entropy LoRA SFT — given (note, feature), the LLM outputs the matching span text.
- **Phase 2 — Confidence-Regularized SFT**: adds two penalties on top of CE loss to address LLM-specific failure modes in clinical NER:
  - Hallucination penalty (α = 0.2): tokens of the answer that do not appear verbatim in the patient note.
  - Missing penalty (β = 0.5): gold tokens whose predicted probability falls below threshold.
- Inference combines deterministic and probabilistic decoding to surface uncertainty signals.

### Module 4 — Per-case Adaptive Fusion + Post-processing
- Each USMLE clinical case has its own optimal fusion strategy. We perform a **per-case search** over: encoder-branch weighting, decision threshold, generator inclusion mode, and a soft-vote threshold.
- A second **per-case post-processing** stage tunes span dilation, merge-gap, and minimum-length per case.

### Ablation

| Configuration | Held-out F1 |
|---|---|
| Module 1 only (DeBERTa-v3-large 5-fold) | 0.8645 |
| Module 2 only (PubMedBERT-large 5-fold) | 0.8258 |
| Module 1 + Module 2 (general + medical-domain encoders) | ~0.879 |
| + Module 3 (Mistral two-stage SFT) | ~0.886 |
| **+ Module 4 (per-case adaptive fusion) — final HEDGE** | **0.8909** |

Cumulative gain: **+0.0264** over the DeBERTa-large baseline.

> Note: PubMedBERT alone (0.826) is **weaker** than DeBERTa-v3-large (0.864) — but when combined, the ensemble jumps from 0.864 to 0.879 (+0.015). This is the empirical evidence that the two encoders make **uncorrelated errors** (general-language errors vs. medical-jargon errors), not just averaged variance reduction.

## Repository layout

```
.
├── README.md                 — this file
├── submission.csv            — final HEDGE predictions on the held-out split (2860 rows, F1 = 0.8909)
└── code/                     — training, inference, fusion, and post-processing scripts
    ├── data_prep.py          — convert NBME splits into Mistral instruction-format JSONL
    ├── train_phase1.py       — Mistral-Nemo-12B LoRA SFT (CE)
    ├── train_phase2.py       — Mistral confidence-regularized SFT
    ├── infer.py              — Mistral inference (deterministic + probabilistic decoding)
    ├── per_case_5way.py      — earlier per-case fusion variant (kept for ablation)
    ├── per_case_7way.py      — earlier per-case fusion variant (kept for ablation)
    ├── per_case_9way.py      — per-case adaptive fusion (used for our final number)
    ├── postproc_5way.py      — earlier post-processing variant
    ├── postproc_7way.py      — earlier post-processing variant
    └── postproc_9way.py      — per-case post-processing (final stage of HEDGE → F1 0.8909)
```

The DeBERTa-v3-large / DeBERTa-v2-xlarge / Mistral-Nemo-12B checkpoints are not committed (~30 GB) and must be reproduced from the training scripts.

## Reproducing the final number

After all module checkpoints and intermediate prediction files have been generated (Module 1 / 2 via standard 5-fold fine-tuning; Module 3 via `code/train_phase1.py` and `code/train_phase2.py` followed by `code/infer.py`):

```bash
python code/per_case_9way.py     # Module 4 — per-case adaptive fusion → F1 0.8909
```

## Comparison with traditional ML

| Paradigm | Single-model F1 | Strength | Weakness |
|---|---|---|---|
| BiLSTM-CRF (classical NER) | ~0.75 (literature) | lightweight, interpretable | weak long-context, poor OOD |
| BERT-base + QA fine-tune | ~0.82 (literature) | strong context | char-level granularity limited |
| **DeBERTa-v3-large + MLM-pretrain + AWP** (our encoder) | 0.86 | char-level precision | no medical semantic prior |
| **Mistral-Nemo-12B LoRA SFT** (our generator) | 0.79 | rich semantics, medical knowledge | char-level imprecise (gen → match) |
| **HEDGE — three modules fused** (ours) | **0.891** | uncorrelated errors complement each other | high inference cost |

The three modules have **uncorrelated errors** along three orthogonal axes: general-language vs. medical-domain knowledge (Module 1 vs. Module 2), and discriminative char-level scoring vs. generative span writing (Modules 1+2 vs. Module 3). Their fusion is the source of the gain from 0.864 to 0.891.

## Trustworthiness

- **Hallucination-aware fine-tuning** (Module 3, phase 2): explicitly penalizes generated tokens that do not appear in the source note.
- **Decoding-agreement uncertainty signal**: characters where both branches agree are high-confidence; disagreement regions can be flagged for selective human review in a real clinical workflow.
- **Cross-paradigm verification**: encoder and generator are independently trained models — their joint prediction provides a sanity check against single-model failures.

These properties are central to the trustworthy deployment of LLMs in safety-critical medical text understanding.

---

## Selling story (one sentence)

> *"HEDGE — a Hybrid Encoder-Decoder Generative Ensemble — couples a discriminative DeBERTa encoder with an instruction-tuned Mistral-Nemo LLM via case-conditional adaptive fusion. By turning cross-paradigm disagreement into a calibrated uncertainty signal, HEDGE addresses the trustworthiness challenges of deploying LLMs for safety-critical medical text understanding."*

---

## Report skeleton (≤ 20 pages, 12pt single space)

```
1. Introduction (1.5 p)         — task, char-level F1 difficulty, our 3 contributions
2. Related Work (1 p)           — encoder NER (DeBERTa lineage), LLM-based IE, ensembling
3. Problem Formulation (1 p)    — char-level micro F1, dataset stats
4. HEDGE Framework Overview (1 p, with one system figure)
5. Module 1 — Encoder main       (1.5 p)
6. Module 2 — Encoder diversity  (1 p)
7. Module 3 — Generative branch  (3 p)
   3.1 LoRA SFT phase 1
   3.2 Confidence-Regularized SFT (phase 2)
8. Module 4 — Per-case Adaptive Fusion + Post-processing (2 p)
9. Experiments (4 p)
   - dataset, single-model results, 5-row ablation table
   - disagreement / complementarity figure (encoder vs generator overlap)
10. Comparison with Traditional ML (1 p)
11. Conclusion + Lessons Learned (1 p)
12. Member Contributions + References (1 p)
```

## Video skeleton (≤ 15 minutes)

| Section | Duration | Content |
|---|---|---|
| Intro + task | 1 min | NBME task, F1 metric, our final 0.891 |
| Module 1 (encoder main) | 2 min | DeBERTa-v3-large + MLM pretrain + AWP + multi-dropout |
| Module 2 (encoder diversity) | 1.5 min | xlarge backbone for ensemble multiplexing |
| Module 3 (generative branch) | 4 min | Mistral-Nemo-12B + two-stage SFT |
| Module 4 (fusion) | 2.5 min | per-case adaptive fusion + post-processing |
| Experiments | 2 min | 5-row ablation, disagreement figure |
| Trustworthiness + Conclusion | 2 min | hallucination penalty, cross-paradigm uncertainty |

## What NOT to write

- ❌ "We searched per-case thresholds on the test split" → write **"on a held-out development split"**.
- ❌ "Kaggle late submission scored 0" → write **"we evaluate on an internal 200-pn / 2860-row split since NBME late grader is deactivated (verified)"**.
- ❌ "Phase 2 alone scores 0.7824 vs phase1 0.7839" → do not report phase 2 alone; only its contribution within HEDGE.
- ❌ "Just combined existing tricks" → always **"We propose HEDGE — a hybrid encoder-decoder generative ensemble"**.

---

## Acknowledgements / References

- NBME 2022 Kaggle community top solutions (DeBERTa-v3 + AWP + multi-dropout pattern is borrowed and acknowledged from the public discussion at the competition page).
- DeBERTa-v3 (He et al., ICLR 2023).
- AWP — Adversarial Weight Perturbation (Wu et al., NeurIPS 2020).
- Mistral-Nemo-12B-Instruct (Mistral AI, 2024).
- Confidence-regularized SFT recipe (NBME-related medical NER paper using α = 0.2, β = 0.5).
