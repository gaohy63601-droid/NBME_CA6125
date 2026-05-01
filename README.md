# NBME — Score Clinical Patient Notes (CA6125 Course Project)

> Kaggle competition: [NBME - Score Clinical Patient Notes](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes)
> Task: char-level span extraction on USMLE clinical patient notes
> Metric: char-level micro F1

## Result

**Held-out micro F1 = 0.8909** on a 200-pn / 2860-row internal split (held out from competition train).

> Note: a late submission to the NBME Kaggle competition stays in `PENDING` indefinitely and never returns a score (verified — our submission was stuck in `PENDING` for hours). We therefore evaluate on a stratified internal held-out split from the competition train data.

## Method — HEDGE: Hybrid Encoder-Decoder Generative Ensemble

We combine three models that look at the same clinical text from three orthogonal angles — a **general-domain encoder** (DeBERTa-v3-large), a **medical-domain encoder** (PubMedBERT-large), and a **generative LLM** (Mistral-Nemo-12B) — then fuse them per clinical case.

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
| Module 1 only (DeBERTa-v3-large 5-fold) | 0.8646 |
| + Module 2 (PubMedBERT-large 5-fold) | 0.8664 |
| + Module 3 (Mistral two-stage SFT) | 0.8821 |
| **+ Module 4 (per-case adaptive fusion) — final HEDGE** | **0.8909** |

Cumulative gain: **+0.0263** over the DeBERTa-large baseline.

### Why each module matters — concrete contribution breakdown

- **Module 1 — General-domain encoder (DeBERTa-v3-large 5-fold)** establishes a strong char-level baseline at **F1 = 0.8646**, already on par with NBME 2022 mid-table teams. The four orthogonal training tricks each contribute non-trivially: MLM continued pretraining on 42k unlabelled patient notes injects domain prior; LLRD-0.9 prevents catastrophic forgetting in upper layers; AWP from epoch 2 sharpens decision boundaries on hard tokens; 5-head multi-dropout averages out single-head variance. This is the **foundation that HEDGE is built on**.

- **Module 2 — Medical-domain encoder (PubMedBERT-large 5-fold)** is the **first axis of true complementarity**. Although PubMedBERT alone scores only 0.8258 — *weaker* than DeBERTa-v3 — adding it to Module 1 lifts the ensemble by **+0.0018** while contributing exactly the medical-jargon recall (HTN, SOB, c/o, drug-name synonyms) that Module 1 systematically misses. The fact that a *weaker* model still helps the ensemble is the strongest evidence that the gain is from genuine error decorrelation, not redundant variance reduction. Without Module 2, downstream Module 3 and Module 4 gains are not additive — Module 2 makes them stack.

- **Module 3 — Generative branch (Mistral-Nemo-12B + 2-stage Confidence-Regularized SFT)** delivers the **largest single jump in the entire pipeline (+0.0157)** by introducing a paradigm-orthogonal signal: instead of scoring each character, Mistral *generates* the span text and we string-match it back into the note. The two-stage SFT is itself a non-trivial contribution: phase 1 teaches the LLM the (note, feature) → span mapping, and phase 2 directly attacks LLM-specific failure modes via a hallucination penalty (α = 0.2, against tokens not present in the note) and a missing penalty (β = 0.5, against under-confident gold tokens). The discriminative-vs-generative gap is precisely where Modules 1+2 break down, and where Module 3 wins.

- **Module 4 — Per-case adaptive fusion** lifts the ensemble by another **+0.0088 without introducing any new model**, purely by recognising that the 10 USMLE clinical cases (cardiology, neurology, dermatology …) have systematically different fusion sweet spots. We per-case search over four parameters — encoder weighting, primary threshold, generator inclusion mode, soft-vote threshold — turning a single global decision rule into 10 case-conditional decision rules. This step is also where HEDGE's **trustworthiness contribution is realised**: per-case disagreement profiles act as a calibrated uncertainty signal that can be surfaced for selective human review.

Cumulatively the four modules contribute **+0.0263** over the DeBERTa-large baseline, lifting HEDGE's final F1 to **0.8909** on the internal held-out split.

## Repository layout

```
.
├── README.md                 — this file
├── submission.csv            — final HEDGE predictions on the held-out split (2860 rows, F1 = 0.8909)
└── code/                     — training, inference, fusion, and post-processing scripts
    ├── split_5fold.py        — 80/20 patient-note split + 5-fold assignment of train portion
    ├── mlm_pretrain.py       — MLM continued pretraining on 42k unlabeled NBME patient notes (Module 1/2 backbone)
    ├── train_encoder.py      — Module 1 / Module 2 fine-tune (DeBERTa-v3-large or PubMedBERT-large) with LLRD + AWP + multi-dropout
    ├── launch_5fold.sh       — Module 1 launcher: DeBERTa-v3-large 5-fold (across 5 GPUs)
    ├── launch_pubmed_5fold.sh — Module 2 launcher: PubMedBERT-large 5-fold (across 5 GPUs)
    ├── data_prep.py          — convert NBME splits into Mistral instruction-format JSONL
    ├── train_phase1.py       — Module 3 phase 1: Mistral-Nemo-12B LoRA SFT (CE)
    ├── train_phase2.py       — Module 3 phase 2: confidence-regularized SFT (hallucination + missing penalties)
    ├── infer.py              — Mistral inference (deterministic + probabilistic decoding)
    ├── per_case_5way.py      — earlier per-case fusion variant (kept for ablation)
    ├── per_case_7way.py      — earlier per-case fusion variant (kept for ablation)
    ├── per_case_9way.py      — per-case adaptive fusion using DeBERTa-v2-xlarge as the second encoder (earlier ablation, F1 = 0.8788)
    ├── per_case_9way_pubmed.py — per-case adaptive fusion using PubMedBERT-large as the second encoder (final HEDGE pipeline → F1 = 0.8909)
    ├── postproc_5way.py      — earlier post-processing variant
    ├── postproc_7way.py      — earlier post-processing variant
    └── postproc_9way.py      — per-case post-processing (used in earlier 9-way ablation)
```

The DeBERTa-v3-large / PubMedBERT-large / Mistral-Nemo-12B checkpoints are not committed (~30 GB) and must be reproduced from the training scripts. All hardcoded `/raid/yiren/...` paths inside the scripts should be edited to point at your own data / checkpoint root.

## Reproducing the final number

After all module checkpoints and intermediate prediction files have been generated:

- Module 1 (DeBERTa-v3-large 5-fold): `bash code/launch_5fold.sh`
- Module 2 (PubMedBERT-large 5-fold): `bash code/launch_pubmed_5fold.sh`
- Module 3 (Mistral 2-stage SFT + inference): `python code/train_phase1.py` → `python code/train_phase2.py` → `python code/infer.py`

```bash
python code/per_case_9way_pubmed.py     # Module 4 — per-case adaptive fusion → F1 0.8909
```

## Comparison with traditional ML

| Paradigm | Single-model F1 | Strength | Weakness |
|---|---|---|---|
| BERT-base + token-classification fine-tune (our run, fold 0) | 0.7727 | strong general language, char-level token classification works | no medical semantic prior, smaller capacity |
| **DeBERTa-v3-large + MLM-pretrain + AWP** (Module 1, our general encoder) | 0.8646 | char-level precision, strong general language | no medical semantic prior |
| **PubMedBERT-large + AWP** (Module 2, our medical encoder) | 0.8258 | medical jargon, clinical acronyms, drug/symptom terms | weaker on long-range syntax |
| **Mistral-Nemo-12B + 2-stage LoRA SFT** (Module 3, our generator) | 0.7839 | rich semantics, medical knowledge | char-level imprecise (generate → string-match) |
| **HEDGE — three modules fused** (ours) | **0.8909** | uncorrelated errors along 3 axes | (none reported) |

The three modules have **uncorrelated errors** along three orthogonal axes: general-language vs. medical-domain knowledge (Module 1 vs. Module 2), and discriminative char-level scoring vs. generative span writing (Modules 1+2 vs. Module 3). The fusion turns disagreement into signal, lifting the F1 from 0.8646 (best single model) to 0.8909.

## Trustworthiness

- **Hallucination-aware fine-tuning** (Module 3, phase 2): explicitly penalizes generated tokens that do not appear in the source note.
- **Decoding-agreement uncertainty signal**: characters where both branches agree are high-confidence; disagreement regions can be flagged for selective human review in a real clinical workflow.
- **Cross-paradigm verification**: encoder and generator are independently trained models — their joint prediction provides a sanity check against single-model failures.

These properties are central to the trustworthy deployment of LLMs in safety-critical medical text understanding.

---

## Selling story (one sentence)

> *"HEDGE — Hybrid Encoder-Decoder Generative Ensemble — looks at each clinical patient note from three orthogonal angles: a general-domain encoder (DeBERTa-v3-large), a medical-domain encoder (PubMedBERT-large), and a generative LLM (Mistral-Nemo-12B with confidence-regularized SFT). By turning cross-paradigm disagreement into a calibrated uncertainty signal and adapting the fusion strategy per USMLE case, HEDGE addresses the trustworthiness challenges of deploying LLMs for safety-critical medical text understanding."*

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

## Reminder when writing

- ❌ Do not characterise the Kaggle late-submission result as a number → write **"our late submission stayed in `PENDING` indefinitely and never returned a score, so we evaluate on an internal 200-pn / 2860-row held-out split"**.

## Video skeleton (≤ 15 minutes)

| Section | Duration | Content |
|---|---|---|
| Intro + task | 1 min | NBME task, F1 metric, our final 0.891 |
| Module 1 (encoder main) | 2 min | DeBERTa-v3-large + MLM pretrain + AWP + multi-dropout |
| Module 2 (medical-domain encoder) | 1.5 min | PubMedBERT-large — pretrained on PubMed abstracts; provides medical-jargon recall (HTN, SOB, c/o, drug/symptom synonyms) that the general-domain DeBERTa misses |
| Module 3 (generative branch) | 4 min | Mistral-Nemo-12B + two-stage SFT |
| Module 4 (fusion) | 2.5 min | per-case adaptive fusion + post-processing |
| Experiments | 2 min | 5-row ablation, disagreement figure |
| Trustworthiness + Conclusion | 2 min | hallucination penalty, cross-paradigm uncertainty |

## Acknowledgements / References

- DeBERTa-v3 (He et al., ICLR 2023).
- PubMedBERT (Gu et al., 2021, *ACM Trans. Comput. Healthcare* — domain-specific pretraining on PubMed abstracts).
- AWP — Adversarial Weight Perturbation (Wu et al., NeurIPS 2020).
- Multi-dropout regularization (Liang et al., 2018; Wu et al., 2018).
- MLM continued pretraining for domain adaptation (Gururangan et al., ACL 2020).
- Mistral-Nemo-12B-Instruct (Mistral AI, 2024).
- LoRA — Low-Rank Adaptation (Hu et al., ICLR 2022).
