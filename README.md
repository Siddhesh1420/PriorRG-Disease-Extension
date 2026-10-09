# PriorRG Extension — 4-Disease Focused Evaluation

This repo contains the code used to extend PriorRG (chest X-ray report generation)
with a prior-report cross-attention fusion module, and the evaluation scripts used
to assess whether focusing on 4 under-performing diseases (Pneumonia, Pneumothorax,
Fracture, Pleural Other) improves results.

## Main scripts
- `main_mimic.py` — baseline report generation (inference only, uses a fixed pretrained checkpoint).
- `main_single_sample_mimic.py` — single-sample variant of the baseline.
- `main_mimic_priorreport.py` — Option B: baseline + a dedicated cross-attention fusion
  module that injects the prior study's report text, trained with `pt_lr=5e-7`.

## Analysis pipeline (`analysis/`)
- `precompute_train_labels.py` — precomputes CheXbert labels for the full training set.
- `build_disease_subset.py` — filters the dataset to samples containing at least one of
  the 4 target diseases (train: 41,198 / val: 354 / test: 951, out of the full
  240,197 / 2,113 / 3,852 split).
- `all_labels_confusion.py` — computes per-condition CheXbert precision/recall/F1 from a
  generated-reports CSV.
- `score_diseases.py` — runs the above across multiple saved experiment runs.
- `score_4disease_run.py` — scores the specialized (disease-filtered) run.
- `score_baseline_filtered.py` — runs/scores baseline on the SAME 951-sample filtered
  test set, for a fair, like-for-like comparison against the specialized model.

## Results (`analysis/results/`)
- `all_labels_confusion.csv` / `disease_scores_baseline.csv` — baseline, full 3,852-sample test set.
- `disease_scores_optionB_pilot_lowLR.csv` — early pilot, random 3,000-sample subset.
- `disease_scores_4disease_specialized.csv` — final model, trained+tested on the disease-filtered subset (951-sample test set).
- `disease_scores_baseline_on_filtered_testset.csv` — baseline re-evaluated on the SAME 951-sample filtered test set as the specialized model (fair comparison, see below).

## Key finding: fair, same-test-set comparison

Baseline and the specialized model were both evaluated on the identical 951-sample
filtered test set to get a like-for-like comparison:

| Disease | Baseline F1 | Specialized F1 | Baseline Acc | Specialized Acc |
|---|---|---|---|---|
| Pneumonia | 0.3311 | 0.3855 | 0.5878 | 0.6082 |
| Pleural Other | 0.0918 | 0.0995 | 0.8128 | 0.8096 |
| Fracture | 0.0711 | 0.0711 | 0.8076 | 0.8075 |
| Pneumothorax | 0.4429 | 0.3623 | 0.9180 | 0.9076 |

**Pneumonia shows a genuine improvement** in both F1 and accuracy. Pleural Other and
Fracture are essentially unchanged. **Pneumothorax regresses** under the specialized
model relative to baseline on this fair test. An earlier comparison (baseline on its
full 3,852-sample test set vs. specialized on the 951-sample filtered set) showed
improvement across all 4 diseases, but that comparison used two different test sets
and is not a valid like-for-like result — it is kept in this repo for reference
(`disease_scores_baseline.csv`) but the table above is the methodologically sound one.

## Requirements
See `requirements.txt`. Note: the METEOR metric requires a Java runtime on PATH
(tested with OpenJDK 17) — see the note at the top of `requirements.txt`.
