import sys
sys.path.insert(0, '/DATA/intern3/PriorRG')
import pandas as pd
from tools.metrics.f1chexbert import F1CheXbert

CSV_PATH = '/DATA/intern3/PriorRG/PriorRG/mimic_cxr/report-generation-test/v1_2026_10_09_09/generated_reports/test_reports_epoch-1_09-10-2026_10-04-29.csv'
CHEXBERT_CKPT = '/DATA/intern3/PriorRG/checkpoints/chexbert.pth'
BERT_PATH = 'google-bert/bert-base-uncased'
TARGET_DISEASES = ['Pneumonia', 'Pneumothorax', 'Fracture', 'Pleural Other']

df = pd.read_csv(CSV_PATH)
print(f'loaded {len(df)} rows')
labeler = F1CheXbert(chexbert_checkpoint=CHEXBERT_CKPT, model_checkpoint=BERT_PATH, tokenizer_checkpoint=BERT_PATH)
names = labeler.target_names

ref_labels_all, hyp_labels_all = [], []
for i, row in df.iterrows():
    ref_labels_all.append(labeler.get_label(str(row['reference_report']).strip(), mode='rrg'))
    hyp_labels_all.append(labeler.get_label(str(row['generated_report']).strip(), mode='rrg'))
    if (i + 1) % 200 == 0: print(f'{i+1}/{len(df)} done')

results = []
for idx, name in enumerate(names):
    TP = FP = FN = TN = 0
    for ref_row, hyp_row in zip(ref_labels_all, hyp_labels_all):
        r, h = ref_row[idx], hyp_row[idx]
        if r == 1 and h == 1: TP += 1
        elif r == 0 and h == 1: FP += 1
        elif r == 1 and h == 0: FN += 1
        else: TN += 1
    gt_pos = TP + FN; pred_pos = TP + FP
    precision = TP / pred_pos if pred_pos else 0.0
    recall = TP / gt_pos if gt_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (TP + TN) / (TP + FP + FN + TN)
    results.append({'condition': name, 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN, 'gt_positive': gt_pos, 'pred_positive': pred_pos, 'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4), 'accuracy': round(acc, 4)})

out = pd.DataFrame(results).sort_values('f1')
out.to_csv('/DATA/intern3/PriorRG/disease_scores_baseline_on_filtered_testset.csv', index=False)
print('\n=== TARGET 4 DISEASES (baseline, on SAME 951-sample filtered test set) ===')
print(out[out['condition'].isin(TARGET_DISEASES)].to_string(index=False))
