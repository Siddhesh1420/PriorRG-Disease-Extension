import sys, os
sys.path.insert(0, '/DATA/intern3/PriorRG')
import pandas as pd
from tools.metrics.f1chexbert import F1CheXbert
CHEXBERT_CKPT = '/DATA/intern3/PriorRG/checkpoints/chexbert.pth'
BERT_PATH = 'google-bert/bert-base-uncased'
RUNS = {
    'baseline': '/DATA/intern3/PriorRG/PriorRG/mimic_cxr/report-generation-test/v1_2026_09_25_22/generated_reports/test_reports_epoch-0_26-09-2026_00-38-47.csv',
    'scot_v2_tuned': '/DATA/intern3/PriorRG/PriorRG/mimic_cxr/report-generation-test/v1_2026_10_02_15/generated_reports/test_reports_epoch-0_04-10-2026_19-55-11.csv',
    'optionB_pilot_defaultLR': '/DATA/intern3/PriorRG/PriorRG/mimic_cxr/report-generation-test/v1_2026_10_07_07/generated_reports/test_reports_epoch-0_07-10-2026_12-01-19.csv',
    'optionB_pilot_lowLR': '/DATA/intern3/PriorRG/PriorRG/mimic_cxr/report-generation-test/v1_2026_10_08_10/generated_reports/test_reports_epoch-0_08-10-2026_15-00-27.csv',
}
labeler = F1CheXbert(chexbert_checkpoint=CHEXBERT_CKPT, model_checkpoint=BERT_PATH, tokenizer_checkpoint=BERT_PATH)
names = labeler.target_names
for run_name, csv_path in RUNS.items():
    if not os.path.exists(csv_path):
        print(f'!!! MISSING: {run_name} -> {csv_path}'); continue
    out_path = f'/DATA/intern3/PriorRG/disease_scores_{run_name}.csv'
    if os.path.exists(out_path):
        print(f'{run_name}: already computed, skipping ({out_path})'); continue
    df = pd.read_csv(csv_path)
    print(f'{run_name}: loaded {len(df)} rows from {csv_path}')
    ref_labels_all, hyp_labels_all = [], []
    for i, row in df.iterrows():
        ref_labels_all.append(labeler.get_label(str(row['reference_report']).strip(), mode='rrg'))
        hyp_labels_all.append(labeler.get_label(str(row['generated_report']).strip(), mode='rrg'))
        if (i + 1) % 500 == 0: print(f'  {run_name}: {i+1}/{len(df)} done')
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
        results.append({'condition': name, 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN, 'gt_positive': gt_pos, 'pred_positive': pred_pos, 'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)})
    out = pd.DataFrame(results).sort_values('f1')
    out.to_csv(out_path, index=False)
    print(f'{run_name}: saved {out_path}')
    print(out.to_string(index=False))
    print()
print('ALL DONE')
