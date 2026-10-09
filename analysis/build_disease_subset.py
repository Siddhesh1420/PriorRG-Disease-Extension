import sys, json
sys.path.insert(0, '/DATA/intern3/PriorRG')
from tools.metrics.f1chexbert import F1CheXbert

ANN_PATH = '/DATA/intern3/PriorRG/checkpoints/mimic-cxr/radiology-reports/priorrg_mimic_cxr_annotation.json'
TRAIN_LABELS_PATH = '/DATA/intern3/PriorRG/checkpoints/mimic-cxr/radiology-reports/train_chexbert_labels.json'
OUT_PATH = '/DATA/intern3/PriorRG/checkpoints/mimic-cxr/radiology-reports/priorrg_mimic_cxr_annotation_4disease.json'
CHEXBERT_CKPT = '/DATA/intern3/PriorRG/checkpoints/chexbert.pth'
BERT_PATH = 'google-bert/bert-base-uncased'
TARGET_DISEASES = ['Pneumonia', 'Pneumothorax', 'Fracture', 'Pleural Other']

ann = json.load(open(ANN_PATH))
print({k: len(v) for k, v in ann.items()})

labeler = F1CheXbert(chexbert_checkpoint=CHEXBERT_CKPT, model_checkpoint=BERT_PATH, tokenizer_checkpoint=BERT_PATH)
names = list(labeler.target_names)
target_idx = [names.index(d) for d in TARGET_DISEASES]
print('target indices:', dict(zip(TARGET_DISEASES, target_idx)))

def has_target(label_vec):
    return any(label_vec[i] == 1 for i in target_idx)

# --- train: use precomputed labels (fast) ---
train_labels = json.load(open(TRAIN_LABELS_PATH))
train_filtered = []
for item in ann['train']:
    sid = f'{item["subject_id"]}_{item["study_id"]}_{item["id"]}'
    lab = train_labels.get(sid)
    if lab is not None and has_target(lab):
        train_filtered.append(item)
print(f'train: {len(train_filtered)} / {len(ann["train"])} kept')

# --- val/test: label on the fly (small, fast) ---
def filter_split(split_name):
    items = ann[split_name]
    kept = []
    for i, item in enumerate(items):
        report = (item.get('findings') or '').strip()
        if not report:
            continue
        lab = labeler.get_label(report, mode='rrg')
        if has_target(lab):
            kept.append(item)
        if (i + 1) % 1000 == 0:
            print(f'  {split_name}: {i+1}/{len(items)} scanned, {len(kept)} kept so far')
    print(f'{split_name}: {len(kept)} / {len(items)} kept')
    return kept

val_filtered = filter_split('val')
test_filtered = filter_split('test')

out = {'train': train_filtered, 'val': val_filtered, 'test': test_filtered}
with open(OUT_PATH, 'w') as f:
    json.dump(out, f)
print(f'\nSaved filtered annotation to {OUT_PATH}')
print({k: len(v) for k, v in out.items()})
