import sys, json, time
sys.path.insert(0, '/DATA/intern3/PriorRG')

from tools.metrics.f1chexbert import F1CheXbert

ANN_PATH = '/DATA/intern3/PriorRG/checkpoints/mimic-cxr/radiology-reports/priorrg_mimic_cxr_annotation.json'
CHEXBERT_CKPT = '/DATA/intern3/PriorRG/checkpoints/chexbert.pth'
BERT_PATH = 'google-bert/bert-base-uncased'
OUT_PATH = '/DATA/intern3/PriorRG/checkpoints/mimic-cxr/radiology-reports/train_chexbert_labels.json'

ann = json.load(open(ANN_PATH))['train']
print(f'Train split: {len(ann)} items')

labeler = F1CheXbert(
    chexbert_checkpoint=CHEXBERT_CKPT,
    model_checkpoint=BERT_PATH,
    tokenizer_checkpoint=BERT_PATH,
)

labels = {}
start = time.time()
for i, item in enumerate(ann):
    if len(item['findings_factual_serialization']) == 0:
        continue  # matches FinetuneDataset's own filter — skip samples it will skip too
    sample_id = f'{item["subject_id"]}_{item["study_id"]}_{item["id"]}'
    report = item['findings'].strip()
    labels[sample_id] = labeler.get_label(report, mode='rrg')  # 14 binary labels

    if (i + 1) % 2000 == 0:
        elapsed = time.time() - start
        rate = (i + 1) / elapsed
        remaining = (len(ann) - i - 1) / rate
        print(f'{i+1}/{len(ann)} done, {elapsed:.0f}s elapsed, ~{remaining/60:.1f} min remaining')

with open(OUT_PATH, 'w') as f:
    json.dump(labels, f)

print(f'\nSaved {len(labels)} label vectors to {OUT_PATH}')
print(f'Total time: {(time.time()-start)/60:.1f} min')
