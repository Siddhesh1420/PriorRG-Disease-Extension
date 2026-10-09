import os
import time
import torch
import pytorch_lightning as pl
from transformers import GPT2TokenizerFast
from pytorch_lightning import seed_everything

from tools.utils_mimic import setup_arguments, setup_seed
from models.model_github_mimic_priorreport import TrainLanguageModel

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.set_float32_matmul_precision('medium')


def main():
    args, logger = setup_arguments()
    args['use_prior_report_fusion'] = True
    args['pt_lr'] = 5e-7  # 10x lower than default 5e-6, tested after full-scale run showed CheXbert drift  # explicit: enable the new fusion module (Option B)

    setup_seed(args['seed'])
    seed_everything(args['seed'])

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=0,
        verbose=False,
        save_last=False,
        save_weights_only=False,
    )

    tokenizer = GPT2TokenizerFast.from_pretrained(args['distilgpt2_path'])

    tokenizer.add_special_tokens({
        'pad_token': '[PAD]',
        'sep_token': '[SEP]',
        'cls_token': '[CLS]'
    })

    tokenizer.add_tokens([
        '[INDICATION]',
        '[HISTORY]',
        '[Similar Cases]',
        '[FINDINGS]'
    ])

    params = ''
    for key, value in args.items():
        params += f'{key}:\t{value}\n'

    logger.info(params)

    strategy = 'auto'

    if args['num_gpus'] > 1:
        strategy = 'ddp_find_unused_parameters_true'

    print(f"Checkpoint: {args['test_ckpt_path']}")
    print(f"Annotation: {args['ann_path']}")
    print(f"Images: {args['images_dir']}")

    model = TrainLanguageModel(
        args=args,
        tokenizer=tokenizer,
        logger=logger
    )

    # ---- Non-strict checkpoint load ----
    # Checkpoint predates PriorReportFusionModule, so prior_report_fusion.* is
    # missing from it. Non-strict restores every pretrained weight and leaves
    # prior_report_fusion randomly (but seeded) initialized.
    ckpt = torch.load(args['test_ckpt_path'], map_location='cpu')
    state_dict = ckpt['state_dict']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    bad_missing = [k for k in missing if not k.startswith('prior_report_fusion.')]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint load mismatch beyond prior_report_fusion.*\n"
            f"Unexpected missing: {bad_missing}\n"
            f"Unexpected extra:   {unexpected}"
        )
    print(f"OK: checkpoint restored, {len(missing)} prior_report_fusion params freshly initialized:")
    for k in missing:
        print(f"  {k}")

    # ---- Freeze everything except the new module ----
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith('prior_report_fusion.')

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"\nTraining only {len(trainable)} params:")
    for n in trainable:
        print(f"  {n}")

    # ---- Pilot: small train/val subset, 1 epoch ----
    pilot_trainer = pl.Trainer(
        accelerator="gpu",
        accumulate_grad_batches=2,
        benchmark=False,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
        strategy=strategy,
        devices=args["num_gpus"],
        deterministic=True,
        max_epochs=1,
        limit_train_batches=1.0,  # full-scale run with low pt_lr=5e-7 (pilot confirmed improvement)
        limit_val_batches=1.0,  # full-scale run with low pt_lr=5e-7 (pilot confirmed improvement)
        logger=None,
        log_every_n_steps=500,
        enable_model_summary=True,
        profiler="simple",
    )

    start = time.time()
    pilot_trainer.fit(model=model)
    end = time.time()
    print(f"\nPilot training time: {end - start:.6f} seconds")

    # ---- Full test-set eval ----
    test_trainer = pl.Trainer(
        accelerator="gpu",
        strategy=strategy,
        devices=args["num_gpus"],
        deterministic=True,
        logger=None,
        limit_test_batches=1.0,
        profiler="simple",
    )

    start = time.time()
    test_trainer.test(model=model)
    end = time.time()
    print(f"\nPilot test-set inference time: {end - start:.6f} seconds")


if __name__ == '__main__':
    main()
