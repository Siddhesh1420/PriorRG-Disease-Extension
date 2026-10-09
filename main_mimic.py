import os
import time
import torch
import pytorch_lightning as pl
from transformers import GPT2TokenizerFast
from pytorch_lightning import seed_everything

from tools.utils_mimic import setup_arguments, setup_seed
from models.model_github_mimic import TrainLanguageModel

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.set_float32_matmul_precision('medium')


def main():
    args, logger = setup_arguments()

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

    trainer = pl.Trainer(
        accelerator="gpu",
        accumulate_grad_batches=2,
        benchmark=False,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=None,
        strategy=strategy,
        devices=args["num_gpus"],
        deterministic=True,
        max_epochs=args['epochs'],
        logger=None,
        log_every_n_steps=500,
        enable_model_summary=True,
        profiler="simple",
    )

    print("Running WHOLE MIMIC-CXR TEST SET inference...")
    print(f"Checkpoint: {args['test_ckpt_path']}")
    print(f"Annotation: {args['ann_path']}")
    print(f"Images: {args['images_dir']}")

    model = TrainLanguageModel(
        args=args,
        tokenizer=tokenizer,
        logger=logger
    )

    start = time.time()

    trainer.test(
        model=model,
        ckpt_path=args['test_ckpt_path']
    )

    end = time.time()

    print(f"\nWhole test-set inference time: {end - start:.6f} seconds")


if __name__ == '__main__':
    main()
