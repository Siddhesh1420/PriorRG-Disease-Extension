import os
import random
import logging
import argparse
import yaml
import numpy as np
import torch

from datetime import datetime


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='report-generation-gpt2')
    parser.add_argument('--phase', type=str, default='inference')

    parser.add_argument('--ann_path', type=str,
                        default='/DATA/intern3/PriorRG/checkpoints/mimic-cxr/radiology-reports/priorrg_mimic_cxr_annotation.json')

    parser.add_argument('--images_dir', type=str,
                        default='/DATA/sharingPoint/MIMIC/data/mimic_cxr_2.1.0/images/')

    parser.add_argument('--view_position_dict', type=str,
                        default='/DATA/intern3/PriorRG/checkpoints/mimic-cxr/radiology-reports/view-positions-dict-mimic.json')

    parser.add_argument('--test_ckpt_path', type=str,
                        default='/DATA/intern3/PriorRG/checkpoints/checkpoints/mimic-cxr/report-generation-gpt2/checkpoint/best_model.ckpt')

    parser.add_argument('--chexbert_path', type=str,
                        default='/DATA/intern3/PriorRG/checkpoints/chexbert.pth')

    parser.add_argument('--bert_path', type=str,
                        default='google-bert/bert-base-uncased')

    parser.add_argument('--radgraph_path', type=str,
                        default='/DATA/intern3/PriorRG/checkpoints/radgraph')

    parser.add_argument('--cxr_bert_path', type=str,
                        default='microsoft/BiomedVLP-CXR-BERT-specialized')

    parser.add_argument('--rad_dino_path', type=str,
                        default='microsoft/rad-dino')

    parser.add_argument('--distilgpt2_path', type=str,
                        default='distilbert/distilgpt2')

    parser.add_argument('--project_name', type=str,
                        default='PriorRG/mimic_cxr/report-generation-test')

    parser.add_argument('--version', type=str, default='v1')

    parser.add_argument('--use_structural_reasoning', action='store_true')

    parser.add_argument('--seed', type=int, default=9233)

    parser.add_argument('--num_gpus', type=int, default=1)

    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--epochs', type=int, default=1)

    parser.add_argument('--max_length', type=int, default=100)

    parser.add_argument('--encoder_max_length', type=int, default=300)

    parser.add_argument('--num_beams', type=int, default=3)

    parser.add_argument('--print_step', type=int, default=500)

    parser.add_argument('--hidden_size', type=int, default=768)

    parser.add_argument('--text_encoder_num_blocks', type=int, default=6)

    parser.add_argument('--temporal_fusion_num_blocks', type=int, default=3)

    parser.add_argument('--perceiver_num_blocks', type=int, default=3)

    parser.add_argument('--num_heads', type=int, default=8)

    parser.add_argument('--num_latents', type=int, default=128)

    parser.add_argument('--pt_lr', type=float, default=5e-6)

    parser.add_argument('--ft_lr', type=float, default=5e-5)

    parser.add_argument('--patience', type=int, default=10)

    parser.add_argument('--monitor_metric', type=str, default='RCB')

    parser.add_argument('--monitor_mode', type=str, default='max')

    parser.add_argument('--temp', type=float, default=0.5)

    parser.add_argument('--is_save_checkpoint', action='store_true')

    parser.add_argument('--load', type=str, default=None)

    parser.add_argument('--resume', type=str, default=None)

    args = vars(parser.parse_args())

    timestamp = datetime.now().strftime('%Y_%m_%d_%H')

    args['project_name'] = (
        f"{args['project_name']}/{args['version']}_{timestamp}"
    )

    os.makedirs(args['project_name'], exist_ok=True)

    config_dir = os.path.join(args['project_name'], 'configs')
    os.makedirs(config_dir, exist_ok=True)

    config_path = os.path.join(
        config_dir,
        f'config_{timestamp}.yaml'
    )

    with open(config_path, 'w') as f:
        yaml.dump(args, f, default_flow_style=False)

    logger = logging.getLogger('PriorRG')
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

    logger.info(
        f"parameters is saved in {config_path}"
    )

    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    return args, logger
