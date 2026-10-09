import argparse
import datetime
import os
import random
import re
import time

import numpy as np
import torch
import yaml
from dateutil import tz
import pytorch_lightning as pl
from PIL import Image
from transformers import GPT2TokenizerFast
from pytorch_lightning import seed_everything

# from tools.utils_github import setup_arguments, setup_seed
from tools.metrics.metrics import compute_all_scores
from models.model_github import TrainLanguageModelOneSample

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.set_float32_matmul_precision('medium')


def str2bool(value):
    if value.lower() in ['yes', 'true', 't', '1']:
        return True
    elif value.lower() in ['no', 'false', 'f', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_arguments():
    # -------------------------------
    # load hyper-param
    # -------------------------------
    parse = argparse.ArgumentParser()
    # basic configuration
    parse.add_argument('--task', type=str, default='pretraining',
                       choices=['report-generation-single-sample'],
                       help='the task to run. gpt2 is DistilGPT2. ')
    parse.add_argument('--phase', type=str, default='inference', choices=['inference'],
                       help='Is this task for fine-tuning or inference mode?')
    # data configuration
    parse.add_argument('--data_name', type=str, choices=['mimic_cxr', 'mimic_abn'], default='mimic_cxr')
    parse.add_argument('--ann_path', type=str, help='annotation for radiology reports',
                       default='/DATA/sharingPoint/MIMIC/data/mimic_cxr_2.1.0/annotation.json',
                       )
    parse.add_argument('--view_position_dict', type=str, help='the dictory of view positions',
                       default='/DATA/intern3/PriorRG/checkpoints/mimic-cxr/radiology-reports/view-positions-dict-mimic.json'
                       )
    parse.add_argument('--images_dir', type=str, default='/DATA/sharingPoint/MIMIC/data/mimic_cxr_2.1.0/images/',
                       help='the directory of images')
    parse.add_argument('--max_length', type=int, default=100, help='the maximum number of generated tokens')
    parse.add_argument('--encoder_max_length', type=int, default=300,
                       help='The maximum number of tokens encoded by the text encoder to avoid excessive memory consumption')
    parse.add_argument('--num_workers', type=int, default=0)
    parse.add_argument('--epochs', type=int, default=50)
    parse.add_argument('--batch_size', type=int, default=32)
    parse.add_argument('--num_gpus', type=int, default=1, help='the number of gpus')
    parse.add_argument('--patience', type=int, default=10, help='used for learning rate')
    parse.add_argument('--is_save_checkpoint', type=str2bool, default='no', help='whether save checkpoint')
    parse.add_argument('--online_checkpoint', type=str2bool, default='no', help='whether using online checkpoint')
    parse.add_argument('--ckpt_zoo_dir', type=str,
                       default='/DATA/intern3/PriorRG/checkpoints/checkpoints/',
                       help='if using local checkpoint, this variable must be provided')
    parse.add_argument('--text_encoder_num_blocks', type=int, default=6)
    parse.add_argument('--temporal_fusion_num_blocks', type=int, default=3)
    parse.add_argument('--perceiver_num_blocks', type=int, default=3)
    parse.add_argument('--num_heads', type=int, default=8)
    parse.add_argument('--num_latents', type=int, default=128, help='the number of latents used in Perceiver.')

    # trainer configuration-
    parse.add_argument('--pt_lr', type=float, default=5.0e-6, help='learning rate of pretraining module.')  # 5.0e-5
    parse.add_argument('--ft_lr', type=float, default=5.0e-5, help='learning rate of non-pretraining module, '
                                                                   'e.g., language generator')  # 5.0e-5
    parse.add_argument('--temp', type=float, default=0.5,
                       help='temperature parameter for instance-wise alignment')  # 5.0e-5
    parse.add_argument('--monitor_metric', type=str, default='RCB',
                       help='the metric is used to selecting best models. '
                            'align phase is all_loss, while fine-tuning is RCB (bleu4+f1-chexpert+f1-radgraph)')
    # choices={all_metrics, metrics, RC, RB, RCB}
    parse.add_argument('--hidden_size', type=int, default=768, help='the dimension of unified space between multimodal')
    parse.add_argument('--test_ckpt_path', type=str, help='checkpoint for test',
                       default='/DATA/intern3/PriorRG/checkpoints/checkpoints/mimic-cxr/report-generation-gpt2/checkpoint/best_model.ckpt'
                       # default='/home/miao/data/Code/PriorRG/script/results/mimic_cxr/align/v0207-align_2025_02_07_22-best/checkpoint/best_model.ckpt'
                       )
    parse.add_argument('--project_name', type=str, default='PriorRG', help='the name of the project')
    # ========= metrics checkpoint config =====#
    parse.add_argument('--chexbert_path', type=str, default='/DATA/intern3/PriorRG/checkpoints/chexbert.pth', help='checkpoint for f1-chexbert')
    parse.add_argument('--bert_path', type=str, default='google-bert/bert-base-uncased', help='checkpoint for f1-chexbert')
    parse.add_argument('--radgraph_path', type=str, default='/DATA/intern3/PriorRG/checkpoints/radgraph', help='checkpoint for f1-radgraph')
    # ========= backbone checkpoint config =====#
    parse.add_argument('--cxr_bert_path', type=str, default='microsoft/BiomedVLP-CXR-BERT-specialized',
                       help='checkpoint for text-encoder')
    parse.add_argument('--rad_dino_path', type=str, default='microsoft/rad-dino',
                       help='checkpoint of image encoder')
    parse.add_argument('--llama_path', type=str, default='meta-llama/Llama-3.2-3B-Instruct',
                       help='checkpoint of language generator')
    parse.add_argument('--distilgpt2_path', type=str, default='distilbert/distilgpt2',
                       help='checkpoint of language generator')
    # ========= implementation config =====#
    parse.add_argument('--seed', type=int, default=9233, help='random seed')
    parse.add_argument('--num_beams', type=int, default=3, help='beam size for language generation')
    # ========= record results ============#
    parse.add_argument('--version', type=str, default='v1',
                       help='the name of version')
    parse.add_argument('--print_step', type=int, default=500, help='the frequency of print')
    # =============finish=====================#
    args = parse.parse_args()
    args = vars(args)
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H")
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    args['project_name'] = f'{args["project_name"]}/{args["data_name"]}/{args["task"]}/{args["version"]}_{extension}'
    os.makedirs(args['project_name'], exist_ok=True)

    # determine absolute path for checkpoints
    if not args['online_checkpoint']:
        candi_list = ['chexbert_path', 'radgraph_path', "bert_path", 'cxr_bert_path',
                      "distilgpt2_path", "rad_dino_path", 'llama_path']
    else:  # for clinical efficacy metrics
        candi_list = ['chexbert_path', 'radgraph_path']

    for candi in candi_list:
        if args[candi] is None:
            continue
        if '/' in args[candi] and not args[candi].startswith('/'):
            if args[candi].startswith(('microsoft/', 'distilbert/', 'google-bert/' , 'google/', 'meta-llama/')):
            	continue

        args[candi] = os.path.join(args['ckpt_zoo_dir'], args[candi])
    # determine the monitor_mode
    args['monitor_mode'] = 'max'
    if args['task'] == 'pretraining':  # pretrain
        args['monitor_mode'] = 'min'
        args['monitor_metric'] = 'val_epoch_loss'

    checkpoint_dir = os.path.join(args['project_name'], 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    args['checkpoint_dir'] = checkpoint_dir
    args['time'] = extension
    # save parameters
    config_dir = f"{args['project_name']}/configs"
    os.makedirs(config_dir, exist_ok=True)
    file_name = f"{config_dir}/config_{extension}.yaml"
    print(f'parameters is saved in {file_name}')
    with open(file_name, 'w') as file:
        yaml.dump(args, file, default_flow_style=False)
    return args


def setup_seed(seed):
    # seed init
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch seed init
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def obtain_aligned_reference_reports(tokenizer, text, max_length):
    inputs = tokenizer(text, padding=True, max_length=max_length,
                       truncation=True, return_tensors='pt')
    ref_reports = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
    # delete illegal characters
    ref_reports = [re.sub(r'[^\x20-\x7E]', '', report.strip()) for report in ref_reports]
    return ref_reports


def build_arguments_model():
    """
    only support inference mode for report generation
    """
    args = setup_arguments()
    setup_seed(args['seed'])
    seed_everything(args['seed'])

    tokenizer = GPT2TokenizerFast.from_pretrained(args['distilgpt2_path'])
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]', 'cls_token': '[CLS]'})
    tokenizer.add_tokens(['[INDICATION]', '[HISTORY]', '[Similar Cases]', '[FINDINGS]'])

    # save the config
    params = ''
    for key, value in args.items():
        params += f'{key}:\t{value}\n'
    print(params)

    # Trainer
    model = TrainLanguageModelOneSample(args, tokenizer)

    cur_model_state = model.state_dict()
    pre_model_state = torch.load(args['test_ckpt_path'])['state_dict']
    valid_state = {k: v for k, v in pre_model_state.items() if
                   k in cur_model_state and v.shape == cur_model_state[k].shape}
    invalid_state = {k for k in pre_model_state.keys() if k not in valid_state}
    print(f"missing {invalid_state}")
    cur_model_state.update(valid_state)
    model.load_state_dict(cur_model_state)

    return model, args


def structured_user_data(image_processor, sep_token, device, current_study, prior_study):

    # -----------------------------
    # Current study
    # -----------------------------
    cur_context = (
        current_study['indication_pure'].strip()
        + f' {sep_token} '
        + current_study['history_pure']
    )

    cur_images = Image.open(current_study['image_path']).convert('RGB')
    cur_images = image_processor(
        cur_images,
        return_tensors='pt'
    ).pixel_values.to(device)

    # -----------------------------
    # Prior study
    # -----------------------------
    if prior_study is not None:

        pri_image = Image.open(
            prior_study['image_path']
        ).convert('RGB')

        pri_image = image_processor(
            pri_image,
            return_tensors='pt'
        ).pixel_values.to(device)

        prior_study_data = {
            'image': pri_image,
            'view_position': [prior_study['view_position']],
            'pri_idx': [0],
            'no_pri_idx': []
        }

    else:

        # Genuine no-prior condition
        prior_study_data = {
            'image': None,
            'view_position': [],
            'pri_idx': [],
            'no_pri_idx': [0]
        }

    # -----------------------------
    # Final model input
    # -----------------------------
    item = {
        'image_ids': [current_study['id']],

        'current_study': {
            'image': cur_images,
            'view_position': [current_study['view_position']],
        },

        'clinical_context': [cur_context],

        'prior_study': prior_study_data,
    }

    return item


def main():

    # ============================================================
    # IU-XRay single-study inference
    # ============================================================

    current_study = {
    'id': 'p19/p19470900/s54867941',
    'image_path': '/DATA/sharingPoint/MIMIC/data/mimic_cxr_2.1.0/images/p19/p19470900/s54867941/4010cf17-87296e32-f34f756c-98ac1b91-da1493f5.jpg',
    'view_position': 'AP',
    'indication_pure': '',
    'history_pure': '',
}

    # No prior study for this test
    prior_study = {
    'image_path': '/DATA/sharingPoint/MIMIC/data/mimic_cxr_2.1.0/images/p19/p19470900/s51280938/5327ddc3-53497167-efbd3504-954526c3-dad201a9.jpg',
    'view_position': 'AP'
}

    # Set to None first so we only test report generation.
    # Later we can add the actual IU-XRay reference report
    # and evaluate BLEU / ROUGE / METEOR / CheXbert / RadGraph.
    reference_report = """FINAL REPORT PORTABLE AP CHEST X-RAY INDICATION: Patient with history of cirrhosis, posterior fossa hemorrhage, decompression, interval change. COMPARISON: ___ to ___. Slight worsening of very mild pulmonary edema. Bibasilar atelectasis has, however, slightly improved. There is no pleural effusion or pneumothorax. ET tube is now too high ending ___.9 cm above the carina, suggest pushing at least 3-4 cm. Right subclavian line is in lower SVC. Dobbhoff tube is in the stomach. CONCLUSION: 1. Mild pulmonary edema has slightly worsened. 2. ET tube is too high, suggest advancing 3-4 cm. The ICU team have been verbally contacted for the results."""

    # ============================================================
    # Load PriorRG model
    # ============================================================

    model, args = build_arguments_model()

    model = model.to(args['device'])
    model.eval()

    image_processor = model.image_processor
    sep_token = model.tokenizer.sep_token

    # ============================================================
    # Prepare IU-XRay image
    # ============================================================

    item = structured_user_data(
        image_processor,
        sep_token,
        args['device'],
        current_study,
        prior_study
    )

    # ============================================================
    # Generate report
    # ============================================================

    generated_report = model(item)

    print(
        f'current image id is {current_study["id"]}, '
        f'generated report is {generated_report}'
    )

    # ============================================================
    # Optional evaluation
    # ============================================================

    if reference_report is not None:

        aligned_reference_report = obtain_aligned_reference_reports(
            model.tokenizer,
            reference_report,
            args['max_length']
        )

        aligned_scores = compute_all_scores(
            generated_report,
            aligned_reference_report,
            args
        )

        print("aligned version performance: ", aligned_scores)

        not_aligned_scores = compute_all_scores(
            generated_report,
            [reference_report],
            args
        )

        print("not aligned version performance: ", not_aligned_scores)


if __name__ == '__main__':
    main()

