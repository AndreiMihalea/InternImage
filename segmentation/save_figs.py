# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import glob
import json
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm

import mmcv_custom  # noqa: F401,F403
import mmseg_custom  # noqa: F401,F403
from mmseg.apis import init_segmentor
from mmseg.apis.inference import LoadImage
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter
from mmseg.core import get_classes
from mmseg.datasets.pipelines import Compose
import cv2

import torch

from segmentation.run_dataset import inference_segmentor_custom
from segmentation.utils import colorize_mask

LIMITS = [-80, -40, 0, 40, 80]
LIMITS_SCENARIOS = [-60, -18, 18, 60]


alpha = 0.5
beta = 1 - alpha

# TODO only take what's after 18 feb
CHECKPOINTS = [
    'work_dirs/mask2former_internimage_b_attcurv_kitti_balanced_sampler_30',
    # 'work_dirs/mask2former_internimage_b_attcat_kitti_balanced_sampler_30',
    # 'work_dirs/mask2former_internimage_b_atttext_kitti_balanced_sampler_30',
    # 'work_dirs/mask2former_internimage_b_attcurv_kitti_balanced_sampler_30',
]

MODEL_CONFIG_PATH = 'configs/mask2former'

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', help='Path to the dataset',
                        default='/raid/andreim/kitti/data_odometry_color/segmentation/')
    parser.add_argument('--split', help='Split of the dataset', default='test')
    parser.add_argument('--horizon', type=int, help='Length of the prediction horizon', default=30)
    parser.add_argument('--save-dir', type=str, help='Path to the directory where stuff is saved')
    parser.add_argument('--res-dir', type=str, help='Path to the directory where results are loaded from')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--soft_output', action='store_true',
                        help='Specifies whether the network gives a soft output')

    args = parser.parse_args()

    save_dir = args.save_dir
    res_dir = args.res_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    horizon = args.horizon
    split = f'{args.split}_{horizon}'

    split_path = os.path.join(args.dataset_path, 'splits', f'{split}.txt')
    with open(split_path, 'r') as f:
        split_data = f.readlines()

    images_path = os.path.join(args.dataset_path, 'images')

    results_df = None

    for checkpoint in CHECKPOINTS:
        checkpoint_name = checkpoint.split('/')[-1]
        checkpoint_res = os.path.join(res_dir, f'{checkpoint_name}.json')
        checkpoint_data = json.load(open(checkpoint_res, 'r'))

        model_df = pd.DataFrame(checkpoint_data['per_frame'])
        model_df.rename(columns={'iou': f'iou_{checkpoint_name}', 'acc': f'acc_{checkpoint_name}'}, inplace=True)

        if not results_df:
            results_df = model_df
            print(results_df.head(5))

    for checkpoint in CHECKPOINTS:

        config = '_'.join(checkpoint.split('/')[-1].split('_')[:5])
        config_path = os.path.join(MODEL_CONFIG_PATH, f'{config}.py')
        checkpoint_pth = glob.glob(f'{checkpoint}/best*.pth')[0]
        model = init_segmentor(config_path, checkpoint=None, device=args.device)
        checkpoint_dict = load_checkpoint(model, checkpoint_pth, map_location='cpu')

        model.CLASSES = checkpoint_dict['meta']['CLASSES']

        if args.soft_output:
            model.output_soft_head = True
            model.decode_head.output_soft_head = True




if __name__ == '__main__':
    main()
