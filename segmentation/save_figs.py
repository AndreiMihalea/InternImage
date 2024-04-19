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
    'work_dirs/mask2former_internimage_b_kitti_balanced_sampler_30',
    'work_dirs/mask2former_internimage_b_imgcat_kitti_balanced_sampler_30_fix',
    'work_dirs/mask2former_internimage_b_imgcurv_kitti_balanced_sampler_30_fix',
    'work_dirs/mask2former_internimage_b_imgtext_kitti_balanced_sampler_30_fix',
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
        split_data = [line.split(',') for line in split_data]
        split_data = pd.DataFrame(split_data, columns=['frame', 'euler_angle', 'euler_angle_additive'])

    images_path = os.path.join(args.dataset_path, 'images')

    all_res_df = pd.DataFrame()
    models = {}

    for checkpoint in CHECKPOINTS:
        checkpoint_name = checkpoint.split('/')[-1]
        checkpoint_res = os.path.join(res_dir, f'{checkpoint_name}.csv')

        model_df = pd.read_csv(checkpoint_res)

        if all_res_df.empty:
            all_res_df = model_df
        else:
            model_df = model_df[[col for col in model_df.columns if checkpoint_name in col]]
            all_res_df = pd.concat([all_res_df, model_df], axis=1)

        config = checkpoint.split('/')[-1].split('_balanced_sampler')[0]
        config_path = os.path.join(MODEL_CONFIG_PATH, f'{config}.py')
        checkpoint_pth = glob.glob(f'{checkpoint}/best*.pth')[0]
        model = init_segmentor(config_path, checkpoint=None, device=args.device)
        checkpoint_dict = load_checkpoint(model, checkpoint_pth, map_location='cpu')

        model.CLASSES = checkpoint_dict['meta']['CLASSES']

        if args.soft_output:
            model.output_soft_head = True
            model.decode_head.output_soft_head = True

        models[checkpoint_name] = model

    all_frames = all_res_df['frame'].values.tolist()
    checkpoint_names = list(models.keys())

    for frame in all_frames[:-1]:
        frame_path = os.path.join(images_path, f'{frame}.png')
        image = cv2.imread(frame_path)

        angle = split_data[split_data['frame'] == f'{frame}.png']['euler_angle'].values[0]
        angle = float(angle)
        limits = [-float('inf'), *LIMITS, float('inf')]
        limits_scenarios = [-float('inf'), *LIMITS_SCENARIOS, float('inf')]
        category = bisect.bisect_right(limits, angle) - 1
        category_scenarios = bisect.bisect_right(limits_scenarios, angle) - 1
        ann_info = {'category': category, 'curvature': int(angle), 'scenario_text': category_scenarios}

        frame_res = all_res_df[all_res_df['frame'] == frame]

        all_bigger = True

        first_iou = frame_res[f'iou_{checkpoint_names[0]}'].values[0]

        for checkpoint_name in checkpoint_names[1:]:
            current_iou = frame_res[f'iou_{checkpoint_name}'].values[0]
            if current_iou > first_iou - 0.2:
                all_bigger = False

        if all_bigger:
            gt_path = frame_path.replace('images', f'self_supervised_labels_{horizon}').replace('segmentation_scsfm',
                                                                                                   'segmentation_gt')

            gt_label = cv2.imread(gt_path)[:, :, 0]
            colored_gt = colorize_mask(gt_label, (0, 255, 0))

            for checkpoint_name in checkpoint_names:
                model = models[checkpoint_name]
                result = inference_segmentor_custom(model, frame_path, ann_info)
                res = result[0].copy()
                colored_res = colorize_mask(res, (0, 0, 255))

                img_cp = image.copy()
                img_cp[np.logical_or(res, gt_label) != 0] //= 3

                mixed_gt_res = cv2.addWeighted(colored_gt, 1, colored_res, 1, 0.)

                final_img_gt_res = cv2.addWeighted(img_cp, 1, mixed_gt_res, 1, 0.)

                cv2.imwrite(os.path.join(save_dir, f'{frame}_{checkpoint_name}.png'), final_img_gt_res)


if __name__ == '__main__':
    main()
