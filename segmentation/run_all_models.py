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
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--soft_output', action='store_true',
                        help='Specifies whether the network gives a soft output')

    args = parser.parse_args()

    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    horizon = args.horizon
    split = f'{args.split}_{horizon}'

    split_path = os.path.join(args.dataset_path, 'splits', f'{split}.txt')
    with open(split_path, 'r') as f:
        split_data = f.readlines()

    images_path = os.path.join(args.dataset_path, 'images')

    for checkpoint in CHECKPOINTS:
        results = []

        checkpoint_name = checkpoint.split('/')[-1]
        config = '_'.join(checkpoint_name.split('_')[:5])
        config_path = os.path.join(MODEL_CONFIG_PATH, f'{config}.py')
        checkpoint_pth = glob.glob(f'{checkpoint}/best*.pth')[0]
        model = init_segmentor(config_path, checkpoint=None, device=args.device)
        checkpoint_dict = load_checkpoint(model, checkpoint_pth, map_location='cpu')

        model.CLASSES = checkpoint_dict['meta']['CLASSES']

        if args.soft_output:
            model.output_soft_head = True
            model.decode_head.output_soft_head = True

        total_intersection = 0
        total_union = 0

        total_matching_pixels = 0
        total_pixels = 0

        for row in tqdm(split_data[0:]):
            row = row.strip()
            if len(row.split(',')) == 3:
                image_file, angle, _ = row.split(',')
            else:
                image_file, angle = row.split(',')
            filename = image_file.replace('.png', '')
            # if filename != '08_frame001389':
            #     continue
            angle = float(angle)
            limits = [-float('inf'), *LIMITS, float('inf')]
            limits_scenarios = [-float('inf'), *LIMITS_SCENARIOS, float('inf')]
            category = bisect.bisect_right(limits, angle) - 1
            category_scenarios = bisect.bisect_right(limits_scenarios, angle) - 1
            ann_info = {'category': category, 'curvature': int(angle), 'scenario_text': category_scenarios}
            # if '09_' not in image_file:
            #     continue
            image_path_og = os.path.join(images_path, image_file)
            gt_path = image_path_og.replace('images', f'self_supervised_labels_{horizon}').replace('segmentation_scsfm',
                                                                                                   'segmentation_gt')
            label_path = image_path_og.replace('images', f'self_supervised_labels_{horizon}')
            if not os.path.exists(gt_path):
                continue

            gt_label = cv2.imread(gt_path)[:, :, 0]
            ss_label = cv2.imread(label_path)[:, :, 0]

            result = inference_segmentor_custom(model, image_path_og, ann_info)
            res = result[0].copy()
            # print(result[1][0][0].max(), result[1][0][0].min())
            # cv2.imshow("soft", result[1][0][1])
            # cv2.waitKey(0)

            intersection = np.logical_and(gt_label, res)
            union = np.logical_or(gt_label, res)

            iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
            acc = np.sum(gt_label == res) / (gt_label.size)

            results.append({'frame': filename, f'iou_{checkpoint_name}': iou, f'acc_{checkpoint_name}': acc})

            total_intersection += np.sum(intersection)
            total_union += np.sum(union)

            mask = np.logical_and(gt_label, res)
            total_matching_pixels += np.sum((gt_label == res) * mask)
            total_pixels += np.sum(gt_label)

        total_results = {
            'frame': 'all',
            f'iou_{checkpoint_name}': round(total_intersection / total_union * 100, 2),
            f'acc_{checkpoint_name}': round(total_matching_pixels / total_pixels * 100, 2)
        }

        per_frame_df = pd.DataFrame(results)
        per_frame_df = per_frame_df.append(total_results, ignore_index=True)
        per_frame_df.to_csv(f'{checkpoint_name}.csv')


if __name__ == '__main__':
    main()
