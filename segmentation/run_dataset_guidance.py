# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import glob
import os
from argparse import ArgumentParser

import numpy as np
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
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--soft_output', action='store_true',
                        help='Specifies whether the network gives a soft output')

    args = parser.parse_args()

    horizon = args.horizon
    split = f'{args.split}_{horizon}'

    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    models = {}

    for checkpoint in CHECKPOINTS:
        checkpoint_name = checkpoint.split('/')[-1]
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

    images_path = os.path.join(args.dataset_path, 'images')

    split_path = os.path.join(args.dataset_path, 'splits', f'{split}.txt')
    with open(split_path, 'r') as f:
        split_data = f.readlines()

    for row in tqdm(split_data[::7]):
        row = row.strip()
        if len(row.split(',')) == 3:
            image_file, _, _ = row.split(',')
        else:
            image_file, _ = row.split(',')
        frame = image_file.split('.')[0]
        for angle in [-80, -60, -40, -20, 0, 20, 40, 60, 80]:
            angle = float(angle)
            limits = [-float('inf'), *LIMITS, float('inf')]
            limits_scenarios = [-float('inf'), *LIMITS_SCENARIOS, float('inf')]
            category_scenarios = bisect.bisect_right(limits_scenarios, angle) - 1
            # print(int(angle), limits[category], limits_scenarios[category_scenarios])
            image_path_og = os.path.join(images_path, image_file)
            gt_path = image_path_og.replace('images', f'self_supervised_labels_{horizon}').replace('segmentation_scsfm',
                                                                                                   'segmentation_gt')
            if not os.path.exists(gt_path):
                continue

            print(category, category_scenarios, angle)

            img = cv2.imread(image_path_og)

            gt_label = cv2.imread(gt_path)[:, :, 0].astype(np.float32)

            for checkpoint_name in models:
                if 'imgcat' in checkpoint_name:
                    guidance_category = category_scenarios
                elif 'imgcurv' in checkpoint_name:
                    guidance_category = int(angle)
                elif 'imgtext' in checkpoint_name:
                    guidance_category = category_scenarios
                else:
                    guidance_category = ''

                output_file = os.path.join(save_dir, f'{frame}_{checkpoint_name}_{guidance_category}.png')

                if os.path.exists(output_file):
                    continue

                ann_info = {'category': category_scenarios, 'curvature': int(angle), 'scenario_text': category_scenarios}

                model = models[checkpoint_name]
                result = inference_segmentor_custom(model, image_path_og, ann_info)
                res = result[0].copy()

                intersection = np.logical_and(gt_label, res)
                union = np.logical_or(gt_label, res)

                # cv2.imshow("soft", intersection.astype(np.float32))
                # cv2.waitKey(0)

                iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
                acc = np.sum(gt_label == res) / (gt_label.size)

                ann_info['scenario_text'] = angle

                colored_res = colorize_mask(res, (0, 0, 255))
                colored_gt = colorize_mask(gt_label, (0, 255, 0))
                mixed_gt_res = cv2.addWeighted(colored_gt, 1, colored_res, 1, 0.)
                img_cp = img.copy()
                img_cp[np.logical_or(res, gt_label) != 0] //= 3
                final_img_gt_res = cv2.addWeighted(img_cp, 1, mixed_gt_res, 1, 0.)

                # cv2.imshow('res', final_img_gt_res)
                # cv2.waitKey(0)

                cv2.imwrite(output_file, final_img_gt_res)




if __name__ == '__main__':
    main()
