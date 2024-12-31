# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import os
from argparse import ArgumentParser

import numpy as np


from tqdm import tqdm

import mmcv_custom  # noqa: F401,F403
# import mmseg_custom  # noqa: F401,F403
from mmseg.apis import init_segmentor
from mmseg.apis.inference import LoadImage
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter
from mmseg.core import get_classes
from mmseg.datasets.pipelines import Compose
import cv2
import seaborn as sns

import torch

from segmentation.run_dataset import inference_segmentor_custom
from segmentation.utils import colorize_mask

LIMITS = [-80, -40, 0, 40, 80]
LIMITS_SCENARIOS = [-60, -18, 18, 60]



alpha = 0.5
beta = 1 - alpha


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--dataset_path', help='Path to the dataset',
                        default='/raid/andreim/kitti/data_odometry_color/segmentation/')
    parser.add_argument('--split', help='Split of the dataset')
    parser.add_argument('--use-all-data', action='store_true', help='Whether to use all the data or just the split')
    parser.add_argument('--save-good-bad', action='store_true', help='Whether to save good and bad examples')
    parser.add_argument('--horizon', type=int, help='Length of the prediction horizon')
    parser.add_argument('--save-dir', type=str, default="demo", help='save_dir dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # build the model from a config file and a checkpoint file

    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)

    horizon = args.horizon
    split = f'{args.split}_{horizon}'

    if not args.use_all_data:
        split_path = os.path.join(args.dataset_path, 'splits', f'{split}.txt')
        with open(split_path, 'r') as f:
            split_data = f.readlines()
            # print(split_data)
    else:
        # take all the splits
        split_data = []
        for split in ['val', 'train', 'test']:
            split_path = os.path.join(args.dataset_path, 'splits', f'{split}.txt')
            with open(split_path, 'r') as f:
                split_data.extend(f.readlines())

    images_path = os.path.join(args.dataset_path, 'images')

    for thr in range(45, 46, 1):
        total_intersection = 0
        total_union = 0

        total_matching_pixels = 0
        total_pixels = 0

        total_jaccard_upper = 0
        total_jaccard_lower = 0

        thr = thr / 100
        for row in tqdm(split_data[0:]):
            row = row.strip()
            if len(row.split(',')) == 3:
                image_file, angle, _ = row.split(',')
            else:
                image_file, angle = row.split(',')
            angle = float(angle)
            limits = [-float('inf'), *LIMITS, float('inf')]
            limits_scenarios = [-float('inf'), *LIMITS_SCENARIOS, float('inf')]
            category = bisect.bisect_right(limits, angle) - 1
            category_scenarios = bisect.bisect_right(limits_scenarios, angle) - 1
            ann_info = {'category': category, 'curvature': int(angle), 'scenario_text': category_scenarios}
            # print(image_file)
            # if '09_' not in image_file:
            #     continue
            image_path_og = os.path.join(images_path, image_file)
            gt_path = image_path_og.replace('images', f'self_supervised_labels_{horizon}').replace('segmentation_scsfm',
                                                                                                   'segmentation_gt')
            label_path = image_path_og.replace('images', f'self_supervised_labels_{horizon}')
            if not os.path.exists(gt_path):
                continue

            img = cv2.imread(image_path_og)

            gt_label = cv2.imread(gt_path)[:, :, 0].astype(np.float32)
            ss_label = cv2.imread(label_path)[:, :, 0].astype(np.float32)


            result = inference_segmentor_custom(model, image_path_og, ann_info)
            # print(type(result), len(result), type(result[0]), result[0].shape)
            res = result[0][1:].sum(axis=0).copy()

            thr_res = res.copy()
            thr_res[thr_res > thr] = 1
            thr_res[thr_res != 1] = 0

            res_show = np.repeat(res[:, :, np.newaxis], 3, axis=2)
            thr_res_show = np.repeat(thr_res[:, :, np.newaxis], 3, axis=2)

            arr_final = np.concatenate((img / 255., res_show, thr_res_show), axis=0)
            # res[res > 0.01] = 1
            # cv2.imshow('res', arr_final)
            # cv2.waitKey(0)
            # print(result[0][1].max(), result[0][1].min())
            heatmap = cv2.applyColorMap((result[0][1] * 255.).astype(np.uint8), cv2.COLORMAP_MAGMA)

            res = thr_res.copy()

            intersection = np.logical_and(gt_label, res)
            union = np.logical_or(gt_label, res)

            gt_label_copy = gt_label.copy()
            for _ in range(12):
                gt_label_copy = cv2.GaussianBlur(gt_label_copy.astype(np.float32), (11, 11), 5)

            abs_pred = np.mean(np.abs(res))
            abs_gt = np.mean(np.abs(gt_label_copy))
            abs_pred_minus_gt = np.mean(np.abs(res - gt_label_copy))

            total_jaccard_upper += abs_pred + abs_gt - abs_pred_minus_gt
            total_jaccard_lower += abs_pred + abs_gt + abs_pred_minus_gt

            # cv2.imshow("soft", heatmap)
            # cv2.waitKey(0)

            iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
            acc = np.sum(gt_label == res) / (gt_label.size)

            total_intersection += np.sum(intersection)
            total_union += np.sum(union)

            mask = np.logical_and(gt_label, res)
            total_matching_pixels += np.sum((gt_label == res) * mask)
            total_pixels += np.sum(gt_label)


            ann_info['scenario_text'] = angle

            colored_res = colorize_mask(res, (0, 0, 255))
            colored_gt = colorize_mask(gt_label, (0, 255, 0))
            mixed_gt_res = cv2.addWeighted(colored_gt, 1, colored_res, 1, 0.)
            img_cp_1 = img.copy()
            img_cp_1[np.logical_or(res, gt_label) != 0] //= 3
            final_img_gt_res = cv2.addWeighted(img_cp_1, 1, mixed_gt_res, 1, 0.)

            # cv2.imshow('final_img', final_img_gt_res)
            # cv2.waitKey()

            # cv2.imwrite(os.path.join(save_dir, f'{image_file}'.replace('.png', '_heatmap.png')), heatmap)
            # cv2.imwrite(os.path.join(save_dir, f'{image_file}'.replace('.png', '_threshold_lower.png')), final_img_gt_res)

        print(round(total_intersection / total_union * 100, 2), round(total_matching_pixels / total_pixels * 100, 2),
              thr, round(total_jaccard_upper / total_jaccard_lower * 100, 2))


if __name__ == '__main__':
    main()
