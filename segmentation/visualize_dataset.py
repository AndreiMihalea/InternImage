# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser

import mmcv
import numpy as np
from tqdm import tqdm

import cv2
import os.path as osp

import torch


def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', help='Path to the dataset',
                        default='/raid/andreim/kitti/data_odometry_color/segmentation/')
    parser.add_argument('--split', help='Split of the dataset')
    parser.add_argument('--horizon', type=int, help='Length of the prediction horizon')
    parser.add_argument('--curvature_min', type=float, default=18, help='Minimum value of curvature to display')
    parser.add_argument('--curvature_max', type=float, default=60, help='Maximum value of curvature to display')
    parser.add_argument('--use-all-data', action='store_true', help='Whether to use all the data or just the split')

    args = parser.parse_args()

    # build the model from a config file and a checkpoint file

    horizon = args.horizon
    curvature_min = args.curvature_min
    curvature_max = args.curvature_max

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

    for row in tqdm(split_data):
        row = row.strip()
        image_file, angle = row.split(',')
        angle = float(angle)

        if curvature_min <= angle <= curvature_max:
            print(angle)
            image_path_og = os.path.join(images_path, image_file)
            gt_path = image_path_og.replace('images', f'self_supervised_labels_{horizon}').replace('segmentation_scsfm',
                                                                                                    'segmentation_gt')
            if not os.path.exists(gt_path):
                continue

            # print(gt_path)
            gt_label = cv2.imread(gt_path)
            gt_label[:, :, 1] = 0
            gt_label[:, :, 2] = 0

            label_path = image_path_og.replace('images', f'self_supervised_labels_{horizon}')

            ss_label = cv2.imread(label_path)
            ss_label_og = ss_label.copy()
            ss_label[:, :, 0] = 0
            ss_label[:, :, 2] = 0

            img_og = cv2.imread(image_path_og)

            alpha = 1.
            beta = 1 - alpha
            # img_og[np.logical_or(gt_label_og, res_og) != 0] //= 3
            img_og[ss_label_og != 0] //= 2
            img_label = cv2.addWeighted(ss_label * 255, alpha, img_og, 1, 0.5)

            cv2.imshow('res', img_label)
            cv2.waitKey(0)




if __name__ == '__main__':
    main()
