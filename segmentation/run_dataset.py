# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import os
from argparse import ArgumentParser

import mmcv
import numpy as np
from tqdm import tqdm

import mmcv_custom  # noqa: F401,F403
import mmseg_custom  # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.apis.inference import LoadImage
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter
from mmseg.core import get_classes
from mmseg.datasets.pipelines import Compose
import cv2
import os.path as osp

import torch


LIMITS = [-80, -40, 0, 40, 80]
LIMITS_SCENARIOS = [-60, -18, 18, 60]


def inference_segmentor_custom(model, imgs, ann_info):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.
        ann_info (dict): Dictionary of annotations

    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = []
    imgs = imgs if isinstance(imgs, list) else [imgs]
    for img in imgs:
        img_data = dict(img=img, ann_info=ann_info)
        img_data = test_pipeline(img_data)
        data.append(img_data)
    data = collate(data, samples_per_gpu=len(imgs))
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]
    # print(data['img'][0].shape, 'shape')
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


alpha = 1.
beta = 1 - alpha


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--additional_config', help='Config file of an optional second model', default=None)
    parser.add_argument('--additional_checkpoint', help='Checkpoint file of an optional second model', default=None)
    parser.add_argument('--dataset_path', help='Path to the dataset',
                        default='/raid/andreim/kitti/data_odometry_color/segmentation/')
    parser.add_argument('--split', help='Split of the dataset')
    parser.add_argument('--use-all-data', action='store_true', help='Whether to use all the data or just the split')
    parser.add_argument('--save-good-bad', action='store_true', help='Whether to save good and bad examples')
    parser.add_argument('--horizon', type=int, help='Length of the prediction horizon')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='ade20k',
        choices=['ade20k', 'cityscapes', 'cocostuff'],
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--soft_output', action='store_true',
                        help='Specifies whether the network gives a soft output')

    args = parser.parse_args()

    save_good_bad = args.save_good_bad

    if save_good_bad:
        out_dirs_lv0 = ['good', 'bad']
        out_dirs_lv1 = ['straight', 'wide', 'tight']

        for lv0 in out_dirs_lv0:
            for lv1 in out_dirs_lv1:
                save_path = os.path.join(lv0, lv1)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

    # build the model from a config file and a checkpoint file

    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.additional_config and args.additional_checkpoint:
        additional_model = init_segmentor(args.additional_config, checkpoint=None, device=args.device)
        additional_checkpoint = load_checkpoint(additional_model, args.additional_checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)

    if 'CLASSES' in additional_checkpoint.get('meta', {}):
        additional_model.CLASSES = additional_checkpoint['meta']['CLASSES']
    else:
        additional_model.CLASSES = get_classes(args.palette)

    if args.soft_output:
        model.output_soft_head = True
        model.decode_head.output_soft_head = True
        if args.additional_config and args.additional_checkpoint:
            additional_model.output_soft_head = True
            additional_model.decode_head.output_soft_head = True

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

    # test a single image
    total_intersection = 0
    total_union = 0

    total_matching_pixels = 0
    total_pixels = 0

    if args.additional_config and args.additional_checkpoint:
        additional_model_total_intersection = 0
        additional_model_total_union = 0

        additional_model_total_matching_pixels  = 0
        additional_model_total_pixels  = 0

    for row in tqdm(split_data):
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

        gt_label = cv2.imread(gt_path)
        gt_label_og = gt_label.copy()
        gt_label[:, :, 1] = 0
        gt_label[:, :, 2] = 0

        ss_label = cv2.imread(label_path)
        ss_label_og = ss_label.copy()
        ss_label[:, :, 0] = 0
        ss_label[:, :, 2] = 0

        result = inference_segmentor_custom(model, image_path_og, ann_info)
        res = result[0].copy()
        res = np.repeat(res[:, :, np.newaxis], repeats=3, axis=2).astype(np.uint8)
        res_og = res.copy()
        res[:, :, 0] = 0
        res[:, :, 1] = 0

        gt_label_2d = gt_label[:, :, 0]
        res_2d = res[:, :, 2]
        intersection = np.logical_and(gt_label_2d, res_2d)
        union = np.logical_or(gt_label_2d, res_2d)

        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
        acc = np.sum(gt_label_2d == res_2d) / (gt_label_2d.size)

        total_intersection += np.sum(intersection)
        total_union += np.sum(union)

        mask = np.logical_and(gt_label_2d, res_2d)
        total_matching_pixels += np.sum((gt_label_2d == res_2d) * mask)
        total_pixels += np.sum(gt_label_2d)

        if args.additional_config and args.additional_checkpoint:
            # for angle in range(-180, 180, 10000):
            ann_info['curvature'] = angle
            additional_model_result = inference_segmentor_custom(additional_model, image_path_og, ann_info)
            additional_model_res = additional_model_result[0].copy()
            additional_model_res = np.repeat(additional_model_res[:, :, np.newaxis], repeats=3, axis=2).astype(np.uint8)
            additional_model_res_og = additional_model_res.copy()
            additional_model_res[:, :, 0] = 0
            additional_model_res[:, :, 2] = 0

            additional_model_res_2d = additional_model_res[:, :, 1]
            additional_model_intersection = np.logical_and(gt_label_2d, additional_model_res_2d)
            additional_model_union = np.logical_or(gt_label_2d, additional_model_res_2d)

            additional_model_iou = np.sum(additional_model_intersection) / np.sum(additional_model_union) if \
                np.sum(additional_model_union) > 0 else 0
            additional_model_acc = np.sum(gt_label_2d == additional_model_res_2d) / (gt_label_2d.size)

            additional_model_total_intersection += np.sum(additional_model_intersection)
            additional_model_total_union += np.sum(additional_model_union)

            additional_model_mask = np.logical_and(gt_label_2d, additional_model_res_2d)
            additional_model_total_matching_pixels += \
                np.sum((gt_label_2d == additional_model_res_2d) * additional_model_mask)
            additional_model_total_pixels += np.sum(gt_label_2d)

        img_og_cp_1 = img.copy()
        img_og_cp_2 = img.copy()

        # img_og[np.logical_or(gt_label_og, res_og) != 0] //= 3
        img_og_cp_1[res_og != 0] //= 2
        img_og_cp_2[ss_label_og != 0] //= 2
        img_res = cv2.addWeighted(res * 255, alpha, img_og_cp_1, 1, 0.5)
        img_label = cv2.addWeighted(ss_label * 255, alpha, img_og_cp_2, 1, 0.5)

        # cv2.imshow('res', img_res)
        # cv2.waitKey(0)

        # cv2.imwrite(f'demo/synasc/{split}_demo_rgb.png', img_og)
        # cv2.imwrite(f'demo/synasc/{split}_demo_label.png', ss_label * 255)
        # cv2.imwrite(f'demo/synasc/{split}_demo_res.png', res * 255)
        # cv2.imwrite(f'demo/synasc/{split}_demo_rgb_label.png', img_label)
        # cv2.imwrite(f'demo/synasc/{split}_demo_rgb_res.png', img_res)

        if args.additional_config and args.additional_checkpoint:
            img_cp = img.copy()
            img_cp[np.logical_or(additional_model_res[:, :, 1], res[:, :, 2]) != 0] //= 3
            additional_img_res = cv2.addWeighted(additional_model_res * 255, alpha, img_cp, 1, 0.5)
            # cv2.imshow(f'frame_{angle}', additional_img_res)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            additional_img_res = cv2.addWeighted(res * 255, alpha, additional_img_res, 1, 0.5)
            if additional_model_iou > iou + 0.2:
                cv2.imshow('improvement', additional_img_res)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        if save_good_bad:
            if np.abs(angle) > 60 and iou > 0.55:
                cv2.imwrite(os.path.join("good", "tight", image_file), img_res)
            elif iou < 0.2:
                cv2.imwrite(os.path.join("bad", "tight", image_file), img_res)
            elif np.abs(angle) < 3 and iou > 0.9:
                cv2.imwrite(os.path.join("good", "straight", image_file), img_res)
            elif np.abs(angle) < 3 and iou < 0.3:
                cv2.imwrite(os.path.join("bad", "straight", image_file), img_res)
            elif np.abs(angle) > 20 and np.abs(angle) < 40 and iou > 0.8:
                cv2.imwrite(os.path.join("good", "wide", image_file), img_res)
            elif iou < 0.4:
                cv2.imwrite(os.path.join("bad", "wide", image_file), img_res)
        # print(f"Result is saved at {out_path}")
    print(round(total_intersection / total_union * 100, 2), round(total_matching_pixels / total_pixels * 100, 2))
    if args.additional_config and args.additional_checkpoint:
        print(round(additional_model_total_intersection / additional_model_total_union * 100, 2),
              round(additional_model_total_matching_pixels / additional_model_total_pixels * 100, 2))


if __name__ == '__main__':
    main()
