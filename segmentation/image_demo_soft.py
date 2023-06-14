# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv
import numpy as np

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


def inference_segmentor_custom(model, imgs):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

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
        img_data = dict(img=img)
        img_data = test_pipeline(img_data)
        data.append(img_data)
    data = collate(data, samples_per_gpu=len(imgs))
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('img', help='Image file')
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

    # build the model from a config file and a checkpoint file

    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)

    if args.soft_output:
        model.output_soft_head = True
        model.decode_head.output_soft_head = True

    # test a single image
    result, soft_result = inference_segmentor(model, args.img)
    print(result[0].shape, soft_result[0].shape)
    # show the results
    if hasattr(model, 'module'):
        model = model.module
    print(model.CLASSES, np.array([[255, 0, 0]]).shape[0])
    # model.CLASSES = [model.CLASSES]
    img = model.show_result(args.img, result,
                            palette=[[255, 0, 0], [0, 0, 255]],  # get_palette(args.palette),
                            show=False, opacity=args.opacity)
    mmcv.mkdir_or_exist(args.out)
    out_path = osp.join(args.out, osp.basename(args.img))
    cv2.imwrite(out_path, img)
    print(f"Result is saved at {out_path}")

    for it, soft_res in enumerate(soft_result[0]):
        hard_res = result[0].copy()
        print(hard_res.max(), hard_res.min())
        # hard_res[hard_res != it] = 0
        # hard_res[hard_res == 0] = 1
        og_img = cv2.imread(args.img) / 255.
        hard_res = hard_res[:, :, np.newaxis]
        hard_res = np.repeat(hard_res, 3, axis=2)
        soft_res = soft_res[:, :, np.newaxis]
        soft_res = np.repeat(soft_res, 3, axis=2)
        img_res = np.concatenate((og_img, hard_res, soft_res), axis=1)
        cv2.imshow('img', img_res)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break


if __name__ == '__main__':
    main()
