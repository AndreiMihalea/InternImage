# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.pipelines.formatting import to_tensor


@PIPELINES.register_module(force=True)
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """
    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            # seg = results['gt_semantic_seg'].copy()
            # seg = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
            # cv2.imshow("img", np.concatenate([img.astype(np.float32), seg.astype(np.float32)]))
            # cv2.imshow("img", img.astype(np.float32))
            # print(results['ann_info'], img.shape, seg.shape)
            # cv2.waitKey(0)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        if 'gt_semantic_seg' in results:
            # convert to long
            results['gt_semantic_seg'] = DC(to_tensor(
                results['gt_semantic_seg'][None, ...].astype(np.int64)),
                                            stack=True)
        if 'gt_masks' in results:
            results['gt_masks'] = DC(to_tensor(results['gt_masks']))
        if 'gt_labels' in results:
            results['gt_labels'] = DC(to_tensor(results['gt_labels']))
        if 'gt_soft_masks' in results:
            results['gt_soft_masks'] = DC(to_tensor(results['gt_soft_masks']))

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class ToMask(object):
    """Transfer gt_semantic_seg to binary mask and generate gt_labels."""
    def __init__(self, ignore_index=2):
        self.ignore_index = ignore_index

    def __call__(self, results):
        gt_semantic_seg = results['gt_semantic_seg']
        gt_semantic_seg[gt_semantic_seg == 255] = 1
        gt_labels = np.unique(gt_semantic_seg)
        # print(gt_labels)
        # remove ignored region
        gt_labels = gt_labels[gt_labels != self.ignore_index]

        gt_masks = []
        for class_id in gt_labels:
            gt_masks.append(gt_semantic_seg == class_id)

        if len(gt_masks) == 0:
            # Some image does not have annotation (all ignored)
            gt_masks = np.empty((0, ) + results['pad_shape'][:-1], dtype=np.int64)
            gt_labels = np.empty((0, ),  dtype=np.int64)
        else:
            gt_masks = np.asarray(gt_masks, dtype=np.int64)
            gt_labels = np.asarray(gt_labels, dtype=np.int64)

        results['gt_labels'] = gt_labels
        results['gt_masks'] = gt_masks
        # print(results.keys())
        # print(gt_masks.shape, 'shape here')
        # for mask in gt_masks:
        #     if mask.shape[0] != 0:
        #         cv2.imshow("mask", mask.astype(np.float32))
        #         cv2.waitKey(0)

        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(ignore_index={self.ignore_index})'


@PIPELINES.register_module()
class ToSoft:
    """
    Turn class segmentation into soft labels (this is thought for two classes)
    """
    def __init__(self, num_iter, kernel_size, std_dev):
        self.num_iter = num_iter
        self.kernel_size = kernel_size
        self.std_dev = std_dev

    def __call__(self, input_dict):
        gt_masks = input_dict['gt_masks'].copy()

        gt_soft_masks = []

        if len(gt_masks) == 0:
            gt_soft_masks = np.empty((0,) + input_dict['pad_shape'][:-1], dtype=np.float32)
            input_dict['gt_soft_masks'] = gt_soft_masks
        else:
            for gt_mask in gt_masks:
                gt_mask_copy = gt_mask.copy()
                for _ in range(self.num_iter):
                    gt_mask_copy = cv2.GaussianBlur(gt_mask_copy.astype(np.float32), self.kernel_size, self.std_dev)
                gt_soft_masks.append(gt_mask_copy)

            input_dict['gt_soft_masks'] = np.array(gt_soft_masks)

        return input_dict


@PIPELINES.register_module()
class LoadCategory(object):

    def __init__(self):
        pass

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        results['category'] = results['ann_info']['category']
        results['curvature'] = results['ann_info']['curvature']
        results['scenario_text'] = results['ann_info']['scenario_text']
        return results

