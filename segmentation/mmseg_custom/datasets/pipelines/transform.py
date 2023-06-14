# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect

import cv2
import mmcv
import numpy as np
import torch
from mmseg.datasets.builder import PIPELINES

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


@PIPELINES.register_module()
class SETR_Resize(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio range
    and multiply it with the image scale.

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range.

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """
    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 crop_size=None,
                 setr_multi_scale=False):

        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            # assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.crop_size = crop_size
        self.setr_multi_scale = setr_multi_scale

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """
        
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None
    
    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""

        if self.keep_ratio:
            if self.setr_multi_scale:
                if min(results['scale']) < self.crop_size[0]:
                    new_short = self.crop_size[0]
                else:
                    new_short = min(results['scale'])

                h, w = results['img'].shape[:2]
                if h > w:
                    new_h, new_w = new_short * h / w, new_short
                else:
                    new_h, new_w = new_short, new_short * w / h
                results['scale'] = (new_h, new_w)

            img, scale_factor = mmcv.imrescale(results['img'],
                                               results['scale'],
                                               return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(results['img'],
                                                  results['scale'],
                                                  return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(results[key],
                                        results['scale'],
                                        interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(results[key],
                                       results['scale'],
                                       interpolation='nearest')
            results['gt_semantic_seg'] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str


@PIPELINES.register_module()
class PadShortSide(object):
    """Pad the image & mask.

    Pad to the minimum size that is equal or larger than a number.
    Added keys are "pad_shape", "pad_fixed_size",

    Args:
        size (int, optional): Fixed padding size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """
    def __init__(self, size=None, pad_val=0, seg_pad_val=255):
        self.size = size
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        h, w = results['img'].shape[:2]
        new_h = max(h, self.size)
        new_w = max(w, self.size)
        padded_img = mmcv.impad(results['img'],
                                shape=(new_h, new_w),
                                pad_val=self.pad_val)

        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        # results['unpad_shape'] = (h, w)

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(results[key],
                                      shape=results['pad_shape'][:2],
                                      pad_val=self.seg_pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        h, w = results['img'].shape[:2]
        if h >= self.size and w >= self.size:  # 短边比窗口大，跳过
            pass
        else:
            self._pad_img(results)
            self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class MapillaryHack(object):
    """map MV 65 class to 19 class like Cityscapes."""
    def __init__(self):
        self.map = [[13, 24, 41], [2, 15], [17], [6], [3],
                    [45, 47], [48], [50], [30], [29], [27], [19], [20, 21, 22],
                    [55], [61], [54], [58], [57], [52]]

        self.others = [i for i in range(66)]
        for i in self.map:
            for j in i:
                if j in self.others:
                    self.others.remove(j)

    def __call__(self, results):
        """Call function to process the image with gamma correction.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """
        gt_map = results['gt_semantic_seg']
        # others -> 255
        new_gt_map = np.zeros_like(gt_map)

        for value in self.others:
            new_gt_map[gt_map == value] = 255

        for index, map in enumerate(self.map):
            for value in map:
                new_gt_map[gt_map == value] = index

        results['gt_semantic_seg'] = new_gt_map

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class CustomCrop:
    def __init__(self, left, right, top, bot):
        """
        :param left: percentage to crop from the left
        :param right: percentage to crop from the right
        :param top: percentage to crop from the top
        :param bot: percentage to crop from the bottom
        """
        self.left = left
        self.right = right
        self.top = top
        self.bot = bot

    def __call__(self, input_dict):
        for key in ['img', 'gt_semantic_seg']:
            if key not in input_dict:
                continue

            img = input_dict[key]
            # unique_values = np.unique(img)

            height, width = img.shape[:2]

            crop_pixels_top = int(height * self.top)
            crop_pixels_bottom = int(height * self.bot)
            crop_pixels_left = int(width * self.left)
            crop_pixels_right = int(width * self.right)

            if len(input_dict[key].shape) == 3:
                input_dict[key] = input_dict[key][crop_pixels_top:height-crop_pixels_bottom,
                                  crop_pixels_left:width-crop_pixels_right, :]
            else:
                input_dict[key] = input_dict[key][crop_pixels_top:height - crop_pixels_bottom,
                                  crop_pixels_left:width - crop_pixels_right]

        return input_dict


@PIPELINES.register_module()
class PerspectiveAug:
    def __init__(self, k, m):
        """
        :param k: camera intrinsic matrix
        :param m: camera extrinsic matrix
        """
        self.k = np.array(k)
        self.m = np.array(m)

    def translate_image(self, image, distance, k, m, border_value):
        """
        :param image: image to transform
        :param distance: positive value for right translation, negative value for left translation
        :param k: intrinsic matrix
        :param m: extrinsic matrix
        :return: translated image
        """
        # translation matrix
        T = np.array([
            [1, 0, distance],
            [0, 1, 0],
            [0, 0, 1]
        ])

        # compute projection & homography matrix
        P = np.delete(np.matmul(k, m), 1, 1)
        H = np.matmul(P, np.matmul(T, np.linalg.inv(P)))

        translated_image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
        return translated_image

    def rotate_image(self, image, angle, k, m, border_value):
        """
        :param image: image to transform
        :param angle: positive value (radians) for CW rotation, negative values (radians) for CCW rotation
        :param k: intrinsic matrix
        :param m: extrinsic matrix
        :return: rotated image
        """
        # rotation matrix
        R = np.array([
            [np.cos(angle), np.sin(angle), 0],
            [-np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        # compute projection & homography matrix
        P = np.delete(np.matmul(k, m), 1, 1)
        H = np.matmul(P, np.matmul(R, np.linalg.inv(P)))

        rotated_image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
        return rotated_image

    def __call__(self, input_dict):
        tx, ry = 0.0, 0.0

        sgnt = 1 if np.random.rand() > 0.5 else -1
        sgnr = 1 if np.random.rand() > 0.5 else -1

        if np.random.rand() < 0.33:
            tx = sgnt * np.random.uniform(0.5, 1.2)
            ry = sgnr * np.random.uniform(0.05, 0.12)
        else:
            if np.random.rand() < 0.5:
                tx = sgnt * np.random.uniform(0.5, 1.5)
            else:
                ry = sgnr * np.random.uniform(0.05, 0.25)

        ry = -ry

        for key in ['img', 'gt_semantic_seg']:
            img = input_dict[key]
            # unique_values = np.unique(img)

            height, width = img.shape[:2]

            k = self.k.copy()
            k[0, :] *= width
            k[1, :] *= height

            m = self.m.copy()
            m = np.linalg.inv(m)[:3, :]

            # transformation object
            # after all transformation, for the old dataset we end up
            border_value = (0, 0, 0) if len(img.shape) == 3 else 0
            output = self.rotate_image(img, ry, k, m, border_value)
            output = self.translate_image(output, tx, k, m, border_value)

            # if key == 'gt_semantic_seg':
            #     diff = np.abs(output.reshape(-1, 1) - unique_values)
            #     indices = np.argmin(diff, axis=1)
            #     output = unique_values[indices].reshape(output.shape)

            input_dict[key] = output

            # print(np.unique(output), input_dict[key].shape)
            # cv2.imshow('img', input_dict[key])
            # cv2.waitKey(0)

        return input_dict


@PIPELINES.register_module()
class Albu:
    """Albumentation augmentation. Adds custom transformations from
    Albumentations library. Please, visit
    `https://albumentations.readthedocs.io` to get more information. An example
    of ``transforms`` is as followed:

    .. code-block::
        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]
    Args:
        transforms (list[dict]): A list of albu transformations
        keymap (dict): Contains {'input key':'albumentation-style key'}
        update_pad_shape (bool): Whether to update padding shape according to \
            the output shape of the last transform
    """

    def __init__(self, transforms, keymap=None, update_pad_shape=False):
        if Compose is None:
            raise ImportError(
                'albumentations is not installed, '
                'we suggest install albumentation by '
                '"pip install albumentations>=0.3.2 --no-binary qudida,albumentations"'  # noqa
            )

        # Args will be modified later, copying it will be safer
        transforms = copy.deepcopy(transforms)

        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape

        self.aug = Compose([self.albu_builder(t) for t in self.transforms])

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
                'gt_masks': 'masks',
            }
        else:
            self.keymap_to_albu = copy.deepcopy(keymap)
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.

        It inherits some of :func:`build_from_cfg` logic.
        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise ImportError(
                    'albumentations is not installed, '
                    'we suggest install albumentation by '
                    '"pip install albumentations>=0.3.2 --no-binary qudida,albumentations"'  # noqa
                )
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper.

        Renames keys according to keymap provided.
        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """

        updated_dict = {}
        for k, _ in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        # Convert to RGB since Albumentations works with RGB images
        results['image'] = cv2.cvtColor(results['image'], cv2.COLOR_BGR2RGB)

        results = self.aug(**results)

        # Convert back to BGR
        results['image'] = cv2.cvtColor(results['image'], cv2.COLOR_RGB2BGR)

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str