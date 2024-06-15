import bisect
import os
from collections import OrderedDict
from copy import deepcopy

import mmcv
import numpy as np
from PIL import Image, ImageFile
from mmseg.core import intersect_and_union
from prettytable import PrettyTable
from tqdm import tqdm

from mmseg_custom.datasets.pipelines.formatting import ToSoft, ToMask
from segmentation.dist_utils import build_dataloader
from ..core.evaluation.metrics import pre_eval_to_metrics, eval_metrics, jaccard_metric

ImageFile.LOAD_TRUNCATED_IMAGES = True
from mmcv import print_log
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.utils import get_root_logger


LIMITS = [-80, -40, 0, 40, 80]
LIMITS_SCENARIOS = [-60, -18, 18, 60]
TURNING_SCENARIOS = ["TIGHT LEFT", "SLIGHT LEFT", "FORWARD", "SLIGHT RIGHT", "TIGHT RIGHT"]


@DATASETS.register_module()
class UPBDataset(CustomDataset):
    CLASSES = ('rest', 'tight left', 'slight left', 'forward', 'slight right', 'tight right')
    # PALETTE = [[0, 0, 255], [255, 0, 0]]
    def __init__(self, split, soft_output=False, **kwargs):
        super(UPBDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            split=split,
            reduce_zero_label=False,
            **kwargs)

        self.soft_output = soft_output

        if self.test_mode:
            for step in kwargs['pipeline']:
                if 'transforms' in step:
                    for transform in step['transforms']:
                        if transform['type'] == 'ToMask':
                            mask_transform = deepcopy(transform)
                            del mask_transform['type']
                            self.to_mask = ToMask(**mask_transform)
                        elif transform['type'] == 'ToSoft':
                            soft_transform = deepcopy(transform)
                            del soft_transform['type']
                            self.to_soft = ToSoft(**soft_transform)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        categories = dict()
        if split is not None:
            nr = 0
            with open(split) as f:
                for line in f:
                    nr += 1
                    line_split = line.strip().split(',')
                    # category is in file (TODO: should probably remove this)
                    if len(line_split) == 3:
                        img_name, euler_pose, _ = line_split
                    else:
                        img_name, euler_pose = line_split
                    euler_pose = float(euler_pose)
                    img_info = dict(filename=img_name)
                    limits = [-float('inf'), *LIMITS, float('inf')]
                    limits_scenarios = [-float('inf'), *LIMITS_SCENARIOS, float('inf')]
                    balancing_category = bisect.bisect_right(limits, euler_pose) - 1
                    scenario_category = bisect.bisect_right(limits_scenarios, euler_pose) - 1
                    if balancing_category in categories:
                        categories[balancing_category] += 1
                    else:
                        categories[balancing_category] = 1
                    if ann_dir is not None:
                        seg_map = img_name
                        img_info['ann'] = dict(seg_map=seg_map,
                                               euler_pose=euler_pose,
                                               category=scenario_category,
                                               category_for_balancing=balancing_category,
                                               curvature=float(euler_pose),
                                               scenario_text=scenario_category)
                    img_infos.append(img_info)
                img_infos = sorted(img_infos, key=lambda x: x['filename'])
            print(categories, sum(categories.values()), nr)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_train_img(idx)
        else:
            return self.prepare_train_img(idx)

    def get_gt_mask_by_idx(self, index):
        """Get one ground truth mask for evaluation"""
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        return self.to_soft(self.to_mask(results))['gt_masks']

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """

        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        for it, (pred, index) in enumerate(zip(preds, indices)):
            if self.soft_output:
                soft_pred = pred
                mask = self.get_gt_mask_by_idx(index)
                jm = jaccard_metric(soft_pred, mask)
                pre_eval_results.append(jm)
            else:
                seg_map = self.get_gt_seg_map_by_idx(index)
                i_and_u = intersect_and_union(
                        pred,
                        seg_map,
                        len(self.CLASSES),
                        self.ignore_index,
                        # as the labels has been converted when dataset initialized
                        # in `get_palette_for_custom_classes ` this `label_map`
                        # should be `dict()`, see
                        # https://github.com/open-mmlab/mmsegmentation/issues/1415
                        # for more ditails
                        label_map=dict(),
                        reduce_zero_label=self.reduce_zero_label)
                pre_eval_results.append(i_and_u)

        return pre_eval_results

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'mIoU_soft']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=dict(),
                reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        return eval_results


def compute_mean_std(data_path):
    means = np.array([0.0, 0.0, 0.0])
    stds = np.array([0.0, 0.0, 0.0])

    count = 0

    for filename in tqdm(os.listdir(data_path)):
        count += 1
        img = Image.open(os.path.join(data_path, filename))
        # img = img.resize((1226, 370))
        img_arr = np.array(img)
        means += img_arr.mean(axis=(0, 1))
        stds += img_arr.std(axis=(0, 1))

    means /= count
    stds /= count

    return np.round(means, 3), np.round(stds, 3)


def main():
    dataset_type = 'UPBDataset'
    data_root = '/raid/andreim/kitti/data_odometry_color/segmentation'
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    crop_size = (512, 512)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=True),
        dict(type='LoadCategory'),
        # dict(type='PerspectiveAug', k=[[0.61, 0, 0.5], [0, 1.09, 0.5], [0, 0, 1]],
        #      m=[[1, 0, 0, 0.00], [0, 1, 0, 1.65], [0, 0, 1, 1.54], [0, 0, 0, 1]]),
        dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg', 'category']),
    ]
    dataset = UPBDataset(
        data_root=data_root,
        img_dir='images',
        ann_dir='self_supervised_labels',
        split='splits/val.txt',
        pipeline=train_pipeline)

    dataloader = build_dataloader(dataset, 4, 4, 4, dist=True, seed=42, drop_last=True)
    categories = dict()
    for i, el in enumerate(dataloader):
        if i % 100 == 0:
            print(i)
            for cat in el['category']:
                print(cat.item())
        for cat in el['category']:
            cat = cat.item()
            if cat in categories:
                categories[cat] += 1
            else:
                categories[cat] = 1
    print(categories)


if __name__ == '__main__':
    data_path = '/raid/andreim/nemodrive/upb_data/segmentation/images'
    from mmcv.utils import Registry
    # print(DATASETS.module_dict)
    # print(compute_mean_std(data_path))
    main()