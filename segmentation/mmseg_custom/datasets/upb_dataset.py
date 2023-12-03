import bisect
import os
import mmcv
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

from segmentation.dist_utils import build_dataloader

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
    CLASSES = ('rest', 'path')
    PALETTE = [[0, 0, 255], [255, 0, 0]]
    def __init__(self, split, **kwargs):
        super(UPBDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            split=split,
            reduce_zero_label=False,
            **kwargs)

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
                    img_name, euler_pose, _ = line.strip().split(',')
                    euler_pose = float(euler_pose)
                    img_info = dict(filename=img_name)
                    limits = [-float('inf'), *LIMITS, float('inf')]
                    limits_scenarios = [-float('inf'), *LIMITS_SCENARIOS, float('inf')]
                    category = bisect.bisect_right(limits, euler_pose) - 1
                    category_scenarios = bisect.bisect_right(limits_scenarios, euler_pose) - 1
                    if category in categories:
                        categories[category] += 1
                    else:
                        categories[category] = 1
                    if ann_dir is not None:
                        seg_map = img_name
                        img_info['ann'] = dict(seg_map=seg_map, euler_pose=euler_pose, category=category,
                                               curvature=int(euler_pose), scenario_text=category_scenarios)
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