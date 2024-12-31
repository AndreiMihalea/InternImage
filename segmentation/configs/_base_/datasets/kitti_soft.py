import os
import platform
from segmentation.configs._base_.ego_traj_generic_params import (num_classes_path, flip_prob, annotations_loader,
                                                                 dataset_name)


hostname = platform.node()
# dataset settings
dataset_type = 'UPBDataset'
if 'nemodrive1' in hostname:
    storage_path = '/mnt/datadisk/andreim'
elif 'nemodrive0' in hostname:
    storage_path = '/mnt/storage/workspace/andreim/nemodrive'
else:
    storage_path = '/raid/andreim'
if dataset_name == 'upb':
    relative_dataset_path = 'upb_data/segmentation'
elif dataset_name == 'kitti':
    relative_dataset_path = 'kitti/data_odometry_color/segmentation'
else:
    relative_dataset_path = ''
data_root = os.path.join(storage_path, relative_dataset_path)
img_norm_cfg = dict(
    mean=[89.497, 93.675, 92.645], std=[76.422, 78.611, 80.487], to_rgb=True)
if dataset_name == 'kitti':
    crop_size = (200, 664)
else:
    crop_size = (288, 640)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type=annotations_loader, reduce_zero_label=False),
    dict(type='LoadCategory'),
    # dict(type='PerspectiveAug', k=[[0.61, 0, 0.5], [0, 1.36, 0.5], [0, 0, 1]],
    #      m=[[1, 0, 0, 0.00], [0, 1, 0, 1.65], [0, 0, 1, 1.54], [0, 0, 0, 1]]),
    dict(type='Resize', img_scale=(664, 200), ratio_range=None),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=flip_prob),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='ToMask', soft_output=True, num_classes=num_classes_path),
    dict(type='ToSoft', num_iter=12, kernel_size=(11, 11), std_dev=5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'gt_masks', 'gt_labels', 'category', 'category_for_balancing',
                               'curvature', 'scenario_text'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsSplitByCategory', reduce_zero_label=False),
    dict(type='LoadCategory'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', prob=flip_prob),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToMask', soft_output=True, num_classes=num_classes_path),
            dict(type='ToSoft', num_iter=12, kernel_size=(11, 11), std_dev=5, soft_output=True),
            dict(type='Collect', keys=['img', 'category', 'category_for_balancing', 'curvature', 'scenario_text']),
        ])
]
inference_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotationsSplitByCategory', reduce_zero_label=False),
    dict(type='LoadCategory'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            # dict(type='ToMask'),
            # dict(type='ToSoft', num_iter=12, kernel_size=(11, 11), std_dev=5),
            dict(type='Collect', keys=['img', 'category', 'curvature', 'scenario_text']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='self_supervised_labels_30',
        split='splits/val_30.txt',
        pipeline=train_pipeline,
        soft_output=True,
        num_classes=num_classes_path),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='self_supervised_labels_30',
        split='splits/test_30.txt',
        pipeline=test_pipeline,
        soft_output=True,
        num_classes=num_classes_path),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='self_supervised_labels_30',
        split='splits/test_30.txt',
        pipeline=test_pipeline,
        soft_output=True,
        num_classes=num_classes_path),
    inference=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='self_supervised_labels_30',
        split='splits/test_30.txt',
        pipeline=inference_pipeline,
        soft_output=True,
        num_classes=num_classes_path))