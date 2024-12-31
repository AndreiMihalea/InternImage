# dataset settings
dataset_type = 'UPBDataset'
data_root = '/raid/andreim/kitti/data_odometry_color/segmentation'
img_norm_cfg = dict(
    mean=[89.497, 93.675, 92.645], std=[76.422, 78.611, 80.487], to_rgb=True)
crop_size = (193, 640)
img_scale = (193, 640)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='LoadCategory'),
    # dict(type='CustomCrop', left=0, right=0, top=0.45, bot=0),
    dict(type='PerspectiveAug', k=[[0.58, 0, 0.5],
                           [0, 1.92, 0.5],
                           [0, 0, 1]],
         m=[[1, 0, 0, 0.00], [0, 1, 0, 1.68], [0, 0, 1, 1.65], [0, 0, 0, 1]]),
    dict(type='Resize', img_scale=img_scale, ratio_range=None),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'category'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='CustomCrop', left=0, right=0, top=0.45, bot=0),
            # dict(type='DefaultFormatBundle'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
inference_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='CustomCrop', left=0, right=0, top=0.45, bot=0),
            # dict(type='DefaultFormatBundle'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=26,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='self_supervised_labels_30',
        split='splits/val_30.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='/raid/andreim/kitti/data_odometry_color/segmentation/self_supervised_labels_30',
        split='splits/test_30.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='/raid/andreim/kitti/data_odometry_color/segmentation/self_supervised_labels_30',
        split='splits/test_30.txt',
        pipeline=test_pipeline),
    inference=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='/raid/andreim/kitti/data_odometry_color/segmentation/self_supervised_labels_30',
        split='splits/test_30.txt',
        pipeline=inference_pipeline))
