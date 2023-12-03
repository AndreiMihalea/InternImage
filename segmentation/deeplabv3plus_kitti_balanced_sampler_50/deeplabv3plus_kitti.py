num_things_classes = 0
num_stuff_classes = 2
num_classes = 2
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='FocalLoss', use_sigmoid=True, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='FocalLoss', use_sigmoid=True, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'UPBDataset'
data_root = '/raid/andreim//kitti/data_odometry_color/segmentation'
img_norm_cfg = dict(
    mean=[89.497, 93.675, 92.645], std=[76.422, 78.611, 80.487], to_rgb=True)
crop_size = (193, 640)
img_scale = (193, 640)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='LoadCategory'),
    dict(
        type='PerspectiveAug',
        k=[[0.58, 0, 0.5], [0, 1.92, 0.5], [0, 0, 1]],
        m=[[1, 0, 0, 0.0], [0, 1, 0, 1.68], [0, 0, 1, 1.65], [0, 0, 0, 1]]),
    dict(type='Resize', img_scale=(193, 640), ratio_range=None),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[89.497, 93.675, 92.645],
        std=[76.422, 78.611, 80.487],
        to_rgb=True),
    dict(type='Pad', size=(193, 640), pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(193, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[89.497, 93.675, 92.645],
                std=[76.422, 78.611, 80.487],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=10,
    workers_per_gpu=4,
    train=dict(
        type='UPBDataset',
        data_root=
        '/raid/andreim//kitti/data_odometry_color/segmentation',
        img_dir='images',
        ann_dir='self_supervised_labels_30',
        split='splits/val_30.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='LoadCategory'),
            dict(
                type='PerspectiveAug',
                k=[[0.58, 0, 0.5], [0, 1.92, 0.5], [0, 0, 1]],
                m=[[1, 0, 0, 0.0], [0, 1, 0, 1.68], [0, 0, 1, 1.65],
                   [0, 0, 0, 1]]),
            dict(type='Resize', img_scale=(193, 640), ratio_range=None),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[89.497, 93.675, 92.645],
                std=[76.422, 78.611, 80.487],
                to_rgb=True),
            dict(type='Pad', size=(193, 640), pad_val=0, seg_pad_val=0),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='UPBDataset',
        data_root=
        '/raid/andreim//kitti/data_odometry_color/segmentation',
        img_dir='images',
        ann_dir=
        '/raid/andreim//kitti/data_odometry_color/segmentation_gt/self_supervised_labels_30',
        split='splits/test_30.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(193, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[89.497, 93.675, 92.645],
                        std=[76.422, 78.611, 80.487],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='UPBDataset',
        data_root=
        '/raid/andreim//kitti/data_odometry_color/segmentation',
        img_dir='images',
        ann_dir=
        '/raid/andreim//kitti/data_odometry_color/segmentation_gt/self_supervised_labels_30',
        split='splits/test_30.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(193, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[89.497, 93.675, 92.645],
                        std=[76.422, 78.611, 80.487],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=2e-05,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=39,
        layer_decay_rate=0.94,
        depths=[5, 5, 24, 5],
        offset_lr_scale=1.0))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=400,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=64000)
checkpoint_config = dict(by_epoch=False, interval=200, max_keep_ckpts=1)
evaluation = dict(
    interval=2000, metric='mIoU', pre_eval=True, save_best='mIoU')
work_dir = 'work_dirs/deeplabv3plus_kitti_balanced_sampler_50'
gpu_ids = range(0, 2)
auto_resume = False
