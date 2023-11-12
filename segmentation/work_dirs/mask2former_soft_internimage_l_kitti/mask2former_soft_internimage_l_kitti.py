num_things_classes = 0
num_stuff_classes = 2
num_classes = 2
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoderMask2Former',
    pretrained=None,
    backbone=dict(
        type='InternImage',
        core_op='DCNv3',
        channels=160,
        depths=[5, 5, 22, 5],
        groups=[10, 20, 40, 80],
        mlp_ratio=4.0,
        drop_path_rate=0.4,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=2.0,
        post_norm=True,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_l_22k_192to384.pth'
        )),
    decode_head=dict(
        type='Mask2FormerSoftHead',
        in_channels=[160, 320, 640, 1280],
        feat_channels=1024,
        out_channels=1024,
        in_index=[0, 1, 2, 3],
        num_things_classes=0,
        num_stuff_classes=2,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=1024,
                        num_heads=32,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=1024,
                        feedforward_channels=4096,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True),
                        with_cp=True),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=512, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=512, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=1024,
                    num_heads=32,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=1024,
                    feedforward_channels=4096,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True,
                    with_cp=True),
                feedforward_channels=4096,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0, 1.0, 0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        loss_soft=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0)),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='MaskHungarianAssignerSoft',
            cls_cost=dict(type='ClassificationCost', weight=2.0),
            mask_cost=dict(
                type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
            dice_cost=dict(
                type='DiceCost', weight=5.0, pred_act=True, eps=1.0),
            soft_cost=dict(
                type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True)),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=True,
        semantic_on=False,
        instance_on=True,
        max_per_image=100,
        iou_thr=0.8,
        filter_low_score=True,
        mode='slide',
        crop_size=(200, 664),
        stride=(426, 426)),
    init_cfg=None)
dataset_type = 'UPBDataset'
data_root = '/mnt/datadisk/andreim/kitti/data_odometry_color/segmentation'
img_norm_cfg = dict(
    mean=[89.497, 93.675, 92.645], std=[76.422, 78.611, 80.487], to_rgb=True)
crop_size = (200, 664)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='LoadCategory'),
    dict(type='Resize', img_scale=(664, 200), ratio_range=None),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[89.497, 93.675, 92.645],
        std=[76.422, 78.611, 80.487],
        to_rgb=True),
    dict(type='Pad', size=(200, 664), pad_val=0, seg_pad_val=0),
    dict(type='ToMask'),
    dict(type='ToSoft', num_iter=12, kernel_size=(11, 11), std_dev=5),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg', 'gt_masks', 'gt_labels', 'category'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(200, 664),
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
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UPBDataset',
        data_root=
        '/mnt/datadisk/andreim/kitti/data_odometry_color/segmentation',
        img_dir='images',
        ann_dir='self_supervised_labels_30',
        split='splits/val_30.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='LoadCategory'),
            dict(type='Resize', img_scale=(664, 200), ratio_range=None),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[89.497, 93.675, 92.645],
                std=[76.422, 78.611, 80.487],
                to_rgb=True),
            dict(type='Pad', size=(200, 664), pad_val=0, seg_pad_val=0),
            dict(type='ToMask'),
            dict(type='ToSoft', num_iter=12, kernel_size=(11, 11), std_dev=5),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_semantic_seg', 'gt_masks', 'gt_labels',
                    'category'
                ])
        ]),
    val=dict(
        type='UPBDataset',
        data_root=
        '/mnt/datadisk/andreim/kitti/data_odometry_color/segmentation',
        img_dir='images',
        ann_dir=
        '/mnt/datadisk/andreim/kitti/data_odometry_color/segmentation_gt/self_supervised_labels_30',
        split='splits/test_30.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(200, 664),
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
        '/mnt/datadisk/andreim/kitti/data_odometry_color/segmentation',
        img_dir='images',
        ann_dir=
        '/mnt/datadisk/andreim/kitti/data_odometry_color/segmentation_gt/self_supervised_labels_30',
        split='splits/test_30.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(200, 664),
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
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(
    interval=16000, metric='mIoU', pre_eval=True, save_best='mIoU')
pretrained = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_l_22k_192to384.pth'
work_dir = 'work_dirs/mask2former_soft_internimage_l_kitti'
gpu_ids = range(0, 2)
auto_resume = False
