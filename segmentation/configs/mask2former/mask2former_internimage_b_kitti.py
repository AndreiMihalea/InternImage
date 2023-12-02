# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    '../_base_/models/mask2former.py', '../_base_/datasets/kitti.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
num_things_classes = 0
num_stuff_classes = 2
num_classes = num_things_classes + num_stuff_classes
crop_size = (200, 664)
pretrained = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_b_1k_224.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=112,
        depths=[4, 4, 21, 4],
        groups=[7, 14, 28, 56],
        mlp_ratio=4.,
        drop_path_rate=0.4,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=True,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    additional_input=None,
    decode_head=dict(
        type='Mask2FormerHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        in_channels=[112, 224, 448, 896],
        feat_channels=256,
        out_channels=256,
        num_queries=100,
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
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        with_cp=True,  # set with_cp=True to save memory
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    with_cp=True,  # set with_cp=True to save memory
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
    ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

optimizer = dict(
    _delete_=True, type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.05,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=39, layer_decay_rate=0.94,
                       depths=[5, 5, 24, 5], offset_lr_scale=1.0))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2, workers_per_gpu=4)
runner = dict(type='IterBasedRunner')
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=16000, metric='mIoU', save_best='mIoU')
# fp16 = dict(loss_scale=dict(init_scale=512))
