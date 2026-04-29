auto_scale_lr = dict(base_batch_size=64)
custom_hooks = [
    dict(
        freeze_epochs=5,
        freeze_layers=[
            'backbone',
        ],
        priority='ABOVE_NORMAL',
        type='FreezeLayersHook',
        unfreeze_backbone=True),
    dict(momentum=0.0002, priority='ABOVE_NORMAL', type='EMAHook'),
]
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=20,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_onehot=True,
    to_rgb=True)
dataset_type = 'VOC'
default_hooks = dict(
    checkpoint=dict(
        interval=10,
        max_keep_ckpts=3,
        rule='greater',
        save_best='multi-label/mAP',
        type='CheckpointHook'),
    early_stopping=dict(
        monitor='multi-label/mAP',
        patience=15,
        rule='greater',
        type='EarlyStoppingHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(
        cbam_ratio=16,
        frozen_stages=-1,
        init_cfg=dict(
            checkpoint=
            'my/checkpoints/backbone/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth',
            prefix='backbone',
            type='Pretrained'),
        kernel_size=3,
        out_indices=(7, ),
        type='CBAMMobileNetV2',
        widen_factor=1.0),
    data_preprocessor=dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        num_classes=20,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_onehot=False,
        to_rgb=True),
    head=dict(
        in_channels=1280,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        num_classes=20,
        topk=1,
        type='MultiLabelLinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            cbam=dict(lr_mult=1.0),
            head=dict(lr_mult=1.0))))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=10,
        start_factor=0.01,
        type='LinearLR'),
    dict(
        T_max=190,
        begin=10,
        by_epoch=True,
        convert_to_iter_based=True,
        end=200,
        eta_min=1e-06,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=42)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data/VOC2012/VOCdevkit/VOC2012',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(
                meta_keys=(
                    'sample_idx',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'gt_label_difficult',
                ),
                type='PackInputs'),
        ],
        split='val',
        type='VOC'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(type='VOCMultiLabelMetric'),
    dict(average='micro', type='VOCMultiLabelMetric'),
    dict(type='VOCAveragePrecision'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(edge='short', scale=256, type='ResizeEdge'),
    dict(crop_size=224, type='CenterCrop'),
    dict(
        meta_keys=(
            'sample_idx',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'gt_label_difficult',
        ),
        type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
train_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data/VOC2012/VOCdevkit/VOC2012',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                transforms=[
                    dict(
                        p=0.5,
                        rotate_limit=15,
                        scale_limit=0.2,
                        shift_limit=0.1,
                        type='ShiftScaleRotate'),
                    dict(
                        brightness_limit=0.3,
                        contrast_limit=0.3,
                        p=0.5,
                        type='RandomBrightnessContrast'),
                    dict(
                        max_height=40,
                        max_holes=6,
                        max_width=40,
                        p=0.3,
                        type='CoarseDropout'),
                ],
                type='Albu'),
            dict(
                crop_ratio_range=(
                    0.7,
                    1.0,
                ),
                scale=224,
                type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        split='train',
        type='VOC'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        transforms=[
            dict(
                p=0.5,
                rotate_limit=15,
                scale_limit=0.2,
                shift_limit=0.1,
                type='ShiftScaleRotate'),
            dict(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5,
                type='RandomBrightnessContrast'),
            dict(
                max_height=40,
                max_holes=6,
                max_width=40,
                p=0.3,
                type='CoarseDropout'),
        ],
        type='Albu'),
    dict(crop_ratio_range=(
        0.7,
        1.0,
    ), scale=224, type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data/VOC2012/VOCdevkit/VOC2012',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(
                meta_keys=(
                    'sample_idx',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'gt_label_difficult',
                ),
                type='PackInputs'),
        ],
        split='val',
        type='VOC'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(type='VOCMultiLabelMetric'),
    dict(average='micro', type='VOCMultiLabelMetric'),
    dict(type='VOCAveragePrecision'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/cbam_mobilenetv2_voc'
