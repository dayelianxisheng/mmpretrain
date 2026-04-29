auto_scale_lr = dict(base_batch_size=224)
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=100,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
dataset_type = 'CustomDataset'
default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, type='CheckpointHook'),
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
        arch='b',
        drop_rate=0.1,
        img_size=224,
        init_cfg=[
            dict(
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear',
                type='Kaiming'),
        ],
        patch_size=16,
        type='CAViT'),
    head=dict(
        in_channels=768,
        loss=dict(
            label_smooth_val=0.1, mode='classy_vision',
            type='LabelSmoothLoss'),
        num_classes=100,
        type='VisionTransformerClsHead'),
    neck=None,
    type='ImageClassifier')
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0005,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0.0))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        type='LinearLR'),
    dict(
        T_max=95,
        begin=5,
        convert_to_iter_based=True,
        end=100,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=None)
resume = True
test_cfg = dict()
test_dataloader = dict(
    batch_size=24,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='meta\\val.txt',
        data_prefix='val',
        data_root='E:\\dataset\\flower100_mmp_r50_tv80',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(edge='short', scale=512, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset',
        with_label=True),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(edge='short', scale=512, type='ResizeEdge'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
train_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='meta\\train.txt',
        data_prefix='train',
        data_root='E:\\dataset\\flower100_mmp_r50_tv80',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=224, type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset',
        with_label=True),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=224, type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=24,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='meta\\val.txt',
        data_prefix='val',
        data_root='E:\\dataset\\flower100_mmp_r50_tv80',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(edge='short', scale=512, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset',
        with_label=True),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
warmup_epochs = 3
work_dir = './work_dirs\\ca_vit_flower100'
