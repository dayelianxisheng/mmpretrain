# VOC2012 数据集配置文件 - batch_size=64
# 参考 configs/_base_/datasets/voc_bs16.py
# 参考 configs/_base_/datasets/voc_bs16.py

dataset_type = 'VOC'
data_preprocessor = dict(
    num_classes=20,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
    to_onehot=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, crop_ratio_range=(0.8, 1.0)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='PackInputs',
        meta_keys=('sample_idx', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'gt_label_difficult')),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='data/VOC2012/VOCdevkit/VOC2012',
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='data/VOC2012/VOCdevkit/VOC2012',
        split='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = [
    dict(type='VOCMultiLabelMetric'),
    dict(type='VOCMultiLabelMetric', average='micro'),
    dict(type='VOCAveragePrecision')
]

test_dataloader = val_dataloader
test_evaluator = val_evaluator
