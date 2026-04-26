
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
    # Albumentations 库增强
    dict(type='Albu',
         transforms=[
             dict(type='ShiftScaleRotate', shift_limit=0.1, scale_limit=0.2,
                  rotate_limit=15, p=0.5),
             dict(type='RandomBrightnessContrast', brightness_limit=0.3,
                  contrast_limit=0.3, p=0.5),
             dict(type='CoarseDropout', max_holes=6, max_height=40,
                  max_width=40, p=0.3),
         ],
         ),
    # 基础几何变换
    dict(type='RandomResizedCrop', scale=224, crop_ratio_range=(0.7, 1.0)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
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
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='data/VOC2012/VOCdevkit/VOC2012',
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
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
