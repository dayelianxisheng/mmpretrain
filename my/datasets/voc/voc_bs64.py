dataset_type = 'VOC'
data_preprocessor = dict(
    num_classes=20,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
    to_onehot=True,  # 多标签需要 one-hot 格式供 MultiLabelClsHead 使用
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
             dict(type='CoarseDropout', num_holes_range=(1, 6),
                  hole_height_range=(1, 40), hole_width_range=(1, 40), p=0.3),
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
    batch_size=64,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/VOC2012/VOCdevkit/VOC2012',
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=2,
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
