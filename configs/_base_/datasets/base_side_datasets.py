dataset_type = 'MyDataset'
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # The category names of your dataset

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    train=dict(
        type=dataset_type,
        data_prefix='D:/data/MNIST/mnista_data/train',
        ann_file='D:/data/MNIST/mnista_data/meta/train.txt',
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix='D:/data/MNIST/mnista_data/val',
        ann_file='D:/data/MNIST/mnista_data/meta/val.txt',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_prefix='D:/data/MNIST/mnista_data/test',
        ann_file='D:/data/MNIST/mnista_data/meta/test.txt',
        classes=classes,
        pipeline=test_pipeline
    )
)
evaluation = dict(interval=1, metric='accuracy')