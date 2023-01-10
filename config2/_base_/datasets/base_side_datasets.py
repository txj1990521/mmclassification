# -*- coding: utf-8 -*-
custom_imports = dict(
    imports=['config2._base_.datasets.class_dataset',
             'side_ai.pipelines.init_pipelines',
             'side_ai.side_after_run_hook',
             'side_ai.side_epoch_based_runner',
             'side_ai.pipelines.save_pipeline',
             'side_ai.pipelines.after_run_pipeline'], allow_failed_imports=True)
dataset_type = 'ClassDataset'
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # The category names of your dataset

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3,),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=31,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

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
train_init_pipeline = [
    dict(type='CopyData2Local', target_dir='/home/txj/data/公共数据缓存', run_rsync=True),
    dict(type='LoadCategoryList', ignore_labels=['屏蔽']),
    dict(type='LoadPathList'),
    dict(type='SplitData', start=0, end=0.75, key='path_set'),  # 分割数据集，用于训练集和验证集
    dict(type='StatCategoryCounter'),
    dict(type='GenerateMmclsAnn'),
]

test_init_pipeline = [
    dict(type='CopyData2Local', target_dir='/home/txj/data/公共数据缓存', run_rsync=True),
    dict(type='LoadCategoryList', ignore_labels=['屏蔽']),
    dict(type='LoadPathList'),
    dict(type='SplitData', start=0.75, end=1, key='json_path_list'),
    dict(type='StatCategoryCounter'),
    dict(type='GenerateMmclsAnn'),
]
data = dict(
    train=dict(
        init_pipeline=train_init_pipeline,
        type=dataset_type,
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        init_pipeline=test_init_pipeline,
        type=dataset_type,
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        init_pipeline=test_init_pipeline,
        type=dataset_type,
        classes=classes,
        pipeline=test_pipeline
    )
)
evaluation = dict(interval=1, metric='accuracy')