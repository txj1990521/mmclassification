_base_ = [
    '/home/txj/mmclassification/configs/_base_/models/resnet18_cifar.py',
    '/home/txj/mmclassification/config2/_base_/datasets/base_side_datasets.py',
    '/home/txj/mmclassification/configs/_base_/schedules/cifar10_bs128_数字测试.py',
    '/home/txj/mmclassification/configs/_base_/default_runtime.py'
]
dataset_path = '/mnt/AIData/txj/data/十分区数据/'
dataset_path_list = [f'{dataset_path}']
label_path = dataset_path + '/label.ini'
checkpoint_config = dict(interval=10)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
data = dict(
    persistent_workers=False,
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        label_path=label_path,
        dataset_path_list=dataset_path_list,
        data_prefix=dataset_path_list,
    ),
    val=dict(
        label_path=label_path,
        dataset_path_list=dataset_path_list,
        data_prefix=dataset_path_list,
    ),
    test=dict(
        label_path=label_path,
        dataset_path_list=dataset_path_list,
        data_prefix=dataset_path_list,
    )
)
evaluation = dict(interval=1, metric='accuracy')
lr_config = dict(policy='step', step=[120, 170])
runner = dict(type='EpochBasedRunner', max_epochs=200)
# optimizer
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[100, 150])
runner = dict(type='EpochBasedRunner', max_epochs=200)
